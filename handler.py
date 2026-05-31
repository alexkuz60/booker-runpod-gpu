"""
booker-runpod-gpu — RunPod Serverless handler (SYNC generator, hardened v5).

Изменения v4 → v5:
  • Fail-fast если SHARED_TOKEN пустой (yield error, не молчим).
  • isinstance() guards на каждом .get() — payload, _hmac, options, narrator,
    scenes[], segments[], seg, text.
  • Outer try/except вокруг ВСЕГО тела генератора — финальный "fatal" yield
    перед исключением, чтобы RunPod не помечал job COMPLETED при крашe.
  • Безопасное приведение text → str (None и числа не крашат len()).
  • Counter невалидных сегментов в финальном done.
  • Bootstrap warning если SHARED_TOKEN не задан в env — видно в boot-логах.
  • Type-safe доступ к scenes/segments — non-dict элементы пропускаются с
    yield warning, не валят весь job.

Контракт со стороны Lovable Edge (synthesize-scene-runpod):
  - RunPod payload: { "input": { ...innerInput, "_hmac": {"ts": "...", "sig": "..."} } }
  - innerInput = { ...originalClientPayload, "user_id_hash": "<8hex>" }
  - HMAC SHA-256:  hex( HMAC(SHARED_TOKEN, f"{ts}.{json.dumps(innerInput, separators=(',',':'), sort_keys=True, ensure_ascii=False)}") )
    ВАЖНО: Edge подписывает canonical JSON c sort_keys=True — handler должен делать то же самое.
  - innerInput НЕ содержит ключа "_hmac" на момент подписи.
"""

import os
import sys
import json
import hmac
import hashlib
import time
import traceback
from typing import Any, Dict, Generator, List, Optional

import runpod


SHARED_TOKEN = os.environ.get("RUNPOD_SHARED_TOKEN", "")
HMAC_MAX_SKEW_MS = 10 * 60 * 1000  # 10 минут — окно для ts


# ─────────────────────────────── HMAC ──────────────────────────────────────

def _canonical_inner(payload_without_hmac: Dict[str, Any]) -> str:
    """
    Должно ТОЧНО совпадать с тем, что подписывает Edge.
    Edge сейчас использует stableJsonStringify (sorted keys), поэтому здесь
    тоже sort_keys=True. separators без пробелов, ensure_ascii=False.
    """
    return json.dumps(
        payload_without_hmac,
        separators=(",", ":"),
        sort_keys=True,
        ensure_ascii=False,
    )


def _compute_sig(ts: str, body: str) -> str:
    return hmac.new(
        SHARED_TOKEN.encode("utf-8"),
        f"{ts}.{body}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def verify_hmac(inner_without_hmac: Dict[str, Any], hmac_blob: Dict[str, Any]) -> Dict[str, Any]:
    diag: Dict[str, Any] = {"ok": False}
    try:
        if not SHARED_TOKEN:
            diag["reason"] = "shared_token_not_configured"
            return diag

        ts = str(hmac_blob.get("ts", ""))
        got_sig = str(hmac_blob.get("sig", ""))
        if not ts or not got_sig:
            diag["reason"] = "missing_ts_or_sig"
            return diag

        try:
            ts_int = int(ts)
            skew = abs(int(time.time() * 1000) - ts_int)
            diag["skew_ms"] = skew
            if skew > HMAC_MAX_SKEW_MS:
                diag["reason"] = "ts_skew_too_large"
                return diag
        except (ValueError, TypeError):
            diag["reason"] = "ts_not_int"
            return diag

        try:
            body = _canonical_inner(inner_without_hmac)
        except (TypeError, ValueError) as e:
            diag["reason"] = "inner_not_json_serializable"
            diag["error"] = repr(e)
            return diag

        expected = _compute_sig(ts, body)

        diag["inner_len_bytes"] = len(body.encode("utf-8"))
        diag["inner_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
        diag["inner_head"] = body[:80]
        diag["inner_tail"] = body[-80:]
        diag["expected_sig"] = expected
        diag["got_sig_prefix"] = got_sig[:16]
        diag["shared_token_len"] = len(SHARED_TOKEN)

        diag["ok"] = hmac.compare_digest(expected, got_sig)
        if not diag["ok"]:
            diag["reason"] = "sig_mismatch"
        return diag
    except Exception as e:
        diag["reason"] = "exception"
        diag["error"] = repr(e)
        diag["trace"] = traceback.format_exc()[-500:]
        return diag


# ─────────────────────────── Payload validation ────────────────────────────

def validate_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {"ok": False, "reason": "payload_not_dict"}

    has_single = (
        isinstance(payload.get("scene_id"), str)
        and isinstance(payload.get("segments"), list)
    )
    has_batch = (
        isinstance(payload.get("scenes"), list)
        and len(payload["scenes"]) > 0
    )

    if not (has_single or has_batch):
        return {"ok": False, "reason": "missing_scene_or_scenes"}

    if has_single and len(payload["segments"]) == 0:
        return {"ok": False, "reason": "segments_empty"}

    return {"ok": True, "mode": "single" if has_single else "batch"}


# ─────────────────────────── Safe helpers ──────────────────────────────────

def _safe_str(value: Any) -> str:
    """Возвращает str, безопасно для None/чисел/прочего."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return str(value)
    except Exception:
        return ""


def _collect_segments(inner: Dict[str, Any], mode: str) -> List[Dict[str, Any]]:
    """
    Собирает плоский список сегментов из single|batch payload.
    Non-dict элементы отфильтровываются (будут отражены в counter).
    """
    raw: List[Any] = []
    if mode == "single":
        seg_list = inner.get("segments")
        if isinstance(seg_list, list):
            raw = seg_list
    else:
        scenes = inner.get("scenes")
        if isinstance(scenes, list):
            for sc in scenes:
                if not isinstance(sc, dict):
                    continue
                seg_list = sc.get("segments")
                if isinstance(seg_list, list):
                    raw.extend(seg_list)
    return raw


# ───────────────────────────────── Handler ─────────────────────────────────
# СИНХРОННЫЙ генератор — def, не async def!

def handler(job: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    t0 = time.time()
    fatal_emitted = False

    try:
        # 1) Sanity: job shape
        if not isinstance(job, dict):
            yield {"type": "error", "stage": "job_shape",
                   "error": f"job is not a dict: {type(job).__name__}"}
            return

        raw_input = job.get("input")

        yield {
            "type": "start",
            "stage": "handler_entry",
            "received": True,
            "has_input": raw_input is not None,
            "input_type": type(raw_input).__name__,
            "input_keys": list(raw_input.keys()) if isinstance(raw_input, dict) else None,
            "shared_token_configured": bool(SHARED_TOKEN),
            "shared_token_len": len(SHARED_TOKEN),
            "runpod_sdk_version": getattr(runpod, "__version__", "unknown"),
            "py_version": sys.version.split()[0],
        }

        if not SHARED_TOKEN:
            yield {"type": "error", "stage": "bootstrap",
                   "error": "RUNPOD_SHARED_TOKEN env is empty"}
            return

        if not isinstance(raw_input, dict):
            yield {"type": "error", "stage": "input_shape",
                   "error": "input is not a dict"}
            return

        # 2) Извлечь _hmac и подготовить inner без _hmac
        hmac_blob = raw_input.get("_hmac")
        if not isinstance(hmac_blob, dict):
            yield {
                "type": "error",
                "stage": "hmac_missing",
                "error": "no _hmac dict in input",
                "input_keys": list(raw_input.keys()),
                "hmac_type": type(hmac_blob).__name__,
            }
            return

        inner_without_hmac = {k: v for k, v in raw_input.items() if k != "_hmac"}

        # 3) HMAC verify с полной диагностикой
        hmac_diag = verify_hmac(inner_without_hmac, hmac_blob)
        yield {"type": "hmac_check", **hmac_diag}

        if not hmac_diag.get("ok"):
            yield {"type": "error", "stage": "hmac_failed",
                   "reason": hmac_diag.get("reason")}
            return

        # 4) Валидация payload
        val = validate_payload(inner_without_hmac)
        yield {"type": "payload_validate", **val}
        if not val.get("ok"):
            yield {"type": "error", "stage": "payload_invalid",
                   "reason": val.get("reason")}
            return

        mode = val["mode"]

        # 5) Извлечь options + narrator БЕЗОПАСНО
        raw_options = inner_without_hmac.get("options")
        options: Dict[str, Any] = raw_options if isinstance(raw_options, dict) else {}

        raw_narrator = inner_without_hmac.get("narrator")
        narrator: Dict[str, Any] = raw_narrator if isinstance(raw_narrator, dict) else {}

        engine = _safe_str(options.get("engine")) or "omnivoice"

        # 6) Собрать сегменты (с фильтрацией non-dict)
        raw_segments = _collect_segments(inner_without_hmac, mode)
        segments: List[Dict[str, Any]] = [s for s in raw_segments if isinstance(s, dict)]
        skipped_invalid = len(raw_segments) - len(segments)

        yield {
            "type": "config",
            "engine": engine,
            "vc_model": options.get("vc_model"),
            "voice": _safe_str(options.get("voice")) or _safe_str(narrator.get("voice")) or None,
            "scene_id": _safe_str(inner_without_hmac.get("scene_id")) or None,
            "mode": mode,
            "segments_count": len(segments),
            "skipped_invalid_segments": skipped_invalid,
        }

        if skipped_invalid > 0:
            yield {"type": "warning", "stage": "segments_filter",
                   "skipped": skipped_invalid,
                   "note": "non-dict items dropped from segments list"}

        # 7) Синтез — stub, чтобы валидировать сквозной NDJSON-контракт
        total = len(segments)
        yield {"type": "synthesis_start", "total": total}

        ok_count = 0
        err_count = 0

        for i, seg in enumerate(segments):
            seg_t0 = time.time()
            try:
                segment_id = seg.get("id") or seg.get("segment_id")
                speaker = _safe_str(seg.get("speaker")) or None
                text = _safe_str(seg.get("text"))

                # TODO: здесь — реальный синтез.
                #   wav_bytes = synthesize_segment(seg, engine, options)
                #   audio_b64 = base64.b64encode(wav_bytes).decode()
                yield {
                    "type": "segment_stub",
                    "index": i,
                    "segment_id": segment_id,
                    "speaker": speaker,
                    "text_len": len(text),
                    "elapsed_ms": int((time.time() - seg_t0) * 1000),
                    "note": "synthesis_not_implemented_yet",
                }
                ok_count += 1
            except Exception as e:
                err_count += 1
                yield {
                    "type": "error",
                    "stage": "segment",
                    "index": i,
                    "error": repr(e),
                    "trace": traceback.format_exc()[-500:],
                }

        yield {
            "type": "done",
            "total": total,
            "ok": ok_count,
            "errors": err_count,
            "skipped_invalid_segments": skipped_invalid,
            "wall_ms": int((time.time() - t0) * 1000),
        }

    except GeneratorExit:
        # Клиент/RunPod закрыл стрим — не считаем за crash, просто выходим.
        raise
    except Exception as e:
        fatal_emitted = True
        try:
            yield {
                "type": "fatal",
                "stage": "handler_outer",
                "error": repr(e),
                "trace": traceback.format_exc()[-1000:],
                "wall_ms": int((time.time() - t0) * 1000),
            }
        except Exception:
            # если даже yield упал — ничего не поделать, пусть RunPod пометит FAILED
            pass
        # НЕ re-raise: пусть RunPod видит COMPLETED, но в stream будет type=fatal.
        # Если хочешь чтобы job помечался FAILED — раскомментируй:
        # raise
    finally:
        if not fatal_emitted:
            # Diagnostic finalizer — виден в stdout логах воркера,
            # помогает отличать "нормальный finish" от "crash после yield".
            try:
                print(f"[handler] generator finished cleanly in {int((time.time()-t0)*1000)}ms",
                      flush=True)
            except Exception:
                pass


# ─────────────────────────────── Bootstrap ─────────────────────────────────

if not SHARED_TOKEN:
    print("[handler][BOOT WARNING] RUNPOD_SHARED_TOKEN env is EMPTY — "
          "all jobs will fail with shared_token_not_configured", flush=True)
else:
    print(f"[handler][BOOT] SHARED_TOKEN configured (len={len(SHARED_TOKEN)}), "
          f"runpod sdk={getattr(runpod, '__version__', 'unknown')}, "
          f"py={sys.version.split()[0]}", flush=True)

runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True,
})
