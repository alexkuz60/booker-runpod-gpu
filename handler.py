"""
booker-runpod-gpu — RunPod Serverless handler (SYNC generator).

ВАЖНО:
  - Используем СИНХРОННЫЙ генератор (def + yield), а НЕ async def + yield.
    Async generators поддерживаются только в RunPod SDK >= 1.5, и даже там
    в режиме `return_aggregate_stream` иногда возвращают пустой stream.
    Sync generator работает на всех версиях SDK без сюрпризов.

  - runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
    → каждый yield попадает в /stream/{id} как chunk.stream[].output
    → и финально дублируется в job.output как массив.

Контракт со стороны Lovable Edge (synthesize-scene-runpod):
  - RunPod payload: { "input": { ...innerInput, "_hmac": {"ts": "...", "sig": "..."} } }
  - innerInput = { ...originalClientPayload, "user_id_hash": "<8hex>" }
  - HMAC SHA-256:  hex( HMAC(SHARED_TOKEN, f"{ts}.{json.dumps(innerInput, separators=(',',':'), ensure_ascii=False)}") )
  - innerInput НЕ содержит ключа "_hmac" на момент подписи (он добавляется ПОСЛЕ подписи).

Первый yield — "start" с эхом ключей payload, чтобы СРАЗУ видеть, что handler
получил данные. Если в Edge-логах [runpod-proxy][chunk] видно stream_len>=1
с type=start — значит handler жив, дальше смотрим следующие events.
"""

import os
import json
import hmac
import hashlib
import time
import traceback
from typing import Any, Dict, Generator

import runpod


SHARED_TOKEN = os.environ.get("RUNPOD_SHARED_TOKEN", "")
HMAC_MAX_SKEW_MS = 10 * 60 * 1000  # 10 минут — окно для ts


# ─────────────────────────────── HMAC ──────────────────────────────────────

def _canonical_inner(payload_without_hmac: Dict[str, Any]) -> str:
    """
    Точное совпадение с тем, что подписывает Edge:
      JSON.stringify(innerInput) ⟶ json.dumps(..., separators=(',',':'), ensure_ascii=False)
    Порядок ключей в Python 3.7+ сохраняется как insertion order — то же, что
    в JavaScript object property order для строковых ключей.
    """
    return json.dumps(payload_without_hmac, separators=(",", ":"), ensure_ascii=False)


def _compute_sig(ts: str, body: str) -> str:
    return hmac.new(
        SHARED_TOKEN.encode("utf-8"),
        f"{ts}.{body}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def verify_hmac(inner_without_hmac: Dict[str, Any], hmac_blob: Dict[str, Any]) -> Dict[str, Any]:
    diag: Dict[str, Any] = {"ok": False}
    try:
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
        except ValueError:
            diag["reason"] = "ts_not_int"
            return diag

        body = _canonical_inner(inner_without_hmac)
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

    has_single = isinstance(payload.get("scene_id"), str) and isinstance(payload.get("segments"), list)
    has_batch = isinstance(payload.get("scenes"), list) and len(payload.get("scenes", [])) > 0
    if not (has_single or has_batch):
        return {"ok": False, "reason": "missing_scene_or_scenes"}

    if has_single and len(payload["segments"]) == 0:
        return {"ok": False, "reason": "segments_empty"}

    return {"ok": True, "mode": "single" if has_single else "batch"}


# ───────────────────────────────── Handler ─────────────────────────────────
# СИНХРОННЫЙ генератор — def, не async def!

def handler(job: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    t0 = time.time()

    # 1) Sanity: job/input shape
    try:
        raw_input = job.get("input")
    except Exception as e:
        yield {"type": "error", "stage": "job_access", "error": repr(e)}
        return

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
    }

    if not isinstance(raw_input, dict):
        yield {"type": "error", "stage": "input_shape", "error": "input is not a dict"}
        return

    # 2) Извлечь _hmac и подготовить inner без _hmac
    hmac_blob = raw_input.get("_hmac")
    if not isinstance(hmac_blob, dict):
        yield {
            "type": "error",
            "stage": "hmac_missing",
            "error": "no _hmac in input",
            "input_keys": list(raw_input.keys()),
        }
        return

    inner_without_hmac = {k: v for k, v in raw_input.items() if k != "_hmac"}

    # 3) HMAC verify с полной диагностикой
    hmac_diag = verify_hmac(inner_without_hmac, hmac_blob)
    yield {"type": "hmac_check", **hmac_diag}

    if not hmac_diag.get("ok"):
        yield {"type": "error", "stage": "hmac_failed", "diag": hmac_diag}
        return

    # 4) Валидация payload
    val = validate_payload(inner_without_hmac)
    yield {"type": "payload_validate", **val}
    if not val.get("ok"):
        yield {"type": "error", "stage": "payload_invalid", "reason": val.get("reason")}
        return

    # 5) Извлечь сегменты + options
    options = inner_without_hmac.get("options", {}) or {}
    engine = options.get("engine", "omnivoice")
    yield {
        "type": "config",
        "engine": engine,
        "vc_model": options.get("vc_model"),
        "voice": options.get("voice") or (inner_without_hmac.get("narrator") or {}).get("voice"),
        "scene_id": inner_without_hmac.get("scene_id"),
        "segments_count": len(inner_without_hmac.get("segments", []))
            if val["mode"] == "single"
            else sum(len(s.get("segments", [])) for s in inner_without_hmac.get("scenes", [])),
    }

    # 6) Синтез — stub, чтобы валидировать сквозной NDJSON-контракт
    try:
        segments = inner_without_hmac.get("segments", []) if val["mode"] == "single" else [
            seg for sc in inner_without_hmac["scenes"] for seg in sc.get("segments", [])
        ]
        total = len(segments)

        yield {"type": "synthesis_start", "total": total}

        for i, seg in enumerate(segments):
            seg_t0 = time.time()
            try:
                # TODO: здесь — реальный синтез.
                #   wav_bytes = synthesize_segment(seg, engine, options)
                #   audio_b64 = base64.b64encode(wav_bytes).decode()
                yield {
                    "type": "segment_stub",
                    "index": i,
                    "segment_id": seg.get("id") or seg.get("segment_id"),
                    "speaker": seg.get("speaker"),
                    "text_len": len(seg.get("text", "")),
                    "elapsed_ms": int((time.time() - seg_t0) * 1000),
                    "note": "synthesis_not_implemented_yet",
                }
            except Exception as e:
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
            "wall_ms": int((time.time() - t0) * 1000),
        }

    except Exception as e:
        yield {
            "type": "error",
            "stage": "synthesis_loop",
            "error": repr(e),
            "trace": traceback.format_exc()[-500:],
        }


# ─────────────────────────────── Bootstrap ─────────────────────────────────

runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True,
})
