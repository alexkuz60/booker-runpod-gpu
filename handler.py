"""
booker-runpod-gpu — RunPod Serverless handler with FULL diagnostic yields.

Контракт со стороны Lovable Edge (synthesize-scene-runpod):
  - RunPod payload: { "input": { ...innerInput, "_hmac": {"ts": "...", "sig": "..."} } }
  - innerInput = { ...originalClientPayload, "user_id_hash": "<8hex>" }
  - HMAC SHA-256:  hex( HMAC(SHARED_TOKEN, f"{ts}.{json.dumps(innerInput, separators=(',',':'), ensure_ascii=False)}") )
  - innerInput НЕ содержит ключа "_hmac" на момент подписи (он добавляется ПОСЛЕ подписи).

КРИТИЧНО:
  1. handler — async generator с yield (НЕ return [...]).
  2. runpod.serverless.start({"handler": handler, "return_aggregate_stream": True}).
  3. Любая ошибка/валидация — yield {"type":"error", ...}; return  (НЕ молчаливый return).
  4. Первый yield — "start" с эхом ключей payload, чтобы видеть, что handler вообще получил данные.
"""

import os
import json
import hmac
import hashlib
import time
import traceback
from typing import Any, Dict, AsyncGenerator

import runpod


SHARED_TOKEN = os.environ.get("RUNPOD_SHARED_TOKEN", "")
HMAC_MAX_SKEW_MS = 10 * 60 * 1000  # 10 минут — окно для ts


# ─────────────────────────────── HMAC ──────────────────────────────────────

def _canonical_inner(payload_without_hmac: Dict[str, Any]) -> str:
    """
    ВАЖНО: точное совпадение с тем, что подписывает Edge:
      JSON.stringify(innerInput)  ⟶ json.dumps(..., separators=(',',':'), ensure_ascii=False)
    Порядок ключей в Python 3.7+ сохраняется как insertion order, что соответствует
    JavaScript object property order для строковых ключей.
    """
    return json.dumps(payload_without_hmac, separators=(",", ":"), ensure_ascii=False)


def _compute_sig(ts: str, body: str) -> str:
    return hmac.new(
        SHARED_TOKEN.encode("utf-8"),
        f"{ts}.{body}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def verify_hmac(inner_without_hmac: Dict[str, Any], hmac_blob: Dict[str, Any]) -> Dict[str, Any]:
    """
    Возвращает dict с диагностикой. Поле `ok` — финальный вердикт.
    Никогда не бросает исключение — всё в диагностику.
    """
    diag: Dict[str, Any] = {"ok": False}
    try:
        ts = str(hmac_blob.get("ts", ""))
        got_sig = str(hmac_blob.get("sig", ""))
        if not ts or not got_sig:
            diag["reason"] = "missing_ts_or_sig"
            return diag

        # Защита от replay (мягкая — только лог, не отказ)
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

async def handler(job: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
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
    }

    if not isinstance(raw_input, dict):
        yield {"type": "error", "stage": "input_shape", "error": "input is not a dict"}
        return

    # 2) Извлечь _hmac и подготовить inner без _hmac (точно как Edge подписывает)
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
        "voice": options.get("voice") or inner_without_hmac.get("narrator", {}).get("voice"),
        "scene_id": inner_without_hmac.get("scene_id"),
        "segments_count": len(inner_without_hmac.get("segments", []))
            if val["mode"] == "single"
            else sum(len(s.get("segments", [])) for s in inner_without_hmac.get("scenes", [])),
    }

    # 6) Синтез — обернуть в try, чтобы любая ошибка летела наружу как event
    try:
        segments = inner_without_hmac.get("segments", []) if val["mode"] == "single" else [
            seg for sc in inner_without_hmac["scenes"] for seg in sc.get("segments", [])
        ]
        total = len(segments)

        yield {"type": "synthesis_start", "total": total}

        for i, seg in enumerate(segments):
            seg_t0 = time.time()
            try:
                # ──────────────────────────────────────────────────────────
                # TODO: здесь должен быть реальный вызов синтеза:
                #   wav_bytes = synthesize_segment(seg, engine, options)
                #   audio_b64 = base64.b64encode(wav_bytes).decode()
                # Пока заглушка, чтобы валидировать сквозной NDJSON-контракт.
                # ──────────────────────────────────────────────────────────
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
                # продолжаем со следующим сегментом

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
