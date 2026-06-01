"""
booker-runpod-gpu — RunPod Serverless handler v6 (REAL synthesis, variant B).

Изменения v5 → v6:
  • Реальный синтез через прямой Python-импорт OmniVoice
    (вариант Б — БЕЗ локального HTTP к omnivoice-server).
  • Глобальный singleton модели (ленивая загрузка при первом job → переживает
    последующие jobs внутри warm-воркера).
  • Per-phrase синтез: один сегмент = N фраз → N base64-WAV → один объединённый
    yield {status:"ready", segment_id, phrase_results:[…]}.
  • Согласовано с клиентским контрактом RunPodTestPanel.tsx:
        obj.status === "ready" && (obj.audio_base64 || obj.phrase_results)
  • Audio Standard проекта: 48 kHz / 16-bit mono WAV (integer ×2 из нативных
    24 kHz OmniVoice, без интерполяции — bit-perfect).
  • Вся v5-диагностика и outer try/except сохранены.
  • voice_configs[speaker] из клиентского payload пробрасывается в опции
    OmniVoice (speed, voice_id, language).

Контракт с Edge (synthesize-scene-runpod) тот же, что в v5:
  RunPod payload: { "input": { ...innerInput, "_hmac": {"ts","sig"} } }
  HMAC-SHA256 канонически: hex( HMAC(SHARED_TOKEN, f"{ts}.{json.dumps(innerInput, sort_keys=True, separators=(',',':'), ensure_ascii=False)}") )
"""

import os
import io
import sys
import json
import hmac
import wave
import base64
import hashlib
import time
import traceback
from typing import Any, Dict, Generator, List, Optional, Tuple

import runpod

# ───────────────────────── ENV / Bootstrap ─────────────────────────────────

SHARED_TOKEN = os.environ.get("RUNPOD_SHARED_TOKEN", "")
HMAC_MAX_SKEW_MS = 10 * 60 * 1000

OMNIVOICE_MODEL_ID = os.environ.get("OMNIVOICE_MODEL_ID", "k2-fsa/OmniVoice")
OMNIVOICE_DEVICE = os.environ.get("OMNIVOICE_DEVICE", "cuda")  # cuda | cpu
OMNIVOICE_CACHE_DIR = os.environ.get("OMNIVOICE_CACHE_DIR", "/runpod-volume/hf-cache")
OMNIVOICE_DEFAULT_STEPS = int(os.environ.get("OMNIVOICE_NUM_STEP", "16"))

NATIVE_SR = 24_000     # OmniVoice native
TARGET_SR = 48_000     # Booker Audio Standard

# ───────────────────────── Model singleton ─────────────────────────────────

_MODEL = None          # global, кэшируется между jobs в warm-воркере
_MODEL_LOAD_ERROR: Optional[str] = None
_MODEL_LOAD_MS: Optional[int] = None


def _load_model_once() -> Tuple[Any, Optional[str], Optional[int]]:
    """Ленивая загрузка OmniVoice. Кэшируется в глобале."""
    global _MODEL, _MODEL_LOAD_ERROR, _MODEL_LOAD_MS
    if _MODEL is not None or _MODEL_LOAD_ERROR is not None:
        return _MODEL, _MODEL_LOAD_ERROR, _MODEL_LOAD_MS

    t0 = time.time()
    try:
        import torch
        from omnivoice import OmniVoice

        dtype_candidates = [torch.bfloat16, torch.float16, torch.float32] \
            if OMNIVOICE_DEVICE == "cuda" else [torch.float32]

        kwargs: Dict[str, Any] = {
            "device_map": OMNIVOICE_DEVICE,
            "cache_dir": OMNIVOICE_CACHE_DIR,
        }
        last_exc: Optional[Exception] = None
        for dtype in dtype_candidates:
            try:
                model = OmniVoice.from_pretrained(
                    OMNIVOICE_MODEL_ID, dtype=dtype, **kwargs
                )
                # Smoke test
                _ = model.generate(text="test", num_step=4)
                _MODEL = model
                _MODEL_LOAD_MS = int((time.time() - t0) * 1000)
                print(f"[handler][BOOT] OmniVoice loaded dtype={dtype} "
                      f"in {_MODEL_LOAD_MS}ms on {OMNIVOICE_DEVICE}", flush=True)
                return _MODEL, None, _MODEL_LOAD_MS
            except Exception as e:
                last_exc = e
                print(f"[handler][BOOT] dtype={dtype} failed: {e!r}", flush=True)
                continue
        raise RuntimeError(f"all dtypes failed; last={last_exc!r}")
    except Exception as e:
        _MODEL_LOAD_ERROR = f"{type(e).__name__}: {e}"
        _MODEL_LOAD_MS = int((time.time() - t0) * 1000)
        print(f"[handler][BOOT][ERROR] OmniVoice load failed in "
              f"{_MODEL_LOAD_MS}ms: {_MODEL_LOAD_ERROR}", flush=True)
        return None, _MODEL_LOAD_ERROR, _MODEL_LOAD_MS


# ───────────────────────── HMAC verification ───────────────────────────────

def _canonical_inner(payload_without_hmac: Dict[str, Any]) -> str:
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
        except ValueError:
            diag["reason"] = "ts_not_int"
            return diag

        canonical = _canonical_inner(inner_without_hmac)
        expected = _compute_sig(ts, canonical)
        diag["sig_match"] = hmac.compare_digest(expected, got_sig)
        if not diag["sig_match"]:
            diag["reason"] = "sig_mismatch"
            diag["canonical_len"] = len(canonical)
            diag["canonical_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            return diag

        diag["ok"] = True
        return diag
    except Exception as e:
        diag["reason"] = f"hmac_exception:{type(e).__name__}"
        diag["error"] = repr(e)
        return diag


# ───────────────────────── Payload validation ──────────────────────────────

def validate_payload(p: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(p, dict):
        return {"ok": False, "reason": "payload_not_dict"}

    has_single = isinstance(p.get("scene_id"), str) and isinstance(p.get("segments"), list)
    has_batch = isinstance(p.get("scenes"), list) and len(p.get("scenes") or []) > 0

    if not has_single and not has_batch:
        return {"ok": False, "reason": "neither_single_nor_batch_shape"}

    return {"ok": True, "mode": "single" if has_single else "batch"}


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        return str(x)
    except Exception:
        return ""


def _collect_segments(inner: Dict[str, Any], mode: str) -> List[Dict[str, Any]]:
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


# ───────────────────────── Audio post-processing ───────────────────────────

def _tensor_to_int16_mono_48k(tensor) -> Tuple[bytes, int]:
    """
    OmniVoice → list[torch.Tensor], каждый (1, T) float в [-1, 1] @ 24kHz.
    Возвращаем (pcm_int16_bytes, duration_ms) уже @ 48kHz mono.
    Bit-perfect integer ×2 upsample (np.repeat) — соответствует Audio Standard.
    """
    import numpy as np
    import torch  # noqa

    arr = tensor.detach().to("cpu").float().numpy()
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    # Clip и в int16
    arr = np.clip(arr, -1.0, 1.0)
    int16 = (arr * 32767.0).astype("<i2")
    # ×2 nearest-neighbor (24k → 48k, integer, без интерполяции)
    upsampled = np.repeat(int16, 2)
    duration_ms = int(len(upsampled) * 1000 / TARGET_SR)
    return upsampled.tobytes(), duration_ms


def _pcm_to_wav_bytes(pcm: bytes, sample_rate: int = TARGET_SR) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


# ───────────────────────── OmniVoice call ──────────────────────────────────

def _synthesize_text(model, text: str, opts: Dict[str, Any]) -> Tuple[bytes, int]:
    """text → (wav_bytes 48kHz int16 mono, duration_ms)."""
    kwargs: Dict[str, Any] = {
        "text": text,
        "num_step": int(opts.get("num_step") or OMNIVOICE_DEFAULT_STEPS),
        "speed": float(opts.get("speed") or 1.0),
    }
    if opts.get("language"):
        kwargs["language"] = str(opts["language"])
    if opts.get("instruct"):
        kwargs["instruct"] = str(opts["instruct"])
    if opts.get("guidance_scale") is not None:
        kwargs["guidance_scale"] = float(opts["guidance_scale"])

    try:
        tensors = model.generate(**kwargs)
    except TypeError as e:
        # Upstream API drift — fallback на минимальные kwargs
        print(f"[handler] generate() TypeError: {e!r}; retry minimal", flush=True)
        tensors = model.generate(text=text, num_step=kwargs["num_step"])

    if not tensors:
        raise RuntimeError("model.generate returned empty tensors list")

    # Конкатенация чанков (OmniVoice может вернуть несколько при длинном тексте)
    import numpy as np
    pieces: List[bytes] = []
    total_ms = 0
    for t in tensors:
        pcm, dur = _tensor_to_int16_mono_48k(t)
        pieces.append(pcm)
        total_ms += dur
    full_pcm = b"".join(pieces)
    wav_bytes = _pcm_to_wav_bytes(full_pcm, TARGET_SR)
    return wav_bytes, total_ms


def _build_opts(seg: Dict[str, Any], voice_configs: Dict[str, Any],
                global_options: Dict[str, Any]) -> Dict[str, Any]:
    speaker = _safe_str(seg.get("speaker"))
    voice_cfg = voice_configs.get(speaker) if isinstance(voice_configs, dict) else None
    voice_cfg = voice_cfg if isinstance(voice_cfg, dict) else {}

    opts: Dict[str, Any] = {
        "speed": voice_cfg.get("speed") or global_options.get("speed") or 1.0,
        "language": voice_cfg.get("language") or global_options.get("language"),
        "num_step": voice_cfg.get("num_step") or global_options.get("num_step"),
        "guidance_scale": voice_cfg.get("guidance_scale")
            or global_options.get("guidance_scale"),
        "instruct": voice_cfg.get("instruct") or global_options.get("instruct"),
    }
    return opts


# ───────────────────────────── Handler ─────────────────────────────────────

def handler(job: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    t0 = time.time()
    fatal_emitted = False

    try:
        # 1) Job shape
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
            "model_preloaded": _MODEL is not None,
            "model_load_error": _MODEL_LOAD_ERROR,
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

        # 2) HMAC
        hmac_blob = raw_input.get("_hmac")
        if not isinstance(hmac_blob, dict):
            yield {"type": "error", "stage": "hmac_missing",
                   "error": "no _hmac dict in input",
                   "input_keys": list(raw_input.keys())}
            return

        inner = {k: v for k, v in raw_input.items() if k != "_hmac"}
        hmac_diag = verify_hmac(inner, hmac_blob)
        yield {"type": "hmac_check", **hmac_diag}
        if not hmac_diag.get("ok"):
            yield {"type": "error", "stage": "hmac_failed",
                   "reason": hmac_diag.get("reason")}
            return

        # 3) Payload
        val = validate_payload(inner)
        yield {"type": "payload_validate", **val}
        if not val.get("ok"):
            yield {"type": "error", "stage": "payload_invalid",
                   "reason": val.get("reason")}
            return
        mode = val["mode"]

        # 4) Options / voice_configs
        global_options = inner.get("options") if isinstance(inner.get("options"), dict) else {}
        voice_configs = inner.get("voice_configs") \
            if isinstance(inner.get("voice_configs"), dict) else {}

        # 5) Lazy-load model (на первом job воркера)
        if _MODEL is None and _MODEL_LOAD_ERROR is None:
            yield {"type": "model_loading", "model_id": OMNIVOICE_MODEL_ID,
                   "device": OMNIVOICE_DEVICE}
        model, err, load_ms = _load_model_once()
        if err:
            yield {"type": "error", "stage": "model_load_failed",
                   "error": err, "load_ms": load_ms}
            return
        yield {"type": "model_ready", "load_ms": load_ms,
               "cold_start": load_ms is not None and load_ms > 100}

        # 6) Collect segments
        raw_segments = _collect_segments(inner, mode)
        segments = [s for s in raw_segments if isinstance(s, dict)]
        skipped_invalid = len(raw_segments) - len(segments)
        yield {"type": "synthesis_start", "total": len(segments),
               "skipped_invalid": skipped_invalid}

        ok_count = 0
        err_count = 0

        # 7) Per-segment synthesis
        for i, seg in enumerate(segments):
            seg_t0 = time.time()
            segment_id = _safe_str(seg.get("segment_id") or seg.get("id")) \
                or f"seg_{i}"
            try:
                phrases = seg.get("phrases")
                if not isinstance(phrases, list) or not phrases:
                    # Фоллбэк: одиночный text-сегмент (старый контракт)
                    text = _safe_str(seg.get("text"))
                    if not text:
                        raise ValueError("no phrases[] and no text")
                    phrases = [{"phrase_id": f"{segment_id}::p0", "text": text}]

                opts = _build_opts(seg, voice_configs, global_options)

                phrase_results: List[Dict[str, Any]] = []
                seg_total_ms = 0
                for ph in phrases:
                    if not isinstance(ph, dict):
                        continue
                    phrase_id = _safe_str(ph.get("phrase_id")) \
                        or f"{segment_id}::p{len(phrase_results)}"
                    text = _safe_str(ph.get("text"))
                    if not text:
                        continue
                    wav, dur_ms = _synthesize_text(model, text, opts)
                    phrase_results.append({
                        "phrase_id": phrase_id,
                        "audio_base64": base64.b64encode(wav).decode("ascii"),
                        "duration_ms": dur_ms,
                        "sample_rate": TARGET_SR,
                    })
                    seg_total_ms += dur_ms

                if not phrase_results:
                    raise RuntimeError("all phrases empty after synthesis")

                # Контракт клиента (RunPodTestPanel):
                #   obj.status === "ready" && (obj.audio_base64 || obj.phrase_results)
                yield {
                    "type": "clip",
                    "status": "ready",
                    "segment_id": segment_id,
                    "phrase_results": phrase_results,
                    "duration_ms": seg_total_ms,
                    "sample_rate": TARGET_SR,
                    "elapsed_ms": int((time.time() - seg_t0) * 1000),
                }
                ok_count += 1

            except Exception as e:
                err_count += 1
                yield {
                    "type": "error",
                    "stage": "segment",
                    "index": i,
                    "segment_id": segment_id,
                    "error": repr(e),
                    "trace": traceback.format_exc()[-600:],
                }

        yield {
            "type": "done",
            "total": len(segments),
            "ok": ok_count,
            "errors": err_count,
            "skipped_invalid_segments": skipped_invalid,
            "wall_ms": int((time.time() - t0) * 1000),
        }

    except GeneratorExit:
        raise
    except Exception as e:
        fatal_emitted = True
        try:
            yield {
                "type": "fatal",
                "stage": "handler_outer",
                "error": repr(e),
                "trace": traceback.format_exc()[-1200:],
                "wall_ms": int((time.time() - t0) * 1000),
            }
        except Exception:
            pass
    finally:
        if not fatal_emitted:
            try:
                print(f"[handler] generator finished cleanly in "
                      f"{int((time.time()-t0)*1000)}ms", flush=True)
            except Exception:
                pass


# ───────────────────────── Bootstrap log ───────────────────────────────────

if not SHARED_TOKEN:
    print("[handler][BOOT WARNING] RUNPOD_SHARED_TOKEN is EMPTY — "
          "all jobs will fail with shared_token_not_configured", flush=True)
else:
    print(f"[handler][BOOT] SHARED_TOKEN ok (len={len(SHARED_TOKEN)}), "
          f"runpod sdk={getattr(runpod, '__version__', 'unknown')}, "
          f"py={sys.version.split()[0]}, "
          f"model_id={OMNIVOICE_MODEL_ID}, device={OMNIVOICE_DEVICE}", flush=True)

# Опционально: прогреть модель сразу (увеличивает cold-start первого job, но
# делает запуск более предсказуемым). Включи, если RunPod держит warm workers.
if os.environ.get("OMNIVOICE_PRELOAD", "0") == "1":
    print("[handler][BOOT] OMNIVOICE_PRELOAD=1 → eager load", flush=True)
    _load_model_once()

runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True,
})
