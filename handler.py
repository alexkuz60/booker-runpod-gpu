# handler.py — Booker GPU backend on RunPod Serverless (OmniVoice via omnivoice-server)
# ============================================================================
# v2 (2026-05-31): добавлена HMAC-диагностика. При bad_signature handler
# возвращает поля для побайтового сравнения с edge function:
#   inner_len_bytes, inner_sha256, inner_head, inner_tail, ts, sig.
# Сравните их с логом `[runpod-proxy][hmac-debug]` в Supabase Edge Logs.
# ============================================================================

import os
import io
import time
import json
import hmac
import base64
import hashlib
import subprocess

import runpod
import httpx

OMNI_HOST = "127.0.0.1"
OMNI_PORT = 8880
OMNI_BASE = f"http://{OMNI_HOST}:{OMNI_PORT}"
TARGET_SR = 44100  # project Audio Standard

GPU_NAME = os.environ.get("RUNPOD_GPU_TYPE", "RTX-4090")
SHARED_TOKEN = os.environ.get("RUNPOD_SHARED_TOKEN", "")

# ── Boot omnivoice-server once per cold-start ────────────────────────────────
_server_proc = None


def _start_omnivoice_server():
    """Boot omnivoice-server as a subprocess and wait for /health."""
    global _server_proc
    if _server_proc and _server_proc.poll() is None:
        return

    runtime_dir = "/tmp/omnivoice-runtime"
    os.makedirs(runtime_dir, exist_ok=True)

    env = os.environ.copy()
    for k in list(env.keys()):
        if k.startswith(("VITE_", "SUPABASE_", "RUNPOD_")):
            env.pop(k, None)

    print(f"[boot] starting omnivoice-server on {OMNI_BASE}", flush=True)
    _server_proc = subprocess.Popen(
        [
            "omnivoice-server",
            "--host", OMNI_HOST,
            "--port", str(OMNI_PORT),
            "--device", "cuda",
        ],
        cwd=runtime_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
    )

    deadline = time.time() + 600
    last_err = None
    started = time.time()
    with httpx.Client(timeout=5.0) as client:
        while time.time() < deadline:
            if _server_proc.poll() is not None:
                out = _server_proc.stdout.read() if _server_proc.stdout else ""
                raise RuntimeError(f"omnivoice-server died on boot:\n{out}")
            try:
                r = client.get(f"{OMNI_BASE}/health")
                if r.status_code == 200:
                    print(f"[boot] omnivoice-server ready in "
                          f"{int(time.time() - started)}s", flush=True)
                    return
            except Exception as e:
                last_err = e
            time.sleep(2)

    raise RuntimeError(f"omnivoice-server /health timeout: {last_err}")


# ── HMAC verification + diagnostics ─────────────────────────────────────────
def _canonical_inner(inp_without_hmac: dict) -> str:
    """
    Re-serialize inner payload matching JS default `JSON.stringify`:
      - no extra whitespace
      - keys NOT sorted (insertion order preserved)
      - ensure_ascii=False (non-ASCII written as UTF-8)
    NOTE: This is the known weak point — Python and JS may still differ on
    float formatting (1.0 vs 1) and Unicode normalization. The debug fields
    below let us pinpoint any mismatch byte-by-byte.
    """
    return json.dumps(inp_without_hmac, separators=(",", ":"), ensure_ascii=False)


def _hmac_debug(inner_str: str, ts: str, sig: str) -> dict:
    inner_bytes = inner_str.encode("utf-8")
    sha = hashlib.sha256(inner_bytes).hexdigest()
    return {
        "ts": ts,
        "sig": sig,
        "inner_len_bytes": len(inner_bytes),
        "inner_sha256": sha,
        "inner_head": inner_str[:80],
        "inner_tail": inner_str[-80:],
        "shared_token_len": len(SHARED_TOKEN),
    }


def _verify_signature(input_str: str, ts: str, sig: str) -> bool:
    if not SHARED_TOKEN or not ts or not sig:
        return False
    try:
        ts_int = int(ts)
    except ValueError:
        return False
    if abs(ts_int - int(time.time() * 1000)) > 5 * 60_000:
        return False
    expected = hmac.new(
        SHARED_TOKEN.encode(),
        f"{ts}.{input_str}".encode(),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, sig)


# ── Resample 24kHz → 44.1kHz, mono, 16-bit PCM ──────────────────────────────
def _resample_wav_to_44100(wav_bytes: bytes) -> bytes:
    import soundfile as sf
    import numpy as np
    from scipy.signal import resample_poly
    from math import gcd

    with io.BytesIO(wav_bytes) as f:
        data, sr = sf.read(f, dtype="float32", always_2d=False)

    if data.ndim > 1:
        data = data.mean(axis=1)

    if sr != TARGET_SR:
        g = gcd(TARGET_SR, sr)
        up, down = TARGET_SR // g, sr // g
        data = resample_poly(data, up, down).astype("float32")

    data = np.clip(data, -1.0, 1.0)
    pcm16 = (data * 32767.0).astype("<i2")

    out = io.BytesIO()
    sf.write(out, pcm16, TARGET_SR, format="WAV", subtype="PCM_16")
    return out.getvalue()


# ── Single-segment synthesis ────────────────────────────────────────────────
def _synth_one(client: httpx.Client, seg: dict, language: str) -> bytes:
    voice = seg.get("voice", {}) or {}
    mode = voice.get("mode", "design")
    text = seg.get("text", "")

    if mode == "clone" and voice.get("ref_audio_b64"):
        ref_wav = base64.b64decode(voice["ref_audio_b64"])
        files = {"ref_audio": ("ref.wav", ref_wav, "audio/wav")}
        data = {
            "input": text,
            "ref_text": voice.get("ref_text", ""),
            "language": language,
        }
        adv = voice.get("advanced") or {}
        if adv:
            data["advanced"] = json.dumps(adv)
        r = client.post(
            f"{OMNI_BASE}/v1/audio/speech/clone",
            files=files, data=data, timeout=300.0,
        )
    else:
        payload = {
            "model": "omnivoice",
            "input": text,
            "voice": voice.get("voice", "default"),
            "instructions": voice.get("instructions", ""),
            "language": language,
            "response_format": "wav",
        }
        adv = voice.get("advanced") or {}
        if adv:
            payload["advanced"] = adv
        r = client.post(
            f"{OMNI_BASE}/v1/audio/speech",
            json=payload, timeout=300.0,
        )

    if r.status_code != 200:
        raise RuntimeError(f"omnivoice {r.status_code}: {r.text[:300]}")

    return r.content


# ── RunPod handler (generator) ──────────────────────────────────────────────
def handler(event):
    """
    RunPod serverless generator handler.
    Yields dicts forwarded as NDJSON lines to the browser.
    """
    inp = event.get("input") or {}

    # Verify HMAC
    hmac_obj = inp.pop("_hmac", None) or {}
    ts = str(hmac_obj.get("ts", ""))
    sig = str(hmac_obj.get("sig", ""))
    inner_str = _canonical_inner(inp)

    debug = _hmac_debug(inner_str, ts, sig)
    print(f"[hmac-debug] {json.dumps(debug)}", flush=True)

    if not _verify_signature(inner_str, ts, sig):
        # Return diagnostics so user can compare with edge function log.
        yield {
            "type": "error",
            "error": "bad_signature",
            "debug": debug,
        }
        return

    # Cold-start the OmniVoice server (idempotent — only on first job)
    try:
        _start_omnivoice_server()
    except Exception as e:
        yield {"type": "error", "error": f"omnivoice_boot_failed: {e}"}
        return

    segments = inp.get("segments") or []
    language = inp.get("language", "en")
    scene_id = inp.get("scene_id", "")
    total = len(segments)

    t0 = time.time()
    ok_count = 0
    fail_count = 0

    yield {
        "type": "start",
        "total": total,
        "warm": True,
        "gpu": GPU_NAME,
        "scene_id": scene_id,
    }

    with httpx.Client() as client:
        for i, seg in enumerate(segments):
            seg_id = seg.get("segment_id", f"seg_{i}")
            speaker = seg.get("speaker", "narrator")
            try:
                wav24 = _synth_one(client, seg, language)
                wav44 = _resample_wav_to_44100(wav24)

                # 16-bit mono → (bytes - 44 header) / 2 samples
                samples = max(0, (len(wav44) - 44) // 2)
                duration_ms = int(samples * 1000 / TARGET_SR)

                wav_b64 = base64.b64encode(wav44).decode("ascii")
                ok_count += 1

                yield {
                    "type": "segment",
                    "index": i,
                    "segment_id": seg_id,
                    "speaker": speaker,
                    "duration_ms": duration_ms,
                    "wav_b64": wav_b64,
                }
            except Exception as e:
                fail_count += 1
                yield {
                    "type": "error",
                    "index": i,
                    "segment_id": seg_id,
                    "speaker": speaker,
                    "error": str(e)[:500],
                }

    yield {
        "type": "done",
        "total_ms": int((time.time() - t0) * 1000),
        "segments_ok": ok_count,
        "segments_failed": fail_count,
    }


# ── Entrypoint ──────────────────────────────────────────────────────────────
runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True,
})
