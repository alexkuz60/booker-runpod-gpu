"""Booker GPU backend — diagnostic-only handler (v1, Step 1)."""
import os
import sys
import platform
import runpod


def _omnivoice_status():
    try:
        import omnivoice  # noqa: F401
        return {"ok": True, "version": getattr(omnivoice, "__version__", "unknown")}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _torch_status():
    try:
        import torch
        return {
            "ok": True,
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def handler(event):
    inp = (event or {}).get("input") or {}

    if inp.get("diagnostic"):
        return {
            "ok": True,
            "python": sys.version,
            "platform": platform.platform(),
            "endpoint_id": os.environ.get("RUNPOD_ENDPOINT_ID"),
            "torch": _torch_status(),
            "omnivoice": _omnivoice_status(),
        }

    return {"ok": False, "error": "not_implemented_in_step1"}


print("[BOOT] Booker GPU v1 handler ready", flush=True)
runpod.serverless.start({"handler": handler})
