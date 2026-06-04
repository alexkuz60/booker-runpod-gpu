"""Booker RunPod GPU — minimal diagnostic handler (Step 1).

Принимает ОДИН тип запроса:
    {"input": {"diagnostic": true}}
Возвращает структуру с инфой про окружение. Никакого TTS.

Полная синтеза появится в Step 4 — после того как этот пинг устойчиво зелёный.
"""

import os
import sys
import time
import platform
import traceback

import runpod


def _safe_import_omnivoice():
    """Пытаемся импортнуть OmniVoice — не падаем при ошибке, возвращаем диагностику."""
    try:
        import omnivoice  # noqa: F401
        return {"ok": True, "version": getattr(omnivoice, "__version__", "unknown")}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _safe_torch_info():
    try:
        import torch
        return {
            "ok": True,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def handler(job):
    """RunPod serverless entrypoint."""
    started = time.time()
    inp = (job or {}).get("input") or {}

    # Шаг 1 v1: поддерживаем только diagnostic. Всё остальное — fail-fast.
    if inp.get("diagnostic") is True:
        return {
            "ok": True,
            "mode": "diagnostic",
            "elapsed_ms": int((time.time() - started) * 1000),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "env_runpod_endpoint": os.environ.get("RUNPOD_ENDPOINT_ID", "n/a"),
            "torch": _safe_torch_info(),
            "omnivoice_import": _safe_import_omnivoice(),
        }

    return {
        "ok": False,
        "error": "not_implemented_in_step1",
        "hint": "Send {\"input\": {\"diagnostic\": true}}. TTS arrives in Step 4.",
    }


if __name__ == "__main__":
    print("[BOOT] starting RunPod serverless worker (diagnostic-only)", flush=True)
    runpod.serverless.start({"handler": handler})

