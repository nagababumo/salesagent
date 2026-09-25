from __future__ import annotations

from typing import Optional

try:
    from jiwer import cer, wer
except ImportError:
    cer = wer = None


def calculate_metrics(reference: Optional[str], hypothesis: str):
    if not reference or wer is None:
        return None, None

    try:
        w = float(wer(reference, hypothesis))
    except Exception:
        w = None

    try:
        c = float(cer(reference, hypothesis))
    except Exception:
        c = None

    return w, c
