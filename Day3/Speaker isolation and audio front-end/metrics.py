from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from modelcodesday1.evaluate_asr import metric, script_aware_metrics


def calculate_metrics(reference: Optional[str], hypothesis: str):
    if not reference:
        return None, None
    scores = metric(reference, hypothesis)
    return scores["wer"], scores["cer"]


def calculate_script_metrics(reference: Optional[str], hypothesis: str):
    if not reference:
        return None
    return script_aware_metrics(reference, hypothesis)
