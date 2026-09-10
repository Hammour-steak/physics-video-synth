"""Stub baseline: copies the source video unchanged as the "prediction".

Two uses:
1. Lower bound / sanity check: what do metrics look like when the model
   does literally nothing? Any real baseline should beat this on the
   perceptual metrics (it's identical to the source, not the target).
2. End-to-end smoke test: verifies run_baseline.py + metrics work without
   needing a GPU.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

from .base import BaselineModel, BaselineResult


class StubBaseline(BaselineModel):
    name = "stub"

    def edit_video(
        self,
        source_video: Path,
        prompt: str,
        output_path: Path,
        *,
        edit_info: dict[str, Any],
    ) -> BaselineResult:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        shutil.copy2(source_video, output_path)
        return BaselineResult(
            output_video=output_path,
            elapsed_sec=time.perf_counter() - t0,
            extra={"note": "source copied verbatim; no editing performed"},
        )
