"""Common interface every baseline implements.

A baseline is anything that maps (source video + edit prompt) -> edited
video. That's it. Keep the interface small so wrapping a new model is a
30-line file: subclass BaselineModel, implement `edit_video`, register in
BASELINES.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class BaselineResult:
    """Where the baseline wrote its output and any per-run stats."""
    output_video: Path
    elapsed_sec: float
    extra: dict[str, Any]        # loss, seed, num_steps, whatever; goes into the run manifest


class BaselineModel(ABC):
    """Interface: load once, edit many videos."""

    # Short id used in output paths ("wan_vace_14b", "stub", ...). Uniqueness
    # is enforced by the top-level runner via the BASELINES registry.
    name: str = "unnamed"

    def __init__(self, **kwargs: Any) -> None:
        self.config = kwargs

    def setup(self) -> None:
        """Lazy heavy loading (model weights, tokenizer, ...). Called once."""
        pass

    def teardown(self) -> None:
        """Free GPU. Called once."""
        pass

    @abstractmethod
    def edit_video(
        self,
        source_video: Path,
        prompt: str,
        output_path: Path,
        *,
        edit_info: dict[str, Any],
    ) -> BaselineResult:
        """Edit `source_video` per `prompt`; write mp4 to `output_path`.

        `edit_info` is the full edit record from benchmark_manifest.json --
        pass through in case the baseline wants to know the property being
        edited, the from/to numbers, etc. Most baselines ignore it.
        """
        raise NotImplementedError
