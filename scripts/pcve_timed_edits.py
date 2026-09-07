"""Shared plumbing for edits that land partway through a clip.

A timed edit reaches a simulator as a schedule rather than as a parameter set:

    [{"frame": 22, "params": {"mallet_active": [1, 0, 1, 0]}}]

The clip runs on the ordinary CLI parameters up to that frame, the params are
written into the live simulation there, and the run carries on from the state
it had reached -- which is what makes every frame before the edit identical to
the source video.

Reading the schedule and the two ways of writing one into a running simulation
(remove a body, change a body's dynamics) are the same in every scene, so they
live here. What each parameter *means* is not: a scene knows that
``marble_active`` indexes its marbles and ``ball_friction`` is its ball's
lateral coefficient, so every ``simulate_<scene>.py`` keeps its own
``apply_timed_params`` and calls these primitives from it.

An applier must raise on a parameter it does not recognise. A silently dropped
edit renders as a video that looks like the baseline, and nothing downstream
would catch it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_timed_edits(path: Path | str | None) -> dict[int, dict[str, Any]]:
    """Read a schedule file into ``{frame: merged params}``.

    Frame 1 is rejected: it is the initial state, which is what the ordinary
    CLI parameters already set, and accepting it here would give the same edit
    two spellings.
    """
    if path is None:
        return {}
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    schedule: dict[int, dict[str, Any]] = {}
    for entry in entries:
        frame = int(entry["frame"])
        if frame < 2:
            raise ValueError(
                f"Timed edit at frame {frame}: frame 1 is the initial state, "
                "which is set through the ordinary CLI parameters."
            )
        schedule.setdefault(frame, {}).update(dict(entry["params"]))
    return schedule


def check_horizon(schedule: dict[int, Any], frame_end: int) -> None:
    """Reject a schedule that fires after the last frame of the clip."""
    latest = max(schedule, default=0)
    if latest > frame_end:
        raise ValueError(
            f"Timed edit at frame {latest} is past the end of a "
            f"{frame_end}-frame clip"
        )


def remove_bodies(p, client: int, bodies: list, active_flags) -> None:
    """Take out every body whose slot has just been switched off.

    ``bodies`` is edited in place: a removed slot holds None afterwards, the
    same as a body that a whole-clip delete never built, so the frame loop
    around it needs no second code path. The removal is a real ``removeBody``
    -- the body stops colliding from this frame on, which is the whole point of
    a delete that lands between two impacts.
    """
    for index, is_active in enumerate(active_flags):
        if int(is_active) or index >= len(bodies) or bodies[index] is None:
            continue
        p.removeBody(bodies[index], physicsClientId=client)
        bodies[index] = None


def set_dynamics(p, client: int, bodies: list, values, field: str) -> None:
    """Write one per-body dynamics field from a per-body list of values."""
    for index, value in enumerate(values):
        if index >= len(bodies) or bodies[index] is None:
            continue
        p.changeDynamics(bodies[index], -1, **{field: float(value)},
                         physicsClientId=client)


def set_one(p, client: int, body, field: str, value) -> None:
    """Write one dynamics field on a single body, skipping a removed one."""
    if body is None:
        return
    p.changeDynamics(body, -1, **{field: float(value)}, physicsClientId=client)


def unknown_param(key: str) -> ValueError:
    """The error an applier raises for a parameter the scene cannot time."""
    return ValueError(
        f"Timed edit sets {key!r}, which this scene cannot change mid-run. "
        "Add it to apply_timed_params or write the edit as a whole-clip edit."
    )
