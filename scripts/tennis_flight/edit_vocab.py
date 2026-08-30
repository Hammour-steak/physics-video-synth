"""Edit vocabulary for the tennis_flight scene.

Declares which properties are editable on the flying tennis ball, how each
maps to the sim's physics parameter dict, and the baseline physics values.
Feed this to ``pcve_edit_dsl`` and prompts / overrides / diffs come out
data-driven.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pcve_edit_dsl as dsl  # noqa: E402


# Must stay in sync with render_tennis_flight.create_scenario().
# Single flying tennis ball -- launched from (-4, 0, 1.5) with initial
# velocity (4.5, 0, 4.5) m/s -- arcs over the net at x=0, lands, and rolls.
BASELINE_PHYSICS = {
    "ball_radius": 0.033,
    "ball_mass": 0.057,
    "ball_friction": 0.5,
    "ball_restitution": 0.05,
    "ball_rolling_friction": 0.015,
    "floor_friction": 0.6,
    "launch_location": [-4.0, 0.0, 1.5],
    "launch_velocity": [4.5, 0.0, 4.5],
    "gravity": [0.0, 0.0, -9.8],
}


OBJECTS = {
    "ball": {"zh": "网球", "en": "the tennis ball"},
}
# Deliberately NOT editable: the ground plane and the surrounding tennis
# court set dressing. Bullet multiplies friction and restitution across
# the pair, so any effective coefficient reachable through the ground is
# also reachable through the ball. launch_location, ball_radius, and
# gravity are scene-wide setup rather than per-object physics.
#
# Single moving object, so there is no DELETE binding -- removing the
# ball would leave an empty court, not a physically meaningful edit.


PROPERTIES = {
    "mass":        dsl.PropertySpec("scalar", "kg", "质量",     "mass"),
    # `friction` collapses lateral + rolling into one "total friction"
    # knob. A rolling tennis ball loses most of its speed to rolling
    # resistance after it lands; editing lateral alone would leave the
    # roll-out mostly unchanged.
    "friction":    dsl.PropertySpec("scalar", "",   "摩擦系数", "friction coefficient"),
    "restitution": dsl.PropertySpec("scalar", "",   "恢复系数", "restitution"),
    # Scalar speed magnitude for the launch. Direction stays the baseline's
    # (+X down the court).
    "initial_velocity": dsl.PropertySpec("scalar", "m/s", "初速度大小", "initial speed"),
}


SIM_BINDINGS = {
    ("ball", "mass"): dsl.SimBinding("ball_mass"),
    # `friction` scales lateral + rolling together (display value is the
    # lateral coefficient).
    ("ball", "friction"): dsl.CompoundBinding(components=(
        dsl.SimBinding("ball_friction"),
        dsl.SimBinding("ball_rolling_friction"),
    )),
    ("ball", "restitution"): dsl.SimBinding("ball_restitution"),
    # launch_velocity is a 3-vector; the +X component dominates (baseline
    # 4.5 m/s down the court, plus a 4.5 m/s vertical component to arc
    # over the net). The DSL knob only edits the +X speed.
    ("ball", "initial_velocity"): dsl.SimBinding("launch_velocity", index=0),
}


# No DELETE bindings: single-object scene.
DELETE_BINDINGS: dict[str, dsl.DeleteBinding] = {}


VOCAB = dsl.Vocabulary(
    objects=OBJECTS,
    properties=PROPERTIES,
    sim_bindings=SIM_BINDINGS,
    delete_bindings=DELETE_BINDINGS,
    baseline_physics=BASELINE_PHYSICS,
)
