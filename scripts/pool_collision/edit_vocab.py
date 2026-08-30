"""Edit vocabulary for the pool_collision scene.

Declares which balls/properties are editable, how each maps to the sim's
physics parameter dict, and the baseline physics values. Feed this to
``pcve_edit_dsl`` and prompts / overrides / diffs come out data-driven.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pcve_edit_dsl as dsl  # noqa: E402


# Must stay in sync with render_pool_collision.create_scenario().
# Slot order: cue_ball = index 0 (the moving striker), target_ball = index 1
# (the one at rest).
BASELINE_PHYSICS = {
    "ball_radius": 0.05715,
    # Per-ball fields. An edit names one ball and writes at its slot.
    "cue_mass": 0.17,
    "cue_friction": 0.15,
    "cue_restitution": 0.90,
    "cue_rolling_friction": 0.02,
    "cue_spinning_friction": 0.02,
    "target_mass": 0.17,
    "target_friction": 0.15,
    "target_restitution": 0.90,
    "target_rolling_friction": 0.02,
    "target_spinning_friction": 0.02,
    "active": [1, 1],
    # Globals kept as CLI fallbacks in the sim.
    "ball_mass": 0.17,
    "ball_friction": 0.15,
    "ball_restitution": 0.90,
    "ball_rolling_friction": 0.02,
    "ball_spinning_friction": 0.02,
    "table_friction": 0.08,
    "table_restitution": 0.10,
    "gravity": [0.0, 0.0, -9.81],
    "cue_initial_location": [0.0, -0.6, 0.0],
    "target_initial_location": [0.0, 0.0, 0.0],
    "cue_initial_velocity": [0.0, 1.0, 0.0],
}


OBJECTS = {
    "cue_ball":    {"zh": "母球",   "en": "the cue ball"},
    "target_ball": {"zh": "目标球", "en": "the target ball"},
}
# Deliberately NOT editable: the table felt and the cushion walls. Bullet
# multiplies friction and restitution across the pair, so any effective
# coefficient reachable through the felt is also reachable through either
# ball. cue_initial_location, ball_radius, and gravity are scene-wide
# (setup, not per-object physics) and do not fit one of the four generic
# PCVE properties, so they are not exposed as edits.


PROPERTIES = {
    "mass":        dsl.PropertySpec("scalar", "kg", "质量",     "mass"),
    # Pool balls are rollers on felt; `friction` collapses lateral and
    # rolling into a single "total friction" knob. Editing lateral alone
    # would leave pure-rolling contact points doing no work, and the ball
    # would keep rolling forever.
    "friction":    dsl.PropertySpec("scalar", "",   "摩擦系数", "friction coefficient"),
    "restitution": dsl.PropertySpec("scalar", "",   "恢复系数", "restitution"),
    # Scalar speed magnitude; the direction is the baseline's (down the
    # table). Only the cue ball has a non-zero baseline velocity, so this
    # property is bound only on it.
    "initial_velocity": dsl.PropertySpec("scalar", "m/s", "初速度大小", "initial speed"),
}


SIM_BINDINGS = {
    ("cue_ball",    "mass"): dsl.SimBinding("cue_mass"),
    ("target_ball", "mass"): dsl.SimBinding("target_mass"),

    # `friction` scales lateral + rolling together. First component is the
    # display value (what shows up in prompts as "the friction coefficient").
    ("cue_ball", "friction"): dsl.CompoundBinding(components=(
        dsl.SimBinding("cue_friction"),
        dsl.SimBinding("cue_rolling_friction"),
    )),
    ("target_ball", "friction"): dsl.CompoundBinding(components=(
        dsl.SimBinding("target_friction"),
        dsl.SimBinding("target_rolling_friction"),
    )),

    ("cue_ball",    "restitution"): dsl.SimBinding("cue_restitution"),
    ("target_ball", "restitution"): dsl.SimBinding("target_restitution"),

    # cue_initial_velocity is a 3-vector; the sim reads it component-wise.
    # For a scalar `initial_velocity` edit we bind it to the +Y component
    # (the direction of the baseline break) via a per-index SimBinding.
    ("cue_ball", "initial_velocity"): dsl.SimBinding("cue_initial_velocity", index=1),
}


DELETE_BINDINGS = {
    "cue_ball":    dsl.DeleteBinding("active", index=0),
    "target_ball": dsl.DeleteBinding("active", index=1),
}


VOCAB = dsl.Vocabulary(
    objects=OBJECTS,
    properties=PROPERTIES,
    sim_bindings=SIM_BINDINGS,
    delete_bindings=DELETE_BINDINGS,
    baseline_physics=BASELINE_PHYSICS,
)
