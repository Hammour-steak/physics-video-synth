"""Edit vocabulary for the toy_car_ball scene.

Declares which objects/properties are editable, how each maps to the sim's
physics parameter dict, and the baseline physics values. Feed this to
``pcve_edit_dsl`` and prompts / overrides / diffs come out data-driven.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pcve_edit_dsl as dsl  # noqa: E402


# Must stay in sync with render_toy_car_ball.create_scenario().
# Slot order: toy_car = index 0 (the driver, pushed at launch_speed along
# -X), toy_ball = index 1 (the target, sitting near the far edge).
BASELINE_PHYSICS = {
    "car_mass": 0.35,
    "car_friction": 0.22,
    "car_restitution": 0.1,
    "ball_mass": 0.05,
    "ball_friction": 0.3,
    "ball_restitution": 0.6,
    "table_friction": 0.2,
    "floor_friction": 0.8,
    "launch_speed": 0.6,
    "car_start_x": 0.1,
    # (baseline computed by render; recorded here just for the diff)
    "ball_start_x": -0.29604,
    "gravity": [0.0, 0.0, -9.8],
    # Two-slot presence list (car, ball). Only slot 1 (ball) is removable
    # in the sim -- the car drives the whole scene so it is always kept.
    "active": [1, 1],
}


OBJECTS = {
    "toy_car":  {"zh": "玩具小车", "en": "the toy car"},
    "toy_ball": {"zh": "玩具球",   "en": "the toy ball"},
}
# Deliberately NOT editable: the table, the floor, the potted plant. Bullet
# multiplies friction and restitution across the pair, so any effective
# coefficient reachable through the table is also reachable through either
# moving object. car_start_x / ball_start_x are scene-wide geometry
# (starting positions), not per-object physics, and do not fit one of the
# four generic PCVE properties.


PROPERTIES = {
    "mass":        dsl.PropertySpec("scalar", "kg", "质量",     "mass"),
    # The sim exposes only lateral friction per object (rolling/spinning
    # are hard-coded small defaults). `friction` binds to lateral only --
    # enough here because the car slides across the table and the ball's
    # ground roll after landing is governed by floor friction times the
    # ball's own coefficient.
    "friction":    dsl.PropertySpec("scalar", "",   "摩擦系数", "friction coefficient"),
    "restitution": dsl.PropertySpec("scalar", "",   "恢复系数", "restitution"),
    # Scalar magnitude of the push given to the car along the table (-X
    # direction). Only the car has a non-zero baseline velocity.
    "initial_velocity": dsl.PropertySpec("scalar", "m/s", "初速度大小", "initial speed"),
}


SIM_BINDINGS = {
    ("toy_car",  "mass"): dsl.SimBinding("car_mass"),
    ("toy_ball", "mass"): dsl.SimBinding("ball_mass"),

    ("toy_car",  "friction"): dsl.SimBinding("car_friction"),
    ("toy_ball", "friction"): dsl.SimBinding("ball_friction"),

    ("toy_car",  "restitution"): dsl.SimBinding("car_restitution"),
    ("toy_ball", "restitution"): dsl.SimBinding("ball_restitution"),

    ("toy_car", "initial_velocity"): dsl.SimBinding("launch_speed"),
}


DELETE_BINDINGS = {
    # Only the ball can be removed; the car drives the whole sim.
    "toy_ball": dsl.DeleteBinding("active", index=1),
}


VOCAB = dsl.Vocabulary(
    objects=OBJECTS,
    properties=PROPERTIES,
    sim_bindings=SIM_BINDINGS,
    delete_bindings=DELETE_BINDINGS,
    baseline_physics=BASELINE_PHYSICS,
)
