"""Edit vocabulary for the table_drop_collision scene.

Declares which balls/properties are editable, how each maps to the sim's
physics parameter dict, and the baseline physics values. Feed this to
``pcve_edit_dsl`` and prompts / overrides / diffs come out data-driven.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pcve_edit_dsl as dsl  # noqa: E402


# Must stay in sync with render_table_drop_collision.create_scenario().
# Slot order: rolling_ball = index 0 (ball A, on the table, thrown +Y with
# launch_speed), target_ball = index 1 (ball B, sitting on the floor).
BASELINE_PHYSICS = {
    "launch_x": 0.315,
    "launch_y": -0.290,
    "launch_speed": 1.28,
    "launch_heading_deg": 180.0,
    "ball_a_radius": 0.0335,
    "ball_b_radius": 0.0335,
    "ball_a_mass": 0.057,
    "ball_b_mass": 0.057,
    "ball_b_x": -0.722,
    "ball_b_y": -0.290,
    "ball_a_restitution": 0.86,
    "ball_b_restitution": 0.86,
    "ball_a_friction": 0.62,
    "ball_b_friction": 0.62,
    "table_friction": 0.55,
    "table_restitution": 0.60,
    "table_rolling_friction": 0.0037,
    "table_spinning_friction": 0.004,
    "floor_friction": 0.58,
    "floor_restitution": 0.87,
    "floor_rolling_friction": 0.0110,
    "floor_spinning_friction": 0.004,
    "gravity_z": -9.8,
    # Two-slot presence list, in fixed order (rolling_ball, target_ball).
    # Only slot 1 (target) is actually removable in the sim -- the whole
    # scene is centred on the rolling ball A leaving the table, so it is
    # always kept.
    "active": [1, 1],
}


OBJECTS = {
    "rolling_ball": {"zh": "新网球", "en": "the new tennis ball"},
    "target_ball":  {"zh": "旧网球", "en": "the old tennis ball"},
}
# Deliberately NOT editable: the coffee table, the floor, or the room set
# dressing. Bullet multiplies friction and restitution across the pair, so
# any effective coefficient reachable through the table felt or the floor
# is also reachable through either ball. launch_heading_deg, launch_x /
# launch_y, ball_b_x / ball_b_y, and ball_radius are scene-wide
# setup/geometry rather than per-object physics, and do not fit one of
# the four generic PCVE properties.
#
# Two visually identical tennis balls -- distinguished by the render's
# hue/saturation/value tint: the "new" ball is bright optic yellow (index
# 0), the "old" ball is greener/duller/darker (index 1). See render's
# scenario["balls"] block. Prompts refer to them by that visible cue.


PROPERTIES = {
    "mass":        dsl.PropertySpec("scalar", "kg", "质量",     "mass"),
    # Tennis balls are rollers on the felt and on the floor; `friction`
    # collapses lateral, rolling, and spinning components into one knob.
    # In this sim the per-ball CLI only exposes lateral friction (--ball-a
    # -friction / --ball-b-friction) -- rolling and spinning friction live
    # on the surfaces (table, floor), so a lateral-only edit here is
    # sufficient: the ball's motion on the tabletop is dominated by
    # tabletop rolling friction times the ball's own lateral coefficient.
    "friction":    dsl.PropertySpec("scalar", "",   "摩擦系数", "friction coefficient"),
    "restitution": dsl.PropertySpec("scalar", "",   "恢复系数", "restitution"),
    # Only the rolling ball has a non-zero baseline velocity (the push
    # along +Y at launch_speed m/s along -X heading). The target ball
    # starts at rest, so this property is bound only on the rolling ball.
    "initial_velocity": dsl.PropertySpec("scalar", "m/s", "初速度大小", "initial speed"),
}


SIM_BINDINGS = {
    ("rolling_ball", "mass"): dsl.SimBinding("ball_a_mass"),
    ("target_ball",  "mass"): dsl.SimBinding("ball_b_mass"),

    ("rolling_ball", "friction"): dsl.SimBinding("ball_a_friction"),
    ("target_ball",  "friction"): dsl.SimBinding("ball_b_friction"),

    ("rolling_ball", "restitution"): dsl.SimBinding("ball_a_restitution"),
    ("target_ball",  "restitution"): dsl.SimBinding("ball_b_restitution"),

    # Scalar magnitude of the push given to ball A on the tabletop.
    # Direction is fixed by launch_heading_deg (180 deg = -X).
    ("rolling_ball", "initial_velocity"): dsl.SimBinding("launch_speed"),
}


DELETE_BINDINGS = {
    # Only ball B (the target) can be removed; ball A rolls off the table
    # and drives the whole simulation, so removing it is not supported.
    "target_ball": dsl.DeleteBinding("active", index=1),
}


VOCAB = dsl.Vocabulary(
    objects=OBJECTS,
    properties=PROPERTIES,
    sim_bindings=SIM_BINDINGS,
    delete_bindings=DELETE_BINDINGS,
    baseline_physics=BASELINE_PHYSICS,
)
