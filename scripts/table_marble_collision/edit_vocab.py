"""Edit vocabulary for the table_marble_collision scene.

Declares which marbles/properties are editable, how each maps to the sim's
physics parameter dict, and the baseline physics values. Feed this to
``pcve_edit_dsl`` and prompts / overrides / diffs come out data-driven.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pcve_edit_dsl as dsl  # noqa: E402


_GLASS_DENSITY = 2500.0  # kg/m^3, solid glass sphere


def _mass(radius: float) -> float:
    return (4.0 / 3.0) * math.pi * radius ** 3 * _GLASS_DENSITY


# Must stay in sync with render_table_marble_collision.create_scenario().
# Slot order: big_marble = index 0 (ball A, 100 mm diameter, rolled at
# launch_speed), small_marble = index 1 (ball B, 50 mm diameter, at rest).
BASELINE_PHYSICS = {
    "launch_x": -0.620,
    "launch_y": 0.020,
    "launch_speed": 1.31,
    "launch_heading_deg": 0.0,
    "ball_a_radius": 0.050,
    "ball_b_radius": 0.025,
    "ball_a_mass": _mass(0.050),
    "ball_b_mass": _mass(0.025),
    "ball_b_x": 0.060,
    "ball_b_y": 0.008,
    "ball_a_restitution": 0.87,
    "ball_b_restitution": 0.87,
    "ball_a_friction": 0.30,
    "ball_b_friction": 0.30,
    "table_friction": 0.42,
    "table_restitution": 0.30,
    "table_rolling_friction": 0.0200,
    "table_spinning_friction": 0.006,
    "gravity_z": -9.8,
    # Two-slot presence list, in fixed order (big, small). Only slot 1
    # (small_marble) is actually removable in the sim -- the big marble
    # drives the whole scene so it is always kept.
    "active": [1, 1],
}


OBJECTS = {
    "big_marble":   {"zh": "大玻璃球", "en": "the big marble"},
    "small_marble": {"zh": "小玻璃球", "en": "the small marble"},
}
# Deliberately NOT editable: the bar table, the walls, the floor, or the
# set dressing (candlesticks, tray). Bullet multiplies friction and
# restitution across the pair, so any effective coefficient reachable
# through the table felt is also reachable through either marble.
# launch_x / launch_y / launch_heading_deg / ball_b_x / ball_b_y and the
# radii are scene-wide geometry rather than per-marble physics -- they do
# not fit one of the four generic PCVE properties.
#
# The two marbles are already visually distinct (100 mm vs 50 mm), so the
# prompt naming by size (big / small) is unambiguous in the shot.


PROPERTIES = {
    "mass":        dsl.PropertySpec("scalar", "kg", "质量",     "mass"),
    # Marbles roll on the felt; `friction` is bound only to lateral because
    # the sim exposes rolling / spinning friction on the surfaces (table),
    # not on each marble. That is enough here: the felt's rolling friction
    # multiplies the marble's lateral coefficient to slow the roll.
    "friction":    dsl.PropertySpec("scalar", "",   "摩擦系数", "friction coefficient"),
    "restitution": dsl.PropertySpec("scalar", "",   "恢复系数", "restitution"),
    # The big marble is rolled at launch_speed along +X; the small marble
    # starts at rest with no direction to scale. Bound only on the big one.
    "initial_velocity": dsl.PropertySpec("scalar", "m/s", "初速度大小", "initial speed"),
}


SIM_BINDINGS = {
    ("big_marble",   "mass"): dsl.SimBinding("ball_a_mass"),
    ("small_marble", "mass"): dsl.SimBinding("ball_b_mass"),

    ("big_marble",   "friction"): dsl.SimBinding("ball_a_friction"),
    ("small_marble", "friction"): dsl.SimBinding("ball_b_friction"),

    ("big_marble",   "restitution"): dsl.SimBinding("ball_a_restitution"),
    ("small_marble", "restitution"): dsl.SimBinding("ball_b_restitution"),

    # Scalar magnitude of the push given to the big marble on the table.
    # Direction is fixed by launch_heading_deg (0 deg = +X).
    ("big_marble", "initial_velocity"): dsl.SimBinding("launch_speed"),
}


DELETE_BINDINGS = {
    # Only the small marble can be removed; the big one rolls the whole
    # scene and is not removable in the current sim.
    "small_marble": dsl.DeleteBinding("active", index=1),
}


VOCAB = dsl.Vocabulary(
    objects=OBJECTS,
    properties=PROPERTIES,
    sim_bindings=SIM_BINDINGS,
    delete_bindings=DELETE_BINDINGS,
    baseline_physics=BASELINE_PHYSICS,
)
