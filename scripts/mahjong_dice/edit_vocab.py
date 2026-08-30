"""Edit vocabulary for the mahjong_dice scene.

Declares which dice/properties are editable, how each maps to the sim's
physics parameter dict, and the baseline physics values. Feed this to
``pcve_edit_dsl`` and prompts / overrides / diffs come out data-driven.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pcve_edit_dsl as dsl  # noqa: E402


# Must stay in sync with render_mahjong_dice.create_scenario().
# Slot order: red_die = index 0 (dropped first, near tray centre, red
# material), white_die = index 1 (dropped 5 cm higher, near-white).
BASELINE_PHYSICS = {
    "die_edge": 0.0833,
    "die_masses": [0.006, 0.006],
    "die_frictions": [0.5, 0.5],
    "die_restitutions": [0.72, 0.72],
    "die_active": [1, 1],
    "die_initial_speeds": [1.5, 0.8],
    "die_mass": 0.006,
    "die_friction": 0.5,
    "die_restitution": 0.72,
    "floor_friction": 0.55,
    "floor_restitution": 0.72,
    "drop_height": 0.6,
    "floor_z": 0.89655,
    "die_0_xy": [-0.1612, -0.0052],
    "die_1_xy": [-0.1991, 0.1425],
    "gravity": [0.0, 0.0, -9.8],
}


OBJECTS = {
    "red_die":   {"zh": "红色骰子", "en": "the red die"},
    "white_die": {"zh": "白色骰子", "en": "the white die"},
}
# Deliberately NOT editable: the tray floor. Bullet multiplies the pair, so
# any effective coefficient reachable through the floor is also reachable
# through a die. drop_height is scene-wide and does not fit one of the four
# generic PCVE properties, so it is not exposed as an edit either.


PROPERTIES = {
    "mass":        dsl.PropertySpec("scalar", "kg", "质量",     "mass"),
    # Dice do not roll as a smooth ball would; `friction` is the die's single
    # lateral coefficient against the tray, no rolling split.
    "friction":    dsl.PropertySpec("scalar", "",   "摩擦系数", "friction coefficient"),
    "restitution": dsl.PropertySpec("scalar", "",   "恢复系数", "restitution"),
    # Scalar downward speed magnitude (the direction is fixed straight down
    # in the sim). Both dice have a non-zero baseline, so this is well-defined
    # on both.
    "initial_velocity": dsl.PropertySpec("scalar", "m/s", "初速度大小", "initial speed"),
}


# Bullet multiplies both friction and restitution across the pair, so an
# edit to die i's coefficient governs only the impacts die i is part of --
# its impacts against the floor and against the other die.
SIM_BINDINGS = {
    ("red_die", "mass"): dsl.SimBinding("die_masses", index=0),
    ("white_die", "mass"): dsl.SimBinding("die_masses", index=1),

    ("red_die", "friction"): dsl.SimBinding("die_frictions", index=0),
    ("white_die", "friction"): dsl.SimBinding("die_frictions", index=1),

    ("red_die", "restitution"): dsl.SimBinding("die_restitutions", index=0),
    ("white_die", "restitution"): dsl.SimBinding("die_restitutions", index=1),

    ("red_die", "initial_velocity"): dsl.SimBinding("die_initial_speeds", index=0),
    ("white_die", "initial_velocity"): dsl.SimBinding("die_initial_speeds", index=1),
}


DELETE_BINDINGS = {
    "red_die": dsl.DeleteBinding("die_active", index=0),
    "white_die": dsl.DeleteBinding("die_active", index=1),
}


VOCAB = dsl.Vocabulary(
    objects=OBJECTS,
    properties=PROPERTIES,
    sim_bindings=SIM_BINDINGS,
    delete_bindings=DELETE_BINDINGS,
    baseline_physics=BASELINE_PHYSICS,
)
