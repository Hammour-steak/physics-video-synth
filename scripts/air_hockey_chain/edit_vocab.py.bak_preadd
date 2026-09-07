"""Edit vocabulary for the air_hockey_chain scene.

Declares which objects/properties are editable, how each maps to the sim's
physics parameter dict, and the baseline physics values. Feed this to
``pcve_edit_dsl`` and prompts / overrides / diffs come out data-driven.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Import the scene-agnostic DSL from scripts/pcve_edit_dsl.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pcve_edit_dsl as dsl  # noqa: E402


# Must stay in sync with render_air_hockey_chain.create_scenario().
# Every per-mallet list is in relay order: blue (pushed), red (middle), white
# (last). Keeping them as lists is what lets an edit name one disc.
BASELINE_PHYSICS = {
    "push_speed": 0.8,
    "mallet_masses": [0.12, 0.12, 0.12],
    "mallet_restitutions": [0.95, 0.95, 0.95],
    "mallet_frictions": [0.06, 0.06, 0.06],
    "mallet_active": [1, 1, 1],
    "surface_friction": 0.06,
    "table_restitution": 0.10,
    "gravity_z": -9.8,
}


OBJECTS = {
    "blue_mallet":  {"zh": "蓝色球槌", "en": "the blue mallet"},
    "red_mallet":   {"zh": "红色球槌", "en": "the red mallet"},
    "white_mallet": {"zh": "白色球槌", "en": "the white mallet"},
}


PROPERTIES = {
    "mass":             dsl.PropertySpec("scalar", "kg", "质量",     "mass"),
    # One friction knob per object, not split into lateral and rolling. Mallets
    # in this scene are pure sliders (no rolling), so `friction` is simply the
    # mallet's lateral coefficient.
    "friction":         dsl.PropertySpec("scalar", "",   "摩擦系数", "friction coefficient"),
    "restitution":      dsl.PropertySpec("scalar", "",   "恢复系数", "restitution"),
    # Scalar speed: magnitude only, direction stays the baseline's (down the
    # table, away from the camera). Only the blue mallet has a non-zero
    # baseline velocity, so it is the only disc this property is bound on --
    # the other two start at rest and have no direction to scale.
    "initial_velocity": dsl.PropertySpec("scalar", "m/s", "初速度大小", "initial speed"),
}


# Note on restitution: PyBullet multiplies the two bodies' values, so setting
# one mallet's restitution changes only the impacts that mallet is part of.
# Blue's value governs the first handoff and white's the second; red's governs
# both. Same for friction, which is multiplied against the table's.
SIM_BINDINGS = {
    ("blue_mallet",  "mass"): dsl.SimBinding("mallet_masses", index=0),
    ("red_mallet",   "mass"): dsl.SimBinding("mallet_masses", index=1),
    ("white_mallet", "mass"): dsl.SimBinding("mallet_masses", index=2),

    ("blue_mallet",  "restitution"): dsl.SimBinding("mallet_restitutions", index=0),
    ("red_mallet",   "restitution"): dsl.SimBinding("mallet_restitutions", index=1),
    ("white_mallet", "restitution"): dsl.SimBinding("mallet_restitutions", index=2),

    ("blue_mallet",  "friction"):         dsl.SimBinding("mallet_frictions", index=0),
    ("red_mallet",   "friction"):         dsl.SimBinding("mallet_frictions", index=1),
    ("white_mallet", "friction"):         dsl.SimBinding("mallet_frictions", index=2),

    # The push is the blue mallet's initial speed along the relay direction.
    ("blue_mallet", "initial_velocity"): dsl.SimBinding("push_speed"),
}

# Deliberately NOT bound: the table surface's own friction or restitution.
# Edits name a moving object, not the ground it moves on -- "make the puck
# grippy" is a property of a thing in the shot, while "make the table grippy"
# edits the set. It costs nothing here: PyBullet multiplies the pair, so
# raising one mallet's friction to 1.5 gives exactly the effective 0.09 that
# raising the table's to 1.5 would, and the sweep confirms the two are
# identical to the millimetre.


DELETE_BINDINGS = {
    "blue_mallet":  dsl.DeleteBinding("mallet_active", index=0),
    "red_mallet":   dsl.DeleteBinding("mallet_active", index=1),
    "white_mallet": dsl.DeleteBinding("mallet_active", index=2),
}


VOCAB = dsl.Vocabulary(
    objects=OBJECTS,
    properties=PROPERTIES,
    sim_bindings=SIM_BINDINGS,
    delete_bindings=DELETE_BINDINGS,
    baseline_physics=BASELINE_PHYSICS,
)
