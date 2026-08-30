"""Edit vocabulary for the dining_chain scene.

Declares which objects/properties are editable, how each maps to the sim's
physics parameter dict, and the baseline physics values. Feed this to
``pcve_edit_dsl`` and prompts / overrides / diffs come out data-driven.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pcve_edit_dsl as dsl  # noqa: E402


# Must stay in sync with render_dining_chain.create_scenario().
# Every per-object list is in chain order (can, cup, milk): the can is
# pushed, the cup is the middle link, the milk carton is the last link.
BASELINE_PHYSICS = {
    "can_initial_speed": 3.3,
    "object_masses": [0.36, 0.30, 0.35],
    "object_frictions": [0.30, 0.30, 0.30],
    "object_restitutions": [0.10, 0.10, 0.10],
    "object_active": [1, 1, 1],
    "table_friction": 0.30,
    "table_restitution": 0.10,
    "gravity_z": -9.8,
}


OBJECTS = {
    "can":  {"zh": "可乐罐",   "en": "the cola can"},
    "cup":  {"zh": "汽水杯",   "en": "the soda cup"},
    "milk": {"zh": "牛奶盒",   "en": "the milk carton"},
}
# Deliberately NOT editable: the tabletop. Bullet multiplies the pair, so any
# effective coefficient reachable through the table is also reachable through
# any container, and "edit the table" reads more like a set change than a
# physics edit.


PROPERTIES = {
    "mass":             dsl.PropertySpec("scalar", "kg", "质量",     "mass"),
    # One friction knob per object -- the containers are flat-based sliders,
    # not rollers, so `friction` is simply the container's lateral coefficient.
    "friction":         dsl.PropertySpec("scalar", "",   "摩擦系数", "friction coefficient"),
    "restitution":      dsl.PropertySpec("scalar", "",   "恢复系数", "restitution"),
    # Scalar speed: magnitude only, direction stays the baseline's (+Y along
    # the chain). Only the can has a non-zero baseline velocity, so it is the
    # only object this property is bound on -- the other two start at rest and
    # have no direction to scale.
    "initial_velocity": dsl.PropertySpec("scalar", "m/s", "初速度大小", "initial speed"),
}


# Note on restitution: PyBullet multiplies the two bodies' values, so editing
# one container's restitution only affects the impacts it is part of. The
# can's value governs the first hit and the milk's the second; the cup's
# governs both. Friction is combined the same way against the table.
SIM_BINDINGS = {
    ("can",  "mass"): dsl.SimBinding("object_masses", index=0),
    ("cup",  "mass"): dsl.SimBinding("object_masses", index=1),
    ("milk", "mass"): dsl.SimBinding("object_masses", index=2),

    ("can",  "friction"): dsl.SimBinding("object_frictions", index=0),
    ("cup",  "friction"): dsl.SimBinding("object_frictions", index=1),
    ("milk", "friction"): dsl.SimBinding("object_frictions", index=2),

    ("can",  "restitution"): dsl.SimBinding("object_restitutions", index=0),
    ("cup",  "restitution"): dsl.SimBinding("object_restitutions", index=1),
    ("milk", "restitution"): dsl.SimBinding("object_restitutions", index=2),

    # Only the can starts with a non-zero velocity (the push).
    ("can", "initial_velocity"): dsl.SimBinding("can_initial_speed"),
}


DELETE_BINDINGS = {
    "can":  dsl.DeleteBinding("object_active", index=0),
    "cup":  dsl.DeleteBinding("object_active", index=1),
    "milk": dsl.DeleteBinding("object_active", index=2),
}


VOCAB = dsl.Vocabulary(
    objects=OBJECTS,
    properties=PROPERTIES,
    sim_bindings=SIM_BINDINGS,
    delete_bindings=DELETE_BINDINGS,
    baseline_physics=BASELINE_PHYSICS,
)
