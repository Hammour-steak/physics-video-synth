"""Edit vocabulary for the domino_chain scene.

Declares which tiles/properties are editable, how each maps to the sim's
physics parameter dict, and the baseline physics values. Feed this to
``pcve_edit_dsl`` and prompts / overrides / diffs come out data-driven.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pcve_edit_dsl as dsl  # noqa: E402


# Must stay in sync with render_domino_chain.create_scenario().
# Row order: index 0 is the pushed tile (starts pre-tilted at 12 deg), then
# 1, 2, 3 follow along +X.
BASELINE_PHYSICS = {
    "domino_count": 4,
    "domino_spacing": 0.8,
    "domino_thickness": 0.20,
    "domino_width": 0.70,
    "domino_height": 1.30,
    "domino_masses": [0.12, 0.12, 0.12, 0.12],
    "domino_frictions": [0.6, 0.6, 0.6, 0.6],
    "domino_restitutions": [0.05, 0.05, 0.05, 0.05],
    "domino_active": [1, 1, 1, 1],
    "domino_mass": 0.12,
    "domino_friction": 0.6,
    "domino_restitution": 0.05,
    "floor_friction": 0.6,
    "push_angle_deg": 12.0,
    "floor_z": -0.0322,
    "gravity": [0.0, 0.0, -9.81],
    "scene_offset_x": 0.0,
    "scene_offset_y": 0.0,
}


OBJECTS = {
    "domino_1": {"zh": "第 1 张骨牌", "en": "the first domino"},
    "domino_2": {"zh": "第 2 张骨牌", "en": "the second domino"},
    "domino_3": {"zh": "第 3 张骨牌", "en": "the third domino"},
    "domino_4": {"zh": "第 4 张骨牌", "en": "the fourth domino"},
}
# Deliberately NOT editable: the floor. Bullet multiplies the pair, so any
# effective coefficient reachable through the floor is also reachable through
# a tile. push_angle_deg is scene-wide (only the first tile is pre-tilted)
# and does not map to one of the four generic PCVE properties, so it is not
# exposed as an edit either.


PROPERTIES = {
    "mass":         dsl.PropertySpec("scalar", "kg", "质量",     "mass"),
    # One friction knob per tile. Tiles slide/tip against the floor rather
    # than roll, so `friction` is simply the tile's lateral coefficient.
    "friction":     dsl.PropertySpec("scalar", "",   "摩擦系数", "friction coefficient"),
    "restitution":  dsl.PropertySpec("scalar", "",   "恢复系数", "restitution"),
}


# Bullet multiplies both friction and restitution across the pair, so an
# edit to tile i's coefficient governs only the impacts tile i is part of
# (tile i-1 hitting tile i, and tile i hitting tile i+1).
SIM_BINDINGS = {
    ("domino_1", "mass"): dsl.SimBinding("domino_masses", index=0),
    ("domino_2", "mass"): dsl.SimBinding("domino_masses", index=1),
    ("domino_3", "mass"): dsl.SimBinding("domino_masses", index=2),
    ("domino_4", "mass"): dsl.SimBinding("domino_masses", index=3),

    ("domino_1", "friction"): dsl.SimBinding("domino_frictions", index=0),
    ("domino_2", "friction"): dsl.SimBinding("domino_frictions", index=1),
    ("domino_3", "friction"): dsl.SimBinding("domino_frictions", index=2),
    ("domino_4", "friction"): dsl.SimBinding("domino_frictions", index=3),

    ("domino_1", "restitution"): dsl.SimBinding("domino_restitutions", index=0),
    ("domino_2", "restitution"): dsl.SimBinding("domino_restitutions", index=1),
    ("domino_3", "restitution"): dsl.SimBinding("domino_restitutions", index=2),
    ("domino_4", "restitution"): dsl.SimBinding("domino_restitutions", index=3),
}


DELETE_BINDINGS = {
    "domino_1": dsl.DeleteBinding("domino_active", index=0),
    "domino_2": dsl.DeleteBinding("domino_active", index=1),
    "domino_3": dsl.DeleteBinding("domino_active", index=2),
    "domino_4": dsl.DeleteBinding("domino_active", index=3),
}


VOCAB = dsl.Vocabulary(
    objects=OBJECTS,
    properties=PROPERTIES,
    sim_bindings=SIM_BINDINGS,
    delete_bindings=DELETE_BINDINGS,
    baseline_physics=BASELINE_PHYSICS,
)
