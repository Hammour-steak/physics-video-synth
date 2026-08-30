"""Edit vocabulary for the picnic_apple_ball scene.

Declares which objects/properties are editable, how each maps to the sim's
physics parameter dict, and the baseline physics values. Feed this to
``pcve_edit_dsl`` and prompts / overrides / diffs come out data-driven.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pcve_edit_dsl as dsl  # noqa: E402


# Must stay in sync with render_picnic_apple_ball.create_scenario().
BASELINE_PHYSICS = {
    "ball_mass": 0.43,
    "ball_friction": 0.25,
    "ball_rolling_friction": 0.015,
    "ball_spinning_friction": 0.01,
    "ball_restitution": 0.35,
    "apple_mass": 0.15,
    "apple_friction": 0.5,
    "apple_restitution": 0.25,
    # Two-slot active list, in fixed order (apple, ball). A DELETE edit
    # writes 0 at the object's slot; the sim skips creating that body and
    # its frame slot is filled with the initial pose + present=false.
    "active": [1, 1],
    "grass_friction": 0.25,
    "drop_height": 1.3,
    "apple_offset_x": 0.105,
    "apple_offset_y": 0.0,
    "gravity_z": -9.8,
}


OBJECTS = {
    "apple":       {"zh": "苹果",       "en": "the apple"},
    "soccer_ball": {"zh": "足球",       "en": "the soccer ball"},
}
# Deliberately NOT editable: the grass ground. Bullet multiplies friction and
# restitution across the pair, so any effective coefficient reachable through
# the ground is also reachable through either sphere. drop_height and
# apple_offset_x are scene-wide (setup, not object physics) and do not fit
# one of the four generic PCVE properties, so they are not exposed as edits.


PROPERTIES = {
    "mass":        dsl.PropertySpec("scalar", "kg", "质量",     "mass"),
    # `friction` collapses lateral + rolling into a single "total friction"
    # knob. The soccer ball is a genuine roller so both channels matter --
    # a lateral-only edit would leave pure-rolling contact points doing no
    # work, and the ball would keep rolling forever. The apple has no
    # rolling channel exposed by the sim CLI, so it is bound to lateral
    # alone (same effective behaviour on this scene's apple-vs-ground and
    # apple-vs-ball impacts, where the apple mostly slides/bounces).
    "friction":    dsl.PropertySpec("scalar", "",   "摩擦系数", "friction coefficient"),
    "restitution": dsl.PropertySpec("scalar", "",   "恢复系数", "restitution"),
    # Both objects start at rest, so `initial_velocity` is not defined on
    # either -- there is no baseline direction to scale.
}


SIM_BINDINGS = {
    ("apple",       "mass"): dsl.SimBinding("apple_mass"),
    ("soccer_ball", "mass"): dsl.SimBinding("ball_mass"),

    # Apple has no exposed rolling knob in the sim CLI; bind friction to
    # lateral only.
    ("apple", "friction"): dsl.SimBinding("apple_friction"),
    # Ball: scale lateral and rolling together so the "make the ball
    # grippy" edit actually slows the rolling motion.
    ("soccer_ball", "friction"): dsl.CompoundBinding(components=(
        dsl.SimBinding("ball_friction"),           # display: the lateral coefficient
        dsl.SimBinding("ball_rolling_friction"),
    )),

    ("apple",       "restitution"): dsl.SimBinding("apple_restitution"),
    ("soccer_ball", "restitution"): dsl.SimBinding("ball_restitution"),
}


DELETE_BINDINGS = {
    "apple":       dsl.DeleteBinding("active", index=0),
    "soccer_ball": dsl.DeleteBinding("active", index=1),
}


VOCAB = dsl.Vocabulary(
    objects=OBJECTS,
    properties=PROPERTIES,
    sim_bindings=SIM_BINDINGS,
    delete_bindings=DELETE_BINDINGS,
    baseline_physics=BASELINE_PHYSICS,
)
