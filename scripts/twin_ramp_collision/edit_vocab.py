"""Edit vocabulary for the twin_ramp_collision scene.

Declares which marbles/properties are editable, how each maps to the sim's
physics parameter dict, and the baseline physics values. Feed this to
``pcve_edit_dsl`` and prompts / overrides / diffs come out data-driven.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pcve_edit_dsl as dsl  # noqa: E402


# Must stay in sync with render_twin_ramp_collision.create_scenario().
# Slot order: purple_ball = index 0 (ball_a, +X ramp, hue-shifted from
# the yellow base GLB to purple), yellow_ball = index 1 (ball_b, -X
# ramp, no hue shift so it keeps the base yellow). Both start at rest
# at their release positions and roll down under gravity.
BASELINE_PHYSICS = {
    "ball_radius": 0.032,
    "ball_mass": 0.343,
    "ball_friction": 0.30,
    "ball_restitution": 0.87,
    "ball_rolling_friction": 0.0012,
    "ball_spinning_friction": 0.004,
    # Per-ball baselines, explicit copies of the globals above so a PCVE
    # `mass` / `friction` / `restitution` edit has a well-defined `from`
    # value to diff against.
    "ball_a_mass": 0.343,
    "ball_a_friction": 0.30,
    "ball_a_restitution": 0.87,
    "ball_a_rolling_friction": 0.0012,
    "ball_b_mass": 0.343,
    "ball_b_friction": 0.30,
    "ball_b_restitution": 0.87,
    "ball_b_rolling_friction": 0.0012,
    "track_friction": 0.55,
    "track_restitution": 0.14,
    "release_inset": 0.030,
    "release_inset_bias": 0.004,
    "gravity_z": -9.8,
    # Two-slot presence list (purple_ball index 0 on +X ramp, yellow_ball
    # index 1 on -X ramp). A PCVE DELETE edit writes 0 at the ball's slot.
    "active": [1, 1],
}


OBJECTS = {
    "purple_ball": {"zh": "紫色玻璃球", "en": "the purple marble"},
    "yellow_ball": {"zh": "黄色玻璃球", "en": "the yellow marble"},
}
# Deliberately NOT editable: the wooden ramps or the plank in the valley.
# Bullet multiplies friction and restitution across the pair, so any
# effective coefficient reachable through the track is also reachable
# through either marble. ramp_angle_deg, release_inset*, ball_radius are
# scene-wide geometry rather than per-object physics and do not fit one
# of the four generic PCVE properties.
#
# The two marbles are the same size and same shape -- distinguishable
# only by the render's hue/saturation tint applied to the yellow base
# GLB: ball_a (+X ramp) has hue 0.12 which shifts to purple, ball_b (-X
# ramp) has hue 0.50 which is the Blender no-op and keeps the base
# yellow. Prompts refer to them by colour, following the pcve-suite
# skill rule that same-model objects must be colour-coded rather than
# named by position.


PROPERTIES = {
    "mass":        dsl.PropertySpec("scalar", "kg", "质量",     "mass"),
    # `friction` collapses lateral + rolling into a single "total
    # friction" knob. The marbles descend by rolling; editing lateral
    # alone would leave pure-rolling contact points doing no work.
    "friction":    dsl.PropertySpec("scalar", "",   "摩擦系数", "friction coefficient"),
    "restitution": dsl.PropertySpec("scalar", "",   "恢复系数", "restitution"),
    # Both marbles start at rest, so `initial_velocity` is not defined on
    # either -- gravity does all the launching.
}


SIM_BINDINGS = {
    ("purple_ball", "mass"): dsl.SimBinding("ball_a_mass"),
    ("yellow_ball",  "mass"): dsl.SimBinding("ball_b_mass"),

    # `friction` scales lateral + rolling together on each ball. First
    # component is the display value (what shows up in prompts).
    ("purple_ball", "friction"): dsl.CompoundBinding(components=(
        dsl.SimBinding("ball_a_friction"),
        dsl.SimBinding("ball_a_rolling_friction"),
    )),
    ("yellow_ball", "friction"): dsl.CompoundBinding(components=(
        dsl.SimBinding("ball_b_friction"),
        dsl.SimBinding("ball_b_rolling_friction"),
    )),

    ("purple_ball", "restitution"): dsl.SimBinding("ball_a_restitution"),
    ("yellow_ball",  "restitution"): dsl.SimBinding("ball_b_restitution"),
}


DELETE_BINDINGS = {
    "purple_ball": dsl.DeleteBinding("active", index=0),
    "yellow_ball":  dsl.DeleteBinding("active", index=1),
}


VOCAB = dsl.Vocabulary(
    objects=OBJECTS,
    properties=PROPERTIES,
    sim_bindings=SIM_BINDINGS,
    delete_bindings=DELETE_BINDINGS,
    baseline_physics=BASELINE_PHYSICS,
)
