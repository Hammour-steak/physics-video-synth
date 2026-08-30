"""Edit vocabulary for the ball_block scene.

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


# Must stay in sync with render_ball_block_impact.create_scenario() evaluated
# at --physics-jitter 0, which is what the suite renders with. Every value the
# renderer jitters is written here at its un-jittered centre; the suite passes
# --physics-jitter 0 precisely so that an edit's "from" value is the truth.
BASELINE_PHYSICS = {
    "motion": "side_impact",
    "ball_radius": 0.34,
    "ball_initial_location": [-3.15, -0.12, 0.341],
    "ball_initial_velocity": [6.0, 0.0, 0.0],
    "block_location": [0.23, -0.02, 0.35],
    "block_yaw_deg": 0.0,
    "ball_mass": 0.58,
    "block_mass": 0.65,
    "floor_friction": 0.82,
    "ball_friction": 0.38,
    "ball_restitution": 0.78,
    "block_friction": 0.32,
    "block_restitution": 0.55,
    "ball_rolling_friction": 0.0015,
    "block_rolling_friction": 0.006,
    # A one-element list, not a bare int: pcve_edit_dsl writes DELETE through a
    # list index, and every other scene's active flags are lists too.
    "block_active": [1],
}


OBJECTS = {
    "ball":       {"zh": "红球",   "en": "the ball"},
    "wood_block": {"zh": "木块",   "en": "the wooden block"},
}


PROPERTIES = {
    "mass":             dsl.PropertySpec("scalar", "kg", "质量",       "mass"),
    # One friction knob per object, not split into lateral and rolling. Both
    # scale together with the ratio of TO/FROM; the value shown in prompts is
    # the object's *lateral* friction. That means "raise the ball's friction
    # by 40x" bumps lateral 0.38 -> 15.2 *and* rolling 0.0015 -> 0.06 -- which
    # is what actually slows a rolling ball. See the "friction knob" note at
    # the bottom of the file for why lateral alone would be invisible here.
    "friction":         dsl.PropertySpec("scalar", "",   "摩擦系数",    "friction coefficient"),
    "restitution":      dsl.PropertySpec("scalar", "",   "恢复系数",    "restitution"),
    # Scalar speed: magnitude only, direction stays the baseline's. The ball is
    # the only body with a non-zero baseline velocity, so it is the only object
    # this property is bound on.
    "initial_velocity": dsl.PropertySpec("scalar", "m/s", "初速度大小", "initial speed"),
}


SIM_BINDINGS = {
    ("ball", "mass"):        dsl.SimBinding("ball_mass"),
    ("ball", "friction"):    dsl.CompoundBinding(components=(
        dsl.SimBinding("ball_friction"),           # display; the lateral coefficient
        dsl.SimBinding("ball_rolling_friction"),
    )),
    ("ball", "restitution"): dsl.SimBinding("ball_restitution"),
    # The baseline velocity is (6, 0, 0), i.e. purely along +x, so writing the
    # magnitude into slot 0 and leaving y/z at zero is exactly "scale the speed
    # along the baseline direction". This binding is only correct while the
    # baseline stays axis-aligned -- if a future baseline gives the ball a
    # sideways component, this needs a real scalar speed parameter instead.
    ("ball", "initial_velocity"): dsl.SimBinding("ball_initial_velocity", index=0),

    ("wood_block", "mass"):        dsl.SimBinding("block_mass"),
    ("wood_block", "friction"):    dsl.CompoundBinding(components=(
        dsl.SimBinding("block_friction"),          # display; the lateral coefficient
        dsl.SimBinding("block_rolling_friction"),
    )),
    ("wood_block", "restitution"): dsl.SimBinding("block_restitution"),
}


# The "friction knob" note. Bullet's rolling resistance combines as
# `rf_a * lat_b + rf_b * lat_a`. If we bound only `lateral_friction` on the
# ball, raising it would multiply in the block's rolling friction (0.006) but
# only at the moment of contact -- the ball is otherwise rolling without
# slipping across the floor, so lateral does no work there and the ball would
# not actually slow down. Scaling `lateral` and `rolling` together fixes this:
# raising `ball.friction` scales `ball_rolling_friction`, which multiplies
# against `floor_friction = 0.82` to slow the roll for the whole approach.

# Deliberately NOT bound: the floor's friction. Edits name a moving object, not
# the ground it moves on. Bullet multiplies the pair anyway, so any effective
# coefficient reachable by editing the floor is reachable by editing the ball
# or the block instead.


# Only the block can be deleted. Removing the ball would leave a scene in which
# nothing ever happens -- not an edit whose effect a model could be scored on.
DELETE_BINDINGS = {
    "wood_block": dsl.DeleteBinding("block_active", index=0, value_when_removed=0),
}


VOCAB = dsl.Vocabulary(
    objects=OBJECTS,
    properties=PROPERTIES,
    sim_bindings=SIM_BINDINGS,
    delete_bindings=DELETE_BINDINGS,
    baseline_physics=BASELINE_PHYSICS,
)
