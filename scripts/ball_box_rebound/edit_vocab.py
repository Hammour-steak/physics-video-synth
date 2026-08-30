"""Edit vocabulary for the ball_box_rebound scene.

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


# Must stay in sync with render_ball_box_rebound.create_scenario()'s physics
# block. Nothing here is randomized -- this scene's scenario is fully pinned --
# so these are exactly the values the source video runs on.
BASELINE_PHYSICS = {
    "launch_x": -0.744,
    "launch_y": 1.030,
    "launch_speed": 2.60,
    "launch_heading_deg": 303.74,
    "ball_b_x": 0.358,
    "ball_b_y": 0.660,
    "chest_restitution": 0.80,
    "chest_friction": 0.55,
    "floor_rolling_friction": 0.0060,
    "floor_spinning_friction": 0.008,
    "floor_friction": 0.60,
    "floor_restitution": 0.45,
    "ball_a_mass": 0.120,
    "ball_a_restitution": 0.88,
    "ball_a_friction": 1.0,
    "ball_b_mass": 0.090,
    "ball_b_restitution": 0.80,
    "ball_b_friction": 1.0,
    "gravity_z": -9.8,
    # One-element list so the DSL can write DELETE by index, matching the
    # other scenes' active flags.
    "ball_b_active": [1],
}


OBJECTS = {
    "ball_a": {"zh": "星星球",  "en": "the star ball"},
    "ball_b": {"zh": "小足球",  "en": "the little football"},
}


PROPERTIES = {
    "mass":             dsl.PropertySpec("scalar", "kg", "质量",       "mass"),
    # One friction knob per object, not split into lateral and rolling. The
    # rolling ball is held at `rolling_friction = 0` in the simulator, so
    # `friction` here is just the lateral coefficient -- the exact combination
    # rule is spelled out in the "NOT bound" note below.
    "friction":         dsl.PropertySpec("scalar", "",   "摩擦系数",    "friction coefficient"),
    "restitution":      dsl.PropertySpec("scalar", "",   "恢复系数",    "restitution"),
    # Scalar speed: magnitude only. The heading stays the baseline's, which is
    # what makes this a legal one-scalar edit -- the aim is not touched, only
    # how hard the ball is pushed along it.
    "initial_velocity": dsl.PropertySpec("scalar", "m/s", "初速度大小", "initial speed"),
}


# On restitution: Bullet multiplies the two bodies' values, so lowering the
# ball's restitution shortens the *normal* component of the rebound and leaves
# the tangential one alone, which turns the ball off the chest panel at a
# flatter angle rather than merely slowing it down. Only the moving side of
# the impact pair is editable -- the chest is a static room panel, not a prop,
# and edits should describe how the ball behaves, not how the set is rebuilt.
SIM_BINDINGS = {
    ("ball_a", "mass"):             dsl.SimBinding("ball_a_mass"),
    ("ball_a", "friction"):         dsl.SimBinding("ball_a_friction"),
    ("ball_a", "restitution"):      dsl.SimBinding("ball_a_restitution"),
    # The push. Direction is launch_heading_deg and is deliberately not
    # editable: re-aiming the shot is a different scene, not an edit to it.
    ("ball_a", "initial_velocity"): dsl.SimBinding("launch_speed"),

    ("ball_b", "mass"):             dsl.SimBinding("ball_b_mass"),
    ("ball_b", "friction"):         dsl.SimBinding("ball_b_friction"),
    ("ball_b", "restitution"):      dsl.SimBinding("ball_b_restitution"),

    # The chest is a static piece of the room the ball bounces off; it has no
    # editable properties. Rebound liveliness is edited on ball_a's side
    # (edit_dead_ball), and the chest's own friction was empirically saturated
    # in the sweep (0.2 / 1.0 / 1.5 give bit-identical output).
}

# Deliberately NOT bound, and why -- both come out of the same Bullet rule,
# `rolling_combined = rf_a * lateral_b + rf_b * lateral_a`:
#
#   * The floor's friction, rolling friction and restitution. Edits name a
#     moving object, not the ground it moves on. Nothing is lost: with both
#     balls' own rolling friction held at 0, the effective rolling resistance
#     is `floor_rolling_friction * ball_a_friction`, so raising the *ball's*
#     lateral friction from 1 to 5 reproduces raising the floor's rolling
#     friction from 0.006 to 0.03 exactly -- verified identical to the
#     millimetre -- and, unlike the floor edit, it leaves the bounce off the
#     chest untouched.
#
#   * Rolling friction on either ball. Raising it multiplies in the *chest's*
#     lateral friction at the bounce as well, so the ball comes off the panel
#     at 10 deg instead of 30 -- it stops being a "how far does it run" edit
#     and starts bending the rebound, which is what the restitution edits are
#     for. `ball_a.lateral_friction` is the clean knob here. The same trap
#     applies to spinning friction.


# Only the target football can be deleted. Removing the rolling ball would
# leave a video in which nothing ever happens, and the chest and the floor
# are the set itself.
DELETE_BINDINGS = {
    "ball_b": dsl.DeleteBinding("ball_b_active", index=0, value_when_removed=0),
}


VOCAB = dsl.Vocabulary(
    objects=OBJECTS,
    properties=PROPERTIES,
    sim_bindings=SIM_BINDINGS,
    delete_bindings=DELETE_BINDINGS,
    baseline_physics=BASELINE_PHYSICS,
)
