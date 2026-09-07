"""Edit vocabulary for the pool_collision scene.

Declares which balls/properties are editable, how each maps to the sim's
physics parameter dict, and the baseline physics values. Feed this to
``pcve_edit_dsl`` and prompts / overrides / diffs come out data-driven.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pcve_edit_dsl as dsl  # noqa: E402


# Must stay in sync with render_pool_collision.create_scenario().
# Slot order: cue_ball = index 0 (the moving striker), target_ball = index 1
# (the one at rest).
# Measured off the table model; build_scene writes this same value back over
# the scenario before the simulator runs. It read 0.05715 here (the ball's
# *diameter*) until ADD arrived -- nothing consumed it, so nothing was wrong --
# but the GEOMETRY anchors below turn "N radii" into metres with it, and at the
# diameter every ADD would sit at twice the distance its prompt states.
BASELINE_PHYSICS = {
    "ball_radius": 0.0288317501544952,
    # Per-ball fields. An edit names one ball and writes at its slot.
    "cue_mass": 0.17,
    "cue_friction": 0.15,
    "cue_restitution": 0.90,
    "cue_rolling_friction": 0.02,
    "cue_spinning_friction": 0.02,
    "target_mass": 0.17,
    "target_friction": 0.15,
    "target_restitution": 0.90,
    "target_rolling_friction": 0.02,
    "target_spinning_friction": 0.02,
    # The yellow one-ball's slot. It is a real ball in the table model -- the
    # GLB ships a full rack and the render hides the ones the shot does not
    # use -- so ADD does not conjure geometry, it turns on a body the scene
    # already knows how to draw. Its physics matches the other two; only its
    # presence and its position are edits.
    "extra_mass": 0.17,
    "extra_friction": 0.15,
    "extra_restitution": 0.90,
    "extra_rolling_friction": 0.02,
    "extra_spinning_friction": 0.02,
    # Where the ADDed ball would sit if it were present. Baseline is a
    # placeholder that nothing reads while slot 2 of `active` is 0; an ADD
    # edit overwrites it with the centre the edit resolves to.
    "extra_initial_location": [0.0, -0.3, 0.0],
    # Slot order: cue = 0, target = 1, the ADDable yellow ball = 2. The third
    # slot reads 0 at baseline, which is what makes ADD the exact mirror of
    # DELETE: both edits write one entry of this list.
    "active": [1, 1, 0],
    # Globals kept as CLI fallbacks in the sim.
    "ball_mass": 0.17,
    "ball_friction": 0.15,
    "ball_restitution": 0.90,
    "ball_rolling_friction": 0.02,
    "ball_spinning_friction": 0.02,
    "table_friction": 0.08,
    "table_restitution": 0.10,
    "gravity": [0.0, 0.0, -9.81],
    "cue_initial_location": [0.0, -0.6, 0.0],
    "target_initial_location": [0.0, 0.0, 0.0],
    "cue_initial_velocity": [0.0, 1.0, 0.0],
}


OBJECTS = {
    "cue_ball":    {"zh": "母球",   "en": "the cue ball"},
    "target_ball": {"zh": "目标球", "en": "the target ball"},
    # Not on the felt at baseline; only an ADD edit puts it there. Named by
    # its colour and number rather than by a slot index, because that is how a
    # viewer would pick it out of the rack.
    "yellow_ball": {"zh": "黄色1号球", "en": "the yellow one-ball"},
}
# Deliberately NOT editable: the table felt and the cushion walls. Bullet
# multiplies friction and restitution across the pair, so any effective
# coefficient reachable through the felt is also reachable through either
# ball. cue_initial_location, ball_radius, and gravity are scene-wide
# (setup, not per-object physics) and do not fit one of the four generic
# PCVE properties, so they are not exposed as edits.


PROPERTIES = {
    "mass":        dsl.PropertySpec("scalar", "kg", "质量",     "mass"),
    # Pool balls are rollers on felt; `friction` collapses lateral and
    # rolling into a single "total friction" knob. Editing lateral alone
    # would leave pure-rolling contact points doing no work, and the ball
    # would keep rolling forever.
    "friction":    dsl.PropertySpec("scalar", "",   "摩擦系数", "friction coefficient"),
    "restitution": dsl.PropertySpec("scalar", "",   "恢复系数", "restitution"),
    # Scalar speed magnitude; the direction is the baseline's (down the
    # table). Only the cue ball has a non-zero baseline velocity, so this
    # property is bound only on it.
    "initial_velocity": dsl.PropertySpec("scalar", "m/s", "初速度大小", "initial speed"),
}


SIM_BINDINGS = {
    ("cue_ball",    "mass"): dsl.SimBinding("cue_mass"),
    ("target_ball", "mass"): dsl.SimBinding("target_mass"),

    # `friction` scales lateral + rolling together. First component is the
    # display value (what shows up in prompts as "the friction coefficient").
    ("cue_ball", "friction"): dsl.CompoundBinding(components=(
        dsl.SimBinding("cue_friction"),
        dsl.SimBinding("cue_rolling_friction"),
    )),
    ("target_ball", "friction"): dsl.CompoundBinding(components=(
        dsl.SimBinding("target_friction"),
        dsl.SimBinding("target_rolling_friction"),
    )),

    ("cue_ball",    "restitution"): dsl.SimBinding("cue_restitution"),
    ("target_ball", "restitution"): dsl.SimBinding("target_restitution"),

    # cue_initial_velocity is a 3-vector; the sim reads it component-wise.
    # For a scalar `initial_velocity` edit we bind it to the +Y component
    # (the direction of the baseline break) via a per-index SimBinding.
    ("cue_ball", "initial_velocity"): dsl.SimBinding("cue_initial_velocity", index=1),
}


DELETE_BINDINGS = {
    "cue_ball":    dsl.DeleteBinding("active", index=0),
    "target_ball": dsl.DeleteBinding("active", index=1),
    # The yellow ball is absent at baseline, so there is nothing to delete.
}


# Where each ball's centre and radius live, so an ADD stated in radii can be
# resolved into table coordinates. All three balls are the same regulation
# 57.15 mm sphere and share one radius parameter, so "3 radii" means the same
# distance whichever ball it is measured from in this scene.
GEOMETRY = {
    "cue_ball":    dsl.GeometryAnchor("cue_initial_location", "ball_radius"),
    "target_ball": dsl.GeometryAnchor("target_initial_location", "ball_radius"),
    "yellow_ball": dsl.GeometryAnchor("extra_initial_location", "ball_radius"),
}


# ADD turns slot 2 of `active` on and writes the resolved centre. The ball's
# own physics is already in BASELINE_PHYSICS under `extra_*` and needs no
# defaults here: those keys are always passed to the sim, and simply have no
# effect while the slot is off.
ADD_BINDINGS = {
    "yellow_ball": dsl.AddBinding(
        presence_key="active",
        presence_index=2,
        location_key="extra_initial_location",
    ),
}


# Frames in this suite's videos: build_pcve_pool_collision.py renders 4.0 s at 24 fps. An
# "AT FRAME n" edit is bounded by this, and the vague prompt reads n against it
# to say whether the edit lands early, midway or late in the clip.
TOTAL_FRAMES = 96


VOCAB = dsl.Vocabulary(
    objects=OBJECTS,
    properties=PROPERTIES,
    sim_bindings=SIM_BINDINGS,
    delete_bindings=DELETE_BINDINGS,
    baseline_physics=BASELINE_PHYSICS,
    geometry=GEOMETRY,
    add_bindings=ADD_BINDINGS,
    total_frames=TOTAL_FRAMES,
)
