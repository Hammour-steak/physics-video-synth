"""Edit vocabulary for the air_hockey_chain scene.

Declares which objects/properties are editable, how each maps to the sim's
physics parameter dict, and the baseline physics values. Feed this to
``pcve_edit_dsl`` and prompts / overrides / diffs come out data-driven.
"""

from __future__ import annotations

import math
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
    "mallet_masses": [0.12, 0.12, 0.12, 0.12],
    "mallet_restitutions": [0.95, 0.95, 0.95, 0.95],
    "mallet_frictions": [0.06, 0.06, 0.06, 0.06],
    "mallet_active": [1, 1, 1, 0],
    "surface_friction": 0.06,
    "table_restitution": 0.10,
    "gravity_z": -9.8,
}


# Geometry the ADD edits need. Radius is measured off the table model; the
# start centres are the quarter/half/three-quarter points the relay is built
# on, and the green slot's centre is a placeholder until an edit overwrites it.
_MALLET_RADIUS = 0.0620
_SEAT_Z = 0.0328


OBJECTS = {
    "blue_mallet":  {"zh": "蓝色球槌", "en": "the blue mallet"},
    "red_mallet":   {"zh": "红色球槌", "en": "the red mallet"},
    "white_mallet": {"zh": "白色球槌", "en": "the white mallet"},
    # Not on the table at baseline; only an ADD edit puts it there.
    "green_mallet": {"zh": "绿色球槌", "en": "the green mallet"},
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


# --- ADD surface -------------------------------------------------------------

BASELINE_PHYSICS["mallet_radius"] = _MALLET_RADIUS
BASELINE_PHYSICS["mallet_0_initial_location"] = [1.79325, 0.0, _SEAT_Z]
BASELINE_PHYSICS["mallet_1_initial_location"] = [1.19550, 0.0, _SEAT_Z]
BASELINE_PHYSICS["mallet_2_initial_location"] = [0.59775, 0.0, _SEAT_Z]
BASELINE_PHYSICS["mallet_3_initial_location"] = [0.94750, 0.0, _SEAT_Z]


GEOMETRY = {
    "blue_mallet":  dsl.GeometryAnchor("mallet_0_initial_location", "mallet_radius"),
    "red_mallet":   dsl.GeometryAnchor("mallet_1_initial_location", "mallet_radius"),
    "white_mallet": dsl.GeometryAnchor("mallet_2_initial_location", "mallet_radius"),
    "green_mallet": dsl.GeometryAnchor("mallet_3_initial_location", "mallet_radius"),
}


ADD_BINDINGS = {
    "green_mallet": dsl.AddBinding(
        presence_key="mallet_active",
        presence_index=3,
        location_key="mallet_3_initial_location",
    ),
}


# The band an ADD edit in this scene must stay inside, and why it exists.
#
# The relay is a chain of near-elastic equal-mass impacts, so each handoff
# should pass (1+e)/2 = 0.951 of the speed along and a four-mallet chain should
# deliver 0.951**3 = 0.860 of the original push to the last disc. It does --
# but only for some placements. Sweeping the green mallet across the legal
# range at the suite's own settings gives:
#
#     k from red   2.5    3.0    3.5    4.0    4.5    5.0    5.5    6.0    7.0
#     efficiency  0.712  0.729  0.866  0.860  0.823  0.862  0.767  0.693  0.848
#
# The 3.5-5.0 run matches theory; the others quietly lose energy in the
# contact solver. This is not something the scene can be tuned out of: solver
# iterations 200 -> 1000 change the pattern not at all, split impulse and the
# restitution velocity threshold do not help, and raising substeps breaks even
# the untouched three-mallet baseline. Contact detection is landing at
# different phases of the substep grid depending on where the disc stands.
#
# So ADD edits here are restricted to the one contiguous verified band, which
# has correct neighbours on both sides and therefore tolerates a placement
# error of about half a radius.
ADD_RADII_BAND = (3.5, 5.0)


def add_radii_from_red(edit) -> float:
    """Where an ADD lands, in red-mallet radii measured from the red mallet.

    The DSL now names a division point of a line rather than a distance, but
    the band below was measured in radii and stays stated that way -- it is a
    property of the contact solver at a distance, not of any one wording. This
    converts one to the other using the scene's own geometry.
    """
    a = BASELINE_PHYSICS[GEOMETRY[edit.endpoint_a].location_key]
    b = BASELINE_PHYSICS[GEOMETRY[edit.endpoint_b].location_key]
    span = math.dist(a, b)
    t = edit.fraction_from_a
    from_red = t if edit.endpoint_a == "red_mallet" else 1.0 - t
    return from_red * span / _MALLET_RADIUS


def check_add_case(edit) -> None:
    """Reject an ADD outside the band the scene was verified in.

    Called by the build script before a case is rendered, so a placement that
    would produce numerically degraded physics fails loudly at build time
    rather than shipping as ground truth.
    """
    if not isinstance(edit, dsl.AddEdit):
        return
    endpoints = (edit.endpoint_a, edit.endpoint_b)
    if "red_mallet" not in endpoints:
        raise ValueError(
            f"air_hockey_chain ADD edits are calibrated against the distance "
            f"from red_mallet, so red_mallet has to be one end of the line; "
            f"this one runs {edit.endpoint_a} -> {edit.endpoint_b}. See "
            "ADD_RADII_BAND."
        )
    radii = add_radii_from_red(edit)
    low, high = ADD_RADII_BAND
    if not (low <= radii <= high):
        raise ValueError(
            f"ADD lands {radii:.2f} radii from red_mallet, outside the "
            f"verified band [{low}, {high}]; the four-body chain loses energy "
            "in the contact solver outside it. The midpoint of the "
            "red-to-white line is 4.82 radii and sits inside it. See "
            "ADD_RADII_BAND."
        )


# Frames in this suite's videos: build_pcve_air_hockey_chain.py renders 4.0 s at 24 fps. An
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
