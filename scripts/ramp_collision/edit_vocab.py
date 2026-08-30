"""Edit vocabulary for the ramp_collision scene.

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


# Must stay in sync with render_ramp_collision.create_scenario().
BASELINE_PHYSICS = {
    "ramp_angle_deg": 12.0,
    "ball_mass": 0.05,
    "ball_friction": 0.45,
    "ball_restitution": 0.6,
    "ball_rolling_friction": 0.002,
    "marble_mass": 0.05,
    "marble_masses": [0.05, 0.05],
    "marble_friction": 0.15,
    "marble_restitution": 0.3,
    "ramp_friction": 0.7,
    "floor_friction": 0.4,
    "marble_active": [1, 1],
    "marble_initial_velocities": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
}


OBJECTS = {
    "red_ball":      {"zh": "红球", "en": "the red ball"},
    "blue_marble":   {"zh": "蓝球", "en": "the blue marble"},
    "yellow_marble": {"zh": "黄球", "en": "the yellow marble"},
}
# Deliberately NOT editable: the tabletop and the ramp. An edit names a moving
# object, not the set it moves through. Bullet multiplies the pair, so any
# effective coefficient reachable through the tabletop/ramp is also reachable
# through the ball or a marble.


PROPERTIES = {
    "mass":              dsl.PropertySpec("scalar", "kg",  "质量",       "mass"),
    # One friction knob per object, not split into lateral and rolling. When
    # SET from x to y, both scale by y/x; the value shown in prompts is the
    # object's lateral coefficient. See the "friction knob" note below.
    "friction":          dsl.PropertySpec("scalar", "",    "摩擦系数",   "friction coefficient"),
    "restitution":       dsl.PropertySpec("scalar", "",    "恢复系数",   "restitution"),
    # Scalar speed: magnitude only. Direction stays whatever the baseline
    # velocity had. Requires a non-zero baseline direction to be meaningful.
    "initial_velocity":  dsl.PropertySpec("scalar", "m/s", "初速度大小", "initial speed"),
}


SIM_BINDINGS = {
    # red_ball -> ball_* keys. The compound friction binding is why raising
    # the ball's friction actually slows it -- see the "friction knob" note.
    ("red_ball", "mass"):        dsl.SimBinding("ball_mass"),
    ("red_ball", "friction"):    dsl.CompoundBinding(components=(
        dsl.SimBinding("ball_friction"),           # display; the lateral coefficient
        dsl.SimBinding("ball_rolling_friction"),
    )),
    ("red_ball", "restitution"): dsl.SimBinding("ball_restitution"),
    # per-marble mass -> marble_masses[i]
    ("blue_marble",   "mass"): dsl.SimBinding("marble_masses", index=0),
    ("yellow_marble", "mass"): dsl.SimBinding("marble_masses", index=1),
}


# The "friction knob" note. Bullet combines rolling resistance as
# `rf_a * lat_b + rf_b * lat_a`. If we bound only `lateral_friction` on the
# ball, raising it would multiply in the marbles' rolling friction (zero) --
# so it would not slow the ball at all until it hit them. Scaling `lateral`
# and `rolling` together fixes this: raising `red_ball.friction` scales
# `ball_rolling_friction`, which multiplies against the ramp's and floor's
# lateral friction to slow the roll for the whole descent and run-out.


DELETE_BINDINGS = {
    "blue_marble":   dsl.DeleteBinding("marble_active", index=0),
    "yellow_marble": dsl.DeleteBinding("marble_active", index=1),
}


VOCAB = dsl.Vocabulary(
    objects=OBJECTS,
    properties=PROPERTIES,
    sim_bindings=SIM_BINDINGS,
    delete_bindings=DELETE_BINDINGS,
    baseline_physics=BASELINE_PHYSICS,
)
