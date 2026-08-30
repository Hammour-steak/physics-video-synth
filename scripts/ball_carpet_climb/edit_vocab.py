"""Edit vocabulary for the ball_carpet_climb scene.

The only moving object is the volleyball. Floor and rug are the set: per the
PCVE hard rules, background surfaces are not editable, and Bullet's pair-product
means any effective rolling resistance we would want to set on the rug is also
reachable by scaling the ball's friction knob.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pcve_edit_dsl as dsl  # noqa: E402


# Must stay in sync with render_ball_carpet_climb.create_scenario().
BASELINE_PHYSICS = {
    'launch_x': -0.57,
    'launch_y': -1.35,
    'launch_speed': 2.05,
    'carpet_thickness': 0.0,
    'carpet_rolling_friction': 0.060,
    'carpet_friction': 0.9,
    'carpet_restitution': 0.1,
    'floor_rolling_friction': 0.0015,
    'floor_friction': 0.45,
    'floor_restitution': 0.7,
    'ball_mass': 0.27,
    'ball_friction': 1.0,
    'ball_restitution': 0.45,
    'gravity_z': -9.8,
}


OBJECTS = {
    'ball': {'zh': '排球', 'en': 'the volleyball'},
}
# Deliberately NOT editable: the wood floor and the rug. They are the set the
# ball rolls across, not props in the scene. Bullet multiplies the pair, so any
# effective rolling resistance reachable through the rug ("deeper pile") is
# also reachable by scaling the ball's own friction -- see the friction knob
# note below.


PROPERTIES = {
    'mass':             dsl.PropertySpec('scalar', 'kg',  '质量',        'mass'),
    'friction':         dsl.PropertySpec('scalar', '',    '摩擦系数',    'friction coefficient'),
    'restitution':      dsl.PropertySpec('scalar', '',    '恢复系数',    'restitution'),
    'initial_velocity': dsl.PropertySpec('scalar', 'm/s', '初速度大小',  'initial speed'),
}


# Friction knob: the ball's baseline rolling_friction is 0 on purpose so that
# rolling resistance comes from the surface side (rf_eff = rf_surface *
# lat_ball). That means scaling the ball's lateral friction alone is exactly
# the total-friction knob for this scene; a CompoundBinding onto
# ball_rolling_friction would multiply by zero and do nothing. So we bind
# 'friction' directly to ball_friction (lateral), same shape as the puck in
# air_hockey_chain and per the skill's 'ball_rolling_friction=0' note.
SIM_BINDINGS = {
    ('ball', 'mass'):             dsl.SimBinding('ball_mass'),
    ('ball', 'friction'):         dsl.SimBinding('ball_friction'),
    ('ball', 'restitution'):      dsl.SimBinding('ball_restitution'),
    ('ball', 'initial_velocity'): dsl.SimBinding('launch_speed'),
}


DELETE_BINDINGS: dict[str, dsl.DeleteBinding] = {}
# Nothing to delete: the ball is the only moving object, and deleting it leaves
# an empty roll.


VOCAB = dsl.Vocabulary(
    objects=OBJECTS,
    properties=PROPERTIES,
    sim_bindings=SIM_BINDINGS,
    delete_bindings=DELETE_BINDINGS,
    baseline_physics=BASELINE_PHYSICS,
)
