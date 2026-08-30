"""Edit vocabulary for the bouncing_ball scene.

Single rigid ball in an empty room. Floor is background (set), so the ball is
the only entry in OBJECTS.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pcve_edit_dsl as dsl  # noqa: E402


# Must stay in sync with render_bouncing_ball.create_scenario().
BASELINE_PHYSICS = {
    'ball_radius': 0.05,
    'ball_mass': 1.0,
    'ball_friction': 0.38,
    'ball_restitution': 0.78,
    'ball_initial_location': [0.0, 0.0, 1.37],
    'ball_initial_velocity': [0.5, 0.0, -0.18],
    'floor_friction': 0.82,
    'gravity': [0.0, 0.0, -9.81],
}


OBJECTS = {
    'ball': {'zh': '红球', 'en': 'the red ball'},
}
# Deliberately NOT editable: the floor and the walls. They are the room the
# ball bounces in, not a prop.


PROPERTIES = {
    'mass':             dsl.PropertySpec('scalar', 'kg',  '质量',        'mass'),
    'friction':         dsl.PropertySpec('scalar', '',    '摩擦系数',    'friction coefficient'),
    'restitution':      dsl.PropertySpec('scalar', '',    '恢复系数',    'restitution'),
    'initial_velocity': dsl.PropertySpec('scalar', 'm/s', '初速度大小',  'initial speed'),
}


# The ball has no rolling_friction knob in this scene; plain SimBinding onto
# lateral is the whole friction knob. initial_velocity is scalar magnitude
# bound to the x component of the baseline [0.5, 0, -0.18] (same shape as
# ball_block's binding).
SIM_BINDINGS = {
    ('ball', 'mass'):             dsl.SimBinding('ball_mass'),
    ('ball', 'friction'):         dsl.SimBinding('ball_friction'),
    ('ball', 'restitution'):      dsl.SimBinding('ball_restitution'),
    ('ball', 'initial_velocity'): dsl.SimBinding('ball_initial_velocity', index=0),
}


DELETE_BINDINGS: dict[str, dsl.DeleteBinding] = {}


VOCAB = dsl.Vocabulary(
    objects=OBJECTS,
    properties=PROPERTIES,
    sim_bindings=SIM_BINDINGS,
    delete_bindings=DELETE_BINDINGS,
    baseline_physics=BASELINE_PHYSICS,
)
