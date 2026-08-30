"""Edit vocabulary for the bowling scene.

Moving objects: the ball and the ten pins. Since the sim only exposes one set
of pin parameters (pin_mass / pin_friction / pin_restitution) shared across
all pins, we expose a single collective 'pins' object rather than pretending
each pin is individually editable -- a SET on 'pins.mass' honestly scales all
ten together, which is how the physics actually works.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pcve_edit_dsl as dsl  # noqa: E402


# Must stay in sync with render_bowling.create_scenario().
BASELINE_PHYSICS = {
    'ball_radius': 0.11,
    'ball_mass': 3.0,
    'ball_friction': 0.4,
    'ball_restitution': 0.5,
    'pin_radius': 0.06,
    'pin_height': 0.38,
    'pin_mass': 0.8,
    'pin_friction': 0.4,
    'pin_restitution': 0.3,
    'floor_friction': 0.5,
    'floor_z': 0.0,
    'ball_initial_location': [10.0, 0.0, 0.18],
    'ball_initial_velocity': [-8.0, 0.0, 0.0],
    'pin_spacing': 0.28,
    'gravity': [0.0, 0.0, -9.81],
    'scene_offset_x': -5.0,
    'scene_offset_y': 26.3,
}


OBJECTS = {
    'ball': {'zh': '保龄球', 'en': 'the bowling ball'},
    'pins': {'zh': '保龄球瓶', 'en': 'the bowling pin set'},
}
# Deliberately NOT editable: the lane floor. It is the set the ball rolls on;
# any effective friction reachable through the lane is also reachable from the
# ball side.
# initial_velocity is also deliberately NOT bound: the baseline is -8 m/s in
# the -x direction, and the DSL treats initial_velocity as a scalar magnitude
# without sign preservation, so 'FROM 8 TO 12' would end up flipping the ball
# to +12 m/s (wrong way). Mass and friction give plenty of visual variation.


PROPERTIES = {
    'mass':        dsl.PropertySpec('scalar', 'kg', '质量',     'mass'),
    'friction':    dsl.PropertySpec('scalar', '',   '摩擦系数', 'friction coefficient'),
    'restitution': dsl.PropertySpec('scalar', '',   '恢复系数', 'restitution'),
}


# No rolling_friction on either the ball or the pins in this sim, so friction
# is a plain SimBinding onto the lateral coefficient (equivalent to a
# single-component compound per the skill's ball_rolling_friction=0 note).
SIM_BINDINGS = {
    ('ball', 'mass'):        dsl.SimBinding('ball_mass'),
    ('ball', 'friction'):    dsl.SimBinding('ball_friction'),
    ('ball', 'restitution'): dsl.SimBinding('ball_restitution'),
    ('pins', 'mass'):        dsl.SimBinding('pin_mass'),
    ('pins', 'friction'):    dsl.SimBinding('pin_friction'),
    ('pins', 'restitution'): dsl.SimBinding('pin_restitution'),
}


DELETE_BINDINGS: dict[str, dsl.DeleteBinding] = {}
# The sim has no per-pin active flag, so we cannot delete individual pins.


VOCAB = dsl.Vocabulary(
    objects=OBJECTS,
    properties=PROPERTIES,
    sim_bindings=SIM_BINDINGS,
    delete_bindings=DELETE_BINDINGS,
    baseline_physics=BASELINE_PHYSICS,
)
