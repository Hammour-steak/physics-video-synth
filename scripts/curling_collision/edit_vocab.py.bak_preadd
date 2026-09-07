"""Edit vocabulary for the curling_collision scene.

Two curling stones -- one red, one yellow -- are launched at each other along
the ice from 5 m apart, meet in the middle and collide. Both stones move.

For initial velocity, three shapes of edit are supported:
  - per-stone: red_stone.initial_velocity, yellow_stone.initial_velocity
    (each stone's own launch magnitude; direction is fixed by scene geometry)
  - collective: stones.initial_velocity (compound edit that scales *both*
    stones' launch magnitudes by the same ratio, e.g. "gentle symmetric
    launch").
Mass is genuinely per-body; restitution is Bullet-pair-shared so only exposed
on the collective stones. The ice surface is deliberately not editable.

This vocab requires simulate_curling_collision.py to accept
--stone-1-launch-speed / --stone-2-launch-speed (both fall back to
--launch-speed when unset), and render_curling_collision.py to forward those
per-stone flags from the physics dict.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pcve_edit_dsl as dsl  # noqa: E402


# Must stay in sync with render_curling_collision.create_scenario() and the
# per-stone launch args in simulate_curling_collision.py.
BASELINE_PHYSICS = {
    'stone_radius': 0.145,
    'stone_height': 0.114,
    'stone_mass': 20.0,
    'stone_2_mass': 20.0,
    'stone_friction': 0.15,
    'stone_restitution': 0.0,
    'ice_friction': 0.015,
    'launch_speed': 1.5,
    'stone_1_launch_speed': 1.5,
    'stone_2_launch_speed': 1.5,
    'start_separation': 5.0,
    'gravity': [0.0, 0.0, -9.8],
}


OBJECTS = {
    'red_stone':    {'zh': '红色冰壶', 'en': 'the red stone'},
    'yellow_stone': {'zh': '黄色冰壶', 'en': 'the yellow stone'},
    'stones':       {'zh': '两只冰壶', 'en': 'the two stones'},
}


PROPERTIES = {
    'mass':             dsl.PropertySpec('scalar', 'kg',  '质量',       'mass'),
    'restitution':      dsl.PropertySpec('scalar', '',    '恢复系数',   'restitution'),
    'initial_velocity': dsl.PropertySpec('scalar', 'm/s', '初速度大小', 'initial speed'),
}


# Per-stone mass and per-stone launch magnitude are real per-body knobs.
# The collective stones.initial_velocity is a CompoundBinding: it scales
# BOTH stones' launch magnitudes by the same ratio, so a symmetric edit like
# "both stones half as fast" is one DSL line rather than two.
# stones.restitution is the pair-shared Bullet coefficient.
#
# stone_friction is deliberately NOT bound: sweep values 0.02 -> 0.60 give
# millimetre-scale differences in final rest positions on this clip length
# (Bullet's rolling friction dominates on the ice and the pair contact only
# lasts a few substeps).
SIM_BINDINGS = {
    ('red_stone',    'mass'):             dsl.SimBinding('stone_mass'),
    ('yellow_stone', 'mass'):             dsl.SimBinding('stone_2_mass'),
    ('red_stone',    'initial_velocity'): dsl.SimBinding('stone_1_launch_speed'),
    ('yellow_stone', 'initial_velocity'): dsl.SimBinding('stone_2_launch_speed'),
    ('stones',       'restitution'):      dsl.SimBinding('stone_restitution'),
    ('stones',       'initial_velocity'): dsl.CompoundBinding(components=(
        dsl.SimBinding('stone_1_launch_speed'),
        dsl.SimBinding('stone_2_launch_speed'),
    )),
}


DELETE_BINDINGS: dict[str, dsl.DeleteBinding] = {}


VOCAB = dsl.Vocabulary(
    objects=OBJECTS,
    properties=PROPERTIES,
    sim_bindings=SIM_BINDINGS,
    delete_bindings=DELETE_BINDINGS,
    baseline_physics=BASELINE_PHYSICS,
)
