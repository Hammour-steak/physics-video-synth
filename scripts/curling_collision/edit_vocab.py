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
    # Explicit start centres, derived from start_separation and forwarded to
    # the sim by the renderer. GEOMETRY below reads them to turn an ADD
    # edit's "N radii" into ice coordinates, so they have to stay in step
    # with start_separation: x = -/+ separation/2, z = stone_height/2.
    'stone_1_initial_location': [-2.5, 0.0, 0.057],
    'stone_2_initial_location': [2.5, 0.0, 0.057],
    # The blue stone. Absent at baseline; an ADD edit turns slot 2 of
    # stone_active on and overwrites this centre.
    'stone_3_mass': 20.0,
    'stone_3_initial_location': [0.0, 0.0, 0.057],
    'stone_active': [1, 1, 0],
    'gravity': [0.0, 0.0, -9.8],
}


OBJECTS = {
    'red_stone':    {'zh': '红色冰壶', 'en': 'the red stone'},
    'yellow_stone': {'zh': '黄色冰壶', 'en': 'the yellow stone'},
    'stones':       {'zh': '两只冰壶', 'en': 'the two stones'},
    # Not on the ice at baseline; only an ADD edit puts it there. It is
    # stationary when placed -- something set down for the two throws to run
    # into, not a third throw -- so it carries no initial_velocity binding.
    'blue_stone':   {'zh': '蓝色冰壶', 'en': 'the blue stone'},
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


# Where each stone's centre and radius live, so an ADD stated in radii can be
# resolved into ice coordinates. All three are the same 0.145 m stone and
# share one radius parameter, so "N radii" means the same distance whichever
# stone it is measured from here.
GEOMETRY = {
    'red_stone':    dsl.GeometryAnchor('stone_1_initial_location', 'stone_radius'),
    'yellow_stone': dsl.GeometryAnchor('stone_2_initial_location', 'stone_radius'),
    'blue_stone':   dsl.GeometryAnchor('stone_3_initial_location', 'stone_radius'),
}


# ADD turns slot 2 of stone_active on and writes the resolved centre. The
# stone's mass already sits in BASELINE_PHYSICS under stone_3_mass and is
# always passed to the sim, where it has no effect while the slot is off.
ADD_BINDINGS = {
    'blue_stone': dsl.AddBinding(
        presence_key='stone_active',
        presence_index=2,
        location_key='stone_3_initial_location',
    ),
}


# Frames in this suite's videos: build_pcve_curling_collision.py renders 4.0 s at 24 fps. An
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
