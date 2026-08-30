"""Edit vocabulary for the car_gap_jump scene.

A 1:24 die-cast toy car is pushed off the top of a book stack on one table
and tries to clear a small gap to a second table of the same height. The
render script only forwards two physics knobs (launch_speed and car_friction),
so the vocab exposes just those -- both on the car itself, the only moving
object. Everything else (the tables, the book stack, the floor, the gap
geometry) is scenery.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pcve_edit_dsl as dsl  # noqa: E402


# Must stay in sync with simulate_car_gap_jump.py defaults and with the
# render command below.
BASELINE_PHYSICS = {
    'car_mass': 0.5,
    'car_restitution': 0.3,
    'car_friction': 0.45,
    'launch_speed': 1.9,
    'gap_width': 0.28,
    'deck_friction': 0.10,
    'far_deck_friction': 0.5,
    'gravity_z': -9.8,
}


OBJECTS = {
    'car': {'zh': '玩具车', 'en': 'the toy car'},
}
# Deliberately NOT editable: the two tables, the book stack, the floor and the
# gap between the tables. They are the set the car jumps across, not props.


PROPERTIES = {
    'friction':         dsl.PropertySpec('scalar', '',    '摩擦系数',   'friction coefficient'),
    'initial_velocity': dsl.PropertySpec('scalar', 'm/s', '初速度大小', 'initial speed'),
}


# Only the car's own knobs are bound. car_mass is a null edit (mass cancels in
# projectile motion and in Bullet's lateral-friction acceleration alike -- the
# sweep gave identical trajectories for 0.05, 0.5, 2.0, 5.0 kg). car_restitution
# barely moves the final rest position (0.02 -> 0.9 shifts final_x by ~7 cm).
# launch_speed and car_friction are the two knobs that actually flip the
# clear/fall outcome.
SIM_BINDINGS = {
    ('car', 'friction'):         dsl.SimBinding('car_friction'),
    ('car', 'initial_velocity'): dsl.SimBinding('launch_speed'),
}


DELETE_BINDINGS: dict[str, dsl.DeleteBinding] = {}
# The car is the only movable object; deleting it leaves an empty shot.


VOCAB = dsl.Vocabulary(
    objects=OBJECTS,
    properties=PROPERTIES,
    sim_bindings=SIM_BINDINGS,
    delete_bindings=DELETE_BINDINGS,
    baseline_physics=BASELINE_PHYSICS,
)
