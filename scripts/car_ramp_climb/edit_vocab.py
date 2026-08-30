"""Edit vocabulary for the car_ramp_climb scene.

A toy car is given an initial up-slope push and either climbs the ramp,
launches off the top, or stalls and slides back. Two knobs actually move the
outcome: the push (launch_speed) and the ramp/car pair friction. Both are
bound on the car, which is the only moving object -- the ramp is scenery.

Note on the friction binding: the sim sets the car's own lateral friction
equal to ramp_friction (see simulate_car_ramp_climb.py's changeDynamics
on the car body), so this single sim key is really the pair friction between
the car and the ramp. Binding it as car.friction is the honest description
of what the edit does -- "make the wheels grippier" -- and it keeps the edit
on the moving side of the impact, matching the rule that static set pieces
stay untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pcve_edit_dsl as dsl  # noqa: E402


# Must stay in sync with render_car_ramp_climb.create_scenario().
BASELINE_PHYSICS = {
    'car_mass': 0.35,
    'car_restitution': 0.05,
    'ramp_angle_deg': 20.0,
    'ramp_length': 0.9,
    'ramp_width': 0.35,
    'ramp_thickness': 0.03,
    'ramp_friction': 0.25,
    'floor_friction': 0.9,
    'launch_speed': 2.7,
    'gravity': [0.0, 0.0, -9.8],
}


OBJECTS = {
    'car': {'zh': '玩具车', 'en': 'the toy car'},
}
# Deliberately NOT editable: the ramp itself (its angle, length, surface) and
# the floor. They are the set the car interacts with.


PROPERTIES = {
    'friction':         dsl.PropertySpec('scalar', '',    '摩擦系数',   'friction coefficient'),
    'initial_velocity': dsl.PropertySpec('scalar', 'm/s', '初速度大小', 'initial speed'),
}


# car_mass and car_restitution are deliberately NOT bound: the sweep found
# both to be null edits on this scene (mass cancels out at fixed launch
# velocity given Bullet's friction formulation; restitution barely touches
# the trajectory because the airtime hop is small and the landing normal is
# almost aligned with gravity).
SIM_BINDINGS = {
    ('car', 'friction'):         dsl.SimBinding('ramp_friction'),
    ('car', 'initial_velocity'): dsl.SimBinding('launch_speed'),
}


DELETE_BINDINGS: dict[str, dsl.DeleteBinding] = {}


VOCAB = dsl.Vocabulary(
    objects=OBJECTS,
    properties=PROPERTIES,
    sim_bindings=SIM_BINDINGS,
    delete_bindings=DELETE_BINDINGS,
    baseline_physics=BASELINE_PHYSICS,
)
