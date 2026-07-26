from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
MODULE_PATH = SCRIPTS_DIR / "simulate_ball_block_impact.py"
sys.path.insert(0, str(SCRIPTS_DIR))
SPEC = importlib.util.spec_from_file_location("simulate_ball_block_impact", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
SIMULATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SIMULATION)


def contact_point_velocity(
    linear_velocity: tuple[float, float, float],
    angular_velocity: tuple[float, float, float],
    radius: float,
) -> tuple[float, float, float]:
    vx, vy, vz = linear_velocity
    wx, wy, _ = angular_velocity
    return vx - radius * wy, vy + radius * wx, vz


def test_floor_supported_ball_starts_without_slip() -> None:
    radius = 0.34
    linear_velocity = (0.6, 4.1, 0.0)

    angular_velocity, mode = SIMULATION.initial_ball_spin(
        linear_velocity=linear_velocity,
        location=(-0.85, -1.32, radius + 0.001),
        radius=radius,
        explicit_angular_velocity=None,
    )

    assert mode == "floor_no_slip"
    assert all(
        math.isclose(value, 0.0, abs_tol=1e-12)
        for value in contact_point_velocity(linear_velocity, angular_velocity, radius)
    )


def test_airborne_ball_defaults_to_zero_spin() -> None:
    angular_velocity, mode = SIMULATION.initial_ball_spin(
        linear_velocity=(0.28, 0.09, -0.18),
        location=(0.02, -0.11, 2.20),
        radius=0.34,
        explicit_angular_velocity=None,
    )

    assert mode == "airborne_zero_spin"
    assert angular_velocity == (0.0, 0.0, 0.0)


def test_explicit_spin_is_preserved() -> None:
    angular_velocity, mode = SIMULATION.initial_ball_spin(
        linear_velocity=(1.0, 0.0, 0.0),
        location=(0.0, 0.0, 1.0),
        radius=0.34,
        explicit_angular_velocity=(1.0, 2.0, 3.0),
    )

    assert mode == "explicit"
    assert angular_velocity == (1.0, 2.0, 3.0)
