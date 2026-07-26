from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
MODULE_PATH = SCRIPTS_DIR / "simulate_domino_chain.py"
SPEC = importlib.util.spec_from_file_location("simulate_domino_chain", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
SIMULATION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SIMULATION
SPEC.loader.exec_module(SIMULATION)


def test_tilted_domino_starts_on_the_floor() -> None:
    dimensions = (0.18, 0.58, 1.18)
    height = SIMULATION.supported_center_height(
        dimensions=dimensions,
        pitch_rad=math.radians(12.0),
    )

    expected = 0.5 * (
        dimensions[2] * math.cos(math.radians(12.0))
        + dimensions[0] * math.sin(math.radians(12.0))
    )
    assert math.isclose(height, expected, abs_tol=1e-9)


def test_gravity_creates_an_ordered_three_domino_chain() -> None:
    result = SIMULATION.simulate(
        SIMULATION.DominoChainConfig(duration_sec=3.0)
    )

    first_contact = result["contact_events"]["domino_1_to_domino_2"]
    second_contact = result["contact_events"]["domino_2_to_domino_3"]
    assert first_contact is not None
    assert second_contact is not None
    assert first_contact["time_sec"] < second_contact["time_sec"]
    assert result["initial_state"]["linear_velocity_is_zero"] is True
    assert result["summary"]["ordered_chain_complete"] is True
    assert all(angle > 65.0 for angle in result["summary"]["max_tilt_deg"])
