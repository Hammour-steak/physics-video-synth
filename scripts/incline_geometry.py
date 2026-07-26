from __future__ import annotations

import math


def grounded_wedge_vertices(
    *,
    location: tuple[float, float, float],
    dimensions: tuple[float, float, float],
    pitch_deg: float,
    floor_z: float = 0.0,
) -> tuple[tuple[float, float, float], ...]:
    """Return a floor-connected wedge whose top matches the tilted ramp box."""
    half_length = 0.5 * float(dimensions[0])
    half_width = 0.5 * float(dimensions[1])
    half_thickness = 0.5 * float(dimensions[2])
    pitch = math.radians(float(pitch_deg))
    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)

    high_x = location[0] - cos_pitch * half_length + sin_pitch * half_thickness
    low_x = location[0] + cos_pitch * half_length + sin_pitch * half_thickness
    high_z = location[2] + sin_pitch * half_length + cos_pitch * half_thickness
    low_z = location[2] - sin_pitch * half_length + cos_pitch * half_thickness
    low_y = location[1] - half_width
    high_y = location[1] + half_width
    return (
        (high_x, low_y, floor_z),
        (low_x, low_y, floor_z),
        (low_x, high_y, floor_z),
        (high_x, high_y, floor_z),
        (high_x, low_y, high_z),
        (low_x, low_y, low_z),
        (low_x, high_y, low_z),
        (high_x, high_y, high_z),
    )


GROUNDED_WEDGE_QUADS = (
    (0, 3, 2, 1),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
    (4, 5, 6, 7),
)


def grounded_wedge_triangles() -> tuple[int, ...]:
    indices: list[int] = []
    for a, b, c, d in GROUNDED_WEDGE_QUADS:
        indices.extend((a, b, c, a, c, d))
    return tuple(indices)
