"""Shared geometry for the twin-ramp scene.

The PyBullet simulator and the Blender renderer have to agree on where the two
wedges sit to sub-millimetre precision, or the balls roll through the air on
one side and through the wood on the other. Rather than writing the same
trigonometry twice, both import it from here.

The apparatus is a real, buildable one: a flat base plank lying on the floor
with a solid wooden wedge screwed down at each end, sloped faces facing each
other. Its important property is that a wedge's sloped face runs all the way
down to the plank's top surface, so a ball leaving the slope meets the flat run
with no step at all. A ramp built the obvious way instead -- a board propped up
on a block, as `scripts/ramp_collision` does -- ends in a board-thickness cliff
at the bottom, and a ball doing 1.2 m/s launches off it. That launch is what
would stop the two balls meeting cleanly at the valley.

Coordinate frame
----------------
World z = 0 is the floor. The plank sits on it, so the surface the balls
actually run on is z = plank_thickness (``track_z``). World x = 0 is the middle
of the valley, i.e. the point the two balls are aimed at; world y = 0 is the
centre line of the track, which is the line both balls roll along.

Sides are named by their sign along x: side ``+1`` is the ramp at +x whose ball
rolls in the -x direction, side ``-1`` is its mirror image.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TrackGeometry:
    """Resolved dimensions of the two-ramp track, all in metres."""

    angle_rad: float
    run: float               # horizontal distance from a toe to that ramp's back
    rise: float              # height of the ramp's back above the track surface
    slope_length: float      # length of the sloped face itself
    valley_half: float       # half the flat gap between the two toes
    ramp_width: float
    ramp_body_thickness: float
    plank_thickness: float
    plank_length: float
    plank_width: float

    @property
    def track_z(self) -> float:
        """Height of the running surface: the top of the base plank."""
        return self.plank_thickness

    def toe(self, side: int) -> tuple[float, float]:
        """(x, z) of the bottom edge of a ramp's sloped face."""
        return (side * self.valley_half, self.track_z)

    def crest(self, side: int) -> tuple[float, float]:
        """(x, z) of the top edge of a ramp's sloped face."""
        return (side * (self.valley_half + self.run), self.track_z + self.rise)

    def slope_normal(self, side: int) -> tuple[float, float, float]:
        """Unit normal of a ramp's sloped face, pointing up out of the wood."""
        return (-side * math.sin(self.angle_rad), 0.0, math.cos(self.angle_rad))

    def ramp_half_extents(self) -> tuple[float, float, float]:
        """Half-extents of the *collision box* standing in for a wedge.

        The collider is a plain box, tilted so that its top face lies exactly on
        the wedge's sloped face; the rest of the box is buried under the plank
        where nothing can reach it. A box is used rather than a convex hull of
        the wedge's own six vertices because Bullet gives hulls a 40 mm collision
        margin by default -- more than half this ball's diameter -- and a ball
        rolling on a hull inflated by that much visibly floats.
        """
        return (
            self.slope_length / 2.0,
            self.ramp_width / 2.0,
            self.ramp_body_thickness / 2.0,
        )

    def ramp_pose(self, side: int) -> tuple[tuple[float, float, float], float]:
        """Centre and Y-axis rotation of a ramp's collision box.

        Returns ``((x, y, z), pitch)`` where ``pitch`` is the Euler-Y angle: a
        positive Y rotation tips local +x downwards, so the ramp at +x (which has
        to rise towards +x) takes -angle and its mirror takes +angle.
        """
        a, _, c = self.ramp_half_extents()
        cos_a = math.cos(self.angle_rad)
        sin_a = math.sin(self.angle_rad)
        # Derived by placing the box's lower top-face corner exactly on the toe.
        center_x = side * (self.valley_half + a * cos_a + c * sin_a)
        center_z = self.track_z + a * sin_a - c * cos_a
        return ((center_x, 0.0, center_z), -side * self.angle_rad)

    def wedge_profile(self, side: int) -> list[tuple[float, float]]:
        """The wedge's (x, z) cross-section, for building the render mesh.

        Right triangle: toe, back-bottom, back-top. Its hypotenuse is the
        sloped face, and coincides with the collision box's top face.
        """
        toe_x, toe_z = self.toe(side)
        crest_x, crest_z = self.crest(side)
        return [(toe_x, toe_z), (crest_x, toe_z), (crest_x, crest_z)]

    def release_center(
        self,
        side: int,
        radius: float,
        inset: float,
        clearance: float = 0.0,
    ) -> tuple[float, float, float]:
        """Centre of a ball resting on a ramp, ``inset`` down-slope from its crest.

        ``inset`` is measured along the sloped face from the crest to the ball's
        contact point, so it is the release height expressed in a way that stays
        meaningful if the ramp angle changes.
        """
        crest_x, crest_z = self.crest(side)
        cos_a = math.cos(self.angle_rad)
        sin_a = math.sin(self.angle_rad)
        # Down-slope direction, i.e. towards the valley.
        contact_x = crest_x - side * inset * cos_a
        contact_z = crest_z - inset * sin_a
        nx, _, nz = self.slope_normal(side)
        offset = radius + clearance
        return (contact_x + nx * offset, 0.0, contact_z + nz * offset)


def build_track(
    *,
    ramp_angle_deg: float,
    ramp_run: float,
    valley_half: float,
    ramp_width: float,
    ramp_body_thickness: float,
    plank_thickness: float,
    plank_length: float,
    plank_width: float,
) -> TrackGeometry:
    angle_rad = math.radians(float(ramp_angle_deg))
    run = float(ramp_run)
    return TrackGeometry(
        angle_rad=angle_rad,
        run=run,
        rise=run * math.tan(angle_rad),
        slope_length=run / math.cos(angle_rad),
        valley_half=float(valley_half),
        ramp_width=float(ramp_width),
        ramp_body_thickness=float(ramp_body_thickness),
        plank_thickness=float(plank_thickness),
        plank_length=float(plank_length),
        plank_width=float(plank_width),
    )
