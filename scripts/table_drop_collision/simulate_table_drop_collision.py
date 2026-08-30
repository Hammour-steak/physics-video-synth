from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pybullet as p


# A tennis ball is rolled west across a round coffee table, runs off the edge,
# falls the height of the table and lands on a second tennis ball waiting on the
# floor. The whole thing crosses the frame rather than coming at the lens, which
# is what makes a 67 mm ball legible from far enough back to hold both the table
# and the floor in one shot.
#
# The scene is built around two things that are decided *before* they happen and
# can be written down in one line each.
#
# 1. **Where it lands is settled at the lip.** Once the ball leaves the edge
#    nothing touches it, so its whole flight is fixed by the two numbers it had
#    when it left: the fall takes
#
#        t = sqrt(2 * (z_lip - r) / g)
#
#    which for this table is 0.313 s and does not depend on the ball, its mass,
#    its size or how fast it was going -- and the horizontal reach is just
#    ``v_lip * t``. Roll it off twice as fast and it lands twice as far out, in
#    exactly the same amount of time. The simulation records the state at the lip
#    and checks the parabola against it on every run; ``--disable-ball-b``
#    removes the target so the predicted touchdown can be compared with a real
#    one rather than with another prediction.
#
# 2. **Where the struck ball goes is settled by the line of centres.** Two
#    spheres exchange momentum only along the line joining their centres, so with
#    ``n`` that unit vector and the target at rest,
#
#        v_B' = (1 + e) * m_A / (m_A + m_B) * (v_A . n) * n
#        v_A' = v_A - (1 + e) * m_B / (m_A + m_B) * (v_A . n) * n
#
#    The balls are two of the same tennis ball, so the mass ratio is 1 and both
#    coefficients collapse to (1 + e)/2 = 0.87. That is what makes the outcome
#    worth filming. The target sits dead in line with the lane, so the whole
#    exchange stays in one vertical plane: the normal has no y component, the
#    struck ball leaves due west on 180.0 deg and the ball that hit it is thrown
#    due east on 360.0 deg, exactly 180 deg apart. Both finish on y = -0.290, the
#    line they started on. There is nothing sideways in the picture to explain.
#
#    The target is not a free body, though: it is sitting on the floor, and the
#    floor takes a quarter of the blow straight back out. See predict_impact.
#
#    **This is still not a central collision, and it cannot be made into one.**
#    A collision is central when the approach velocity lies along the line of
#    centres. Here the ball arrives 68 deg below horizontal -- it is falling, not
#    rolling -- while the line of centres is only 25 deg below horizontal, so the
#    two are 43 deg apart and the impact parameter is 0.68 of a full width. That
#    is a glancing blow by any measure, and squaring it up means putting the
#    target where the line of centres points along the fall, i.e. almost directly
#    under the ball. It works, and it is useless: the impact parameter drops to
#    0.02 and **the target moves 14 mm**, because the momentum then goes into the
#    floor rather than across it. A shallower arrival needs a much harder push,
#    which walks the whole exchange 0.5 m further west, out of frame and, at
#    3.0 m/s off the lip, into the glazed wall. The obliquity is the price of
#    dropping a ball onto another ball, and it is what the scene is about.
#
# There is a third fact that is invisible and changes the numbers anyway: **a
# tennis ball is a hollow shell, not a solid sphere.** Its moment of inertia is
# 2/3 m r^2, not 2/5, which is 67 per cent more resistance to being spun up. It
# is set explicitly with ``localInertiaDiagonal`` because PyBullet derives the
# inertia of a GEOM_SPHERE as a solid, and it is what makes the struck ball skid
# noticeably before it starts rolling.
#
# Coordinates are metric and match render_table_drop_collision.py's world frame:
# z = 0 is the living-room floor, the origin is under the centre of the round
# coffee table, +x runs east and +y north, toward the sofa.


# The coffee table, measured off assets/models/living_room.glb by raycasting; see
# the render script's calibration block. The top is a circle and it is dead flat:
# 335 probe points spread over it all came back at the same z to five decimals.
TABLE_TOP_Z = 0.48245        # top surface above the floor
TABLE_RADIUS = 0.44840       # radius of the flat top
TABLE_RIM_RADIUS = 0.45031   # widest point, 1.9 mm further out under a chamfer
TABLE_PLATE = 0.075          # thickness given to the collision cylinder

# The sofa's front face. Nothing in the hero take goes near it -- everything
# travels south, away from it -- but a ball that ends up behind this line has
# gone somewhere the shot does not intend, and that is reported rather than
# silently rendered as a ball inside a sofa.
SOFA_FRONT_Y = 0.806

# The floor plane's own extent. Past this the model simply stops and a ball
# would be rolling on nothing.
FLOOR_X = (-2.487, 1.111)
FLOOR_Y = (-2.467, 2.029)

SETTLE_SPEED = 0.02          # m/s below which a ball counts as stopped
SETTLE_SPIN = 0.30           # rad/s below which it counts as stopped *turning*
RESIDUAL_SPIN_LIMIT = 0.30

# ITF Type 2: 65.4-68.6 mm and 56.0-59.4 g. The middle of both.
TENNIS_DIAMETER = 0.067
TENNIS_MASS = 0.057
# A tennis ball is a pressurised rubber shell with a felt cover and nothing in
# the middle, so its inertia is that of a thin spherical shell rather than a
# solid ball. 2/3 against 2/5 is not a detail: it is 67 per cent more resistance
# to being spun up, and it is why the struck ball skids before it rolls.
SHELL_INERTIA_FACTOR = 2.0 / 3.0


def shell_inertia(mass: float, radius: float) -> float:
    return SHELL_INERTIA_FACTOR * mass * radius ** 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--props-collider", type=Path, default=None,
        help="Metre-scale OBJ of what is already standing on the table -- the "
        "cup, the open magazine and the reading glasses -- written by "
        "render_table_drop_collision.py's --mode colliders pass. It is a "
        "separate body from the table so that touching it is reported as its own "
        "event: the hero take rolls 44 mm clear of the magazine's edge, and a "
        "ball that reaches it has gone somewhere the shot does not intend.",
    )
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-sec", type=float, default=2.8)
    parser.add_argument(
        "--substeps", type=int, default=60,
        help="Physics steps per rendered frame. Both events this scene measures "
        "-- leaving the lip and the impact -- happen inside a single frame at 24 "
        "fps, and the state either side of them is sampled at substep resolution.",
    )
    parser.add_argument("--gravity-z", type=float, default=-9.8)

    # --- the ball that is rolled off the table ---
    parser.add_argument("--ball-a-radius", type=float, default=TENNIS_DIAMETER / 2.0)
    parser.add_argument("--ball-a-mass", type=float, default=TENNIS_MASS)
    parser.add_argument(
        "--ball-a-restitution", type=float, default=0.86,
        help="Bullet multiplies the two bodies' restitutions, so ball on ball "
        "sees this squared: 0.86 each gives 0.74, which is a tennis ball.",
    )
    parser.add_argument("--ball-a-friction", type=float, default=0.62)

    # --- the ball waiting on the floor ---
    parser.add_argument("--ball-b-radius", type=float, default=TENNIS_DIAMETER / 2.0)
    parser.add_argument("--ball-b-mass", type=float, default=TENNIS_MASS)
    parser.add_argument("--ball-b-restitution", type=float, default=0.86)
    parser.add_argument("--ball-b-friction", type=float, default=0.62)
    parser.add_argument(
        "--ball-b-x", type=float, default=-0.722,
        help="How far along the fall the target waits, and the most sensitive "
        "number in the scene. The falling ball's own touchdown is at x = -0.672; "
        "this sits 50 mm *past* it, so the ball meets the target on the way down "
        "and catches it high on its east face rather than coming to rest first. "
        "That placement is the difference between a shot and a thud. Put the "
        "target on the touchdown itself and the ball lands squarely on its crown, "
        "the line of centres comes out 60 deg below horizontal, the blow is spent "
        "driving it into the floor and it leaves at 0.15 m/s; 50 mm out the normal "
        "is 21 deg and it leaves at 1.25. Another 10 mm and the ball misses "
        "altogether -- which is this scene's own thesis biting, because the "
        "landing point is settled at the lip and the target is only 67 mm wide.",
    )
    parser.add_argument(
        "--ball-b-y", type=float, default=-0.290,
        help="How far off the rolling ball's line the target sits. Dead in line "
        "with the lane, so the whole exchange stays in one vertical plane: the "
        "contact normal comes out with a y component of -0.0000, the two balls "
        "leave 180.0 deg apart, and both finish the shot on y = -0.290, the line "
        "they started on. There is nothing sideways in the picture to explain, "
        "which is the point. Note what this does NOT buy: the collision is still "
        "43 deg off the line of centres, because the ball arrives at 68 deg below "
        "horizontal and the line of centres is only 25 deg below it. That part is "
        "not tunable -- see the module comment. This lane cost 22 mm of offset to "
        "adopt, and the offset was not decoration: it threw the struck ball clear "
        "of two failures that dead-in-line walks straight into, both of which "
        "floor_rolling_friction now pays for instead.",
    )
    parser.add_argument(
        "--disable-ball-b", action="store_true",
        help="Leave the target out, so the falling ball's own touchdown can be "
        "measured. This is the run that checks the projectile prediction against "
        "something other than itself, and it is how the aim was set.",
    )
    # PCVE DELETE edit for the target: 0 removes ball_b, equivalent to
    # --disable-ball-b (both are honoured). ball_a is required -- the
    # whole simulation is built around its roll off the table.
    parser.add_argument("--ball-b-active", type=int, default=1)
    parser.add_argument(
        "--ball-rolling-friction", type=float, default=0.0,
        help="Kept at 0 so all rolling resistance comes from the surfaces. "
        "Bullet does not combine rolling friction as a plain product: it uses "
        "rf_a * lateral_b + rf_b * lateral_a, so a ball given rf = 1 to 'let the "
        "surface through' multiplies in the *surface's lateral* friction and "
        "turns the roll into a skid. With the balls at 0 the effective value is "
        "the surface's rolling friction times the ball's lateral friction, which "
        "is why the deceleration this actually produces is measured and reported "
        "rather than asserted.",
    )
    parser.add_argument("--ball-spinning-friction", type=float, default=0.0)

    # --- surfaces ---
    parser.add_argument(
        "--table-rolling-friction", type=float, default=0.0037,
        help="Rolling resistance of the lacquered table top. Felt on wood is "
        "lossy -- far more so than the glass marbles this room's other scene "
        "rolls -- and at this value the ball loses 0.40 m/s^2, so a 1.28 m/s "
        "push arrives at the lip at 1.05. That is a rolling resistance "
        "coefficient of 0.041, which is a tennis ball on a hard smooth surface. "
        "It is also this shot's clock: it sets the speed at the lip, and the "
        "speed at the lip sets everything after it.",
    )
    parser.add_argument("--table-friction", type=float, default=0.55)
    parser.add_argument("--table-restitution", type=float, default=0.60)
    parser.add_argument(
        "--table-spinning-friction", type=float, default=0.004,
        help="Resistance to a ball turning on the spot about the vertical. "
        "Nothing else in Bullet opposes it -- rolling friction resists rolling, "
        "lateral friction resists sliding -- so left at zero a ball comes to a "
        "dead stop and goes on turning where it stands, which is plainly visible "
        "in the render.",
    )
    parser.add_argument(
        "--floor-rolling-friction", type=float, default=0.0110,
        help="Rolling resistance of the bare wooden floor. Set for the shot "
        "rather than for the physics, and it is the only stretch in the scene, so "
        "it is worth saying exactly what it buys. With the target dead in line "
        "the struck ball runs due west and the ball that hit it is thrown due "
        "east, and at the honest 0.0060 both of them leave the picture: the "
        "hitter comes to rest at radius 0.344, under the table's overhang and "
        "behind the top, and the struck one slides to x = -1.204, which projects "
        "to screen x = -0.016 -- twenty pixels past the left edge of frame. At "
        "0.0110 the hitter stops at radius 0.511, a clear 62 mm outside the "
        "table's rim, and the struck one at -1.026, comfortably inside. The old "
        "22 mm lateral offset used to buy both of those by throwing the hitter "
        "sideways; this pays for them without bending the exchange. Beware the "
        "ceiling: above roughly 0.02 the balls stop rolling and start sliding, "
        "lateral friction takes over, and 0.065, 0.095 and 0.130 all give "
        "bit-for-bit identical resting positions.",
    )
    parser.add_argument("--floor-friction", type=float, default=0.58)
    parser.add_argument(
        "--floor-restitution", type=float, default=0.87,
        help="Against the balls' own 0.86 this is an effective 0.75, which is "
        "the ITF rebound spec: dropped 2.54 m onto a rigid surface a tennis ball "
        "comes back up between 1.35 and 1.47 m.",
    )
    parser.add_argument("--floor-spinning-friction", type=float, default=0.004)
    parser.add_argument("--props-friction", type=float, default=0.55)
    parser.add_argument("--props-restitution", type=float, default=0.25)

    # --- the push ---
    parser.add_argument(
        "--launch-x", type=float, default=0.315,
        help="Where the push starts, 27 mm inside the table's east rim. The run "
        "west to the lip is 0.657 m, which is as much of this table as there is "
        "on a line that clears the magazine.",
    )
    parser.add_argument(
        "--launch-y", type=float, default=-0.290,
        help="The lane the ball is rolled down: a chord across the near, southern "
        "third of the round top. The open magazine lies across the middle of the "
        "table and its south edge is at y = -0.174, so a lane here passes it with "
        "82 mm to spare -- and it keeps the whole roll on the camera's side of "
        "the table, where the ball is a ball rather than a dot behind a cup.",
    )
    parser.add_argument(
        "--launch-speed", type=float, default=1.28,
        help="A firm but unhurried hand roll. After 0.657 m of table the ball "
        "reaches the lip at 1.053 m/s, which throws it 0.331 m clear of the edge "
        "-- far enough out that the fall and the impact happen in the open and "
        "not tucked under the table's overhang.",
    )
    parser.add_argument(
        "--launch-heading-deg", type=float, default=180.0,
        help="Due west. The ball leaves by the table's west lip because that is "
        "the side with somewhere to go: the floor runs 2.0 m further west before "
        "the glazed wall, while to the east it stops 0.21 m past the table's rim. "
        "It also puts the whole shot across the frame rather than into the lens, "
        "which is the only way a 67 mm ball reads at this distance.",
    )
    return parser.parse_args()


def add_box(client: int, half_extents, centre, friction: float, restitution: float,
            rolling_friction: float = 0.0, spinning_friction: float = 0.0) -> int:
    shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=list(half_extents),
                                   physicsClientId=client)
    body = p.createMultiBody(0.0, shape, -1, list(centre), physicsClientId=client)
    p.changeDynamics(
        body, -1,
        lateralFriction=friction, rollingFriction=rolling_friction,
        spinningFriction=spinning_friction, restitution=restitution,
        collisionMargin=0.0005, physicsClientId=client,
    )
    return body


def add_table(client: int, args: argparse.Namespace) -> int:
    """The round top, as a cylinder with its top face at TABLE_TOP_Z.

    A primitive rather than the model's own triangles, and here that is the
    honest choice rather than the cheap one: the top *is* a circle and it *is*
    flat. Sixteen radial bisections put the rim at 0.4484 m within 0.15 per cent,
    and 335 probes spread over the surface all came back at one z to five
    decimals. What a trimesh would add is the pedestal underneath, which nothing
    in this scene can reach.

    Bullet's default 0.04 m collision margin would push the rim out by 40 mm --
    more than half a tennis ball -- on the one edge the whole scene depends on
    the ball leaving cleanly, so it is cut to 0.5 mm.
    """
    shape = p.createCollisionShape(
        p.GEOM_CYLINDER, radius=TABLE_RADIUS, height=TABLE_PLATE,
        physicsClientId=client)
    body = p.createMultiBody(0.0, shape, -1, [0.0, 0.0, TABLE_TOP_Z - TABLE_PLATE / 2.0],
                             physicsClientId=client)
    p.changeDynamics(
        body, -1,
        lateralFriction=float(args.table_friction),
        rollingFriction=float(args.table_rolling_friction),
        spinningFriction=float(args.table_spinning_friction),
        restitution=float(args.table_restitution),
        collisionMargin=0.0005, physicsClientId=client,
    )
    return body


def add_floor(client: int, args: argparse.Namespace) -> int:
    return add_box(client, (4.0, 4.0, 0.25), (0.0, 0.0, -0.25),
                   float(args.floor_friction), float(args.floor_restitution),
                   float(args.floor_rolling_friction),
                   float(args.floor_spinning_friction))


def add_props(client: int, obj_path: Path, friction: float, restitution: float) -> int:
    if not obj_path.exists():
        raise FileNotFoundError(
            f"Table dressing collision mesh not found: {obj_path}\n"
            "Generate it first with:\n"
            "  blender -b --python scripts/table_drop_collision/"
            "render_table_drop_collision.py -- --mode colliders "
            "--out-dir /tmp/colliders"
        )
    shape = p.createCollisionShape(
        p.GEOM_MESH, fileName=str(obj_path), meshScale=[1.0, 1.0, 1.0],
        flags=p.GEOM_FORCE_CONCAVE_TRIMESH, physicsClientId=client,
    )
    body = p.createMultiBody(0.0, shape, -1, [0.0, 0.0, 0.0], physicsClientId=client)
    p.changeDynamics(body, -1, lateralFriction=friction, rollingFriction=0.0,
                     spinningFriction=0.0, restitution=restitution,
                     collisionMargin=0.0005, physicsClientId=client)
    return body


def add_ball(client: int, radius: float, mass: float, position, friction: float,
             restitution: float, rolling_friction: float,
             spinning_friction: float) -> int:
    """A tennis ball: a sphere with the inertia of a shell.

    PyBullet works out the inertia of a GEOM_SPHERE as a solid body, 2/5 m r^2.
    A tennis ball is a pressurised rubber shell with nothing inside it, so the
    right figure is 2/3 m r^2. Overriding it is the only way the ball spins up
    and skids the way one actually does.
    """
    shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=client)
    body = p.createMultiBody(mass, shape, -1, list(position), physicsClientId=client)
    inertia = shell_inertia(mass, radius)
    p.changeDynamics(
        body, -1,
        localInertiaDiagonal=[inertia, inertia, inertia],
        lateralFriction=friction, rollingFriction=rolling_friction,
        spinningFriction=spinning_friction, restitution=restitution,
        linearDamping=0.0, angularDamping=0.0,
        collisionMargin=0.0005, physicsClientId=client,
    )
    return body


def horiz(v) -> float:
    return math.hypot(v[0], v[1])


def norm3(v) -> float:
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def heading_deg(v) -> float:
    return math.degrees(math.atan2(v[1], v[0])) % 360.0


def signed_turn(before, after) -> float:
    """Degrees the horizontal heading turned, positive counter-clockwise."""
    if horiz(before) < 1e-9 or horiz(after) < 1e-9:
        return float("nan")
    a = math.atan2(before[1], before[0])
    b = math.atan2(after[1], after[0])
    return math.degrees((b - a + math.pi) % (2.0 * math.pi) - math.pi)


def predict_flight(lip_position, lip_velocity, radius: float, gravity: float) -> dict:
    """The parabola the ball is committed to the moment it leaves the lip.

    Nothing touches the ball between the edge of the table and the floor, so the
    six numbers it has when it leaves fix the whole of the rest of it. Solving
    for the moment its centre comes back down to one radius above the floor,

        t = (vz + sqrt(vz^2 + 2 g (z - r))) / g          (g positive downward)

    and the reach is the horizontal speed times that. The fall time is the part
    worth pointing at: with vz = 0 it is sqrt(2 (z - r) / g), which contains no
    property of the ball at all -- not its mass, not its size, not its speed.
    """
    g = abs(gravity)
    drop = lip_position[2] - radius
    vz = lip_velocity[2]
    disc = vz * vz + 2.0 * g * drop
    if disc < 0.0 or g <= 0.0:
        return {}
    t = (vz + math.sqrt(disc)) / g
    speed_h = horiz(lip_velocity)
    return {
        "fall_time": t,
        # What the fall would take from rest at the same height: the number that
        # depends on nothing but the table.
        "fall_time_from_rest": math.sqrt(2.0 * drop / g),
        "drop": drop,
        "reach": speed_h * t,
        "landing": [lip_position[0] + lip_velocity[0] * t,
                    lip_position[1] + lip_velocity[1] * t,
                    radius],
        "impact_speed": math.hypot(speed_h, vz - g * t),
        "impact_angle_from_vertical_deg": math.degrees(
            math.atan2(speed_h, abs(vz - g * t))),
    }


def predict_impact(v_a, normal, m_a: float, m_b: float, e: float,
                   floor_friction: float) -> dict:
    """The closed form this scene's second half exists to check.

    Two spheres exchange momentum only along the line joining their centres, and
    ``normal`` is that line, pointing from the moving ball toward the one at
    rest. Everything across it is untouched:

        u    = v_A . n
        v_B' = (1 + e) m_A / (m_A + m_B) * u * n
        v_A' = v_A - (1 + e) m_B / (m_A + m_B) * u * n

    Done in three dimensions rather than in plan, because in this scene the
    normal is nowhere near horizontal: the ball arrives from the table 18.9 deg
    off vertical and catches the target high on its east face, so the line of
    centres comes out 21 deg below horizontal, and a plan-view treatment would
    throw away a third of the impact and all of the next paragraph.

    **The target is not a free body.** It is sitting on the floor, and the blow
    has a downward component -- ``nu u sin(theta)`` per unit mass, with theta the
    normal's tilt below horizontal. The floor cancels that within the contact,
    and the normal impulse it needs to do so drags on the ball through ordinary
    Coulomb friction, taking ``mu`` of it back out of the horizontal. So what
    the struck ball actually leaves with is

        v_B,horizontal = nu * u * cos(theta) * (1 - mu_floor * tan(theta))

    which at this scene's geometry is 78 per cent of the free-space figure: the
    smooth-sphere closed form says the target leaves at 1.619 m/s, this says
    1.258, and the solver returns 1.252. The correction is not a fudge factor --
    it is one line of Coulomb friction, with no constant in it that was not
    already in the scene -- and both numbers are reported so the two can be told
    apart.

    Its limit is worth stating rather than discovering. It assumes the target
    slides against the floor for as long as the contact lasts. Past a tilt of
    arctan(1/mu), about 60 deg here, ``mu tan(theta)`` passes 1 and the
    expression returns zero for a target that in fact still creeps away at 0.15
    m/s. Checked against the solver at tilts from 4 to 62 deg it is within 3 per
    cent below 35 deg, within 7 to 50, and wrong beyond 60;
    ``floor_correction_out_of_range`` says when a case has left that band.

    The struck ball's *heading* is untouched by any of this: the floor's friction
    acts along the ball's own direction of travel, so it changes the speed and
    not the aim. The aim is the line of centres and nothing else, which is why
    it is the tighter of the two checks.
    """
    n = list(normal)
    length = norm3(n)
    if length < 1e-9:
        return {}
    n = [c / length for c in n]
    u = sum(v_a[i] * n[i] for i in range(3))
    nu = (1.0 + e) * m_a / (m_a + m_b)
    lam = (1.0 + e) * m_b / (m_a + m_b)
    v_b = [nu * u * n[i] for i in range(3)]
    v_a_out = [v_a[i] - lam * u * n[i] for i in range(3)]

    tilt = math.asin(max(-1.0, min(1.0, -n[2])))
    floor_share = max(0.0, floor_friction * math.tan(tilt)) if tilt > 0.0 else 0.0
    b_h_free = horiz(v_b)
    return {
        "normal": n,
        "normal_below_horizontal_deg": math.degrees(tilt),
        "approach_along_normal": u,
        "b_velocity": v_b,
        "b_speed": norm3(v_b),
        "b_horizontal_speed_free": b_h_free,
        "b_horizontal_speed": b_h_free * max(0.0, 1.0 - floor_share),
        "floor_share_of_horizontal": floor_share,
        # Past mu tan(theta) ~ 1 the target stops sliding on the floor and the
        # correction has nothing left to say; 0.8 is where it starts drifting.
        "floor_correction_out_of_range": floor_share > 0.8,
        "b_heading_deg": heading_deg(v_b),
        "a_velocity": v_a_out,
        "a_speed": norm3(v_a_out),
        "a_horizontal_speed": horiz(v_a_out),
        "a_heading_deg": heading_deg(v_a_out),
        # With two of the same ball this is the whole story: the component along
        # the normal is handed over almost entire and the ball that did the
        # hitting is left with what was across it.
        "handover_fraction": nu,
        # Whether the ball that did the hitting is thrown back the way it came.
        # Judged in plan, because that is the part the camera can see: in three
        # dimensions the dropper is still falling hard after the contact and the
        # dot product is dominated by a vertical component nobody reads as
        # "forward".
        "a_horizontal_reverses": (
            v_a_out[0] * v_a[0] + v_a_out[1] * v_a[1] < 0.0),
    }


def off_floor(position):
    """Whether a ball has run off the end of the modelled floor."""
    x, y = position[0], position[1]
    if x < FLOOR_X[0]:
        return "west (past the window wall)"
    if x > FLOOR_X[1]:
        return "east"
    if y < FLOOR_Y[0]:
        return "south (out of the modelled room)"
    if y > SOFA_FRONT_Y:
        return "north (into the sofa)"
    return None


def simulate(args: argparse.Namespace) -> dict:
    fps = int(args.fps)
    frame_end = max(2, int(round(float(args.duration_sec) * fps)))
    substeps = int(args.substeps)
    dt = 1.0 / float(fps * substeps)

    r_a = float(args.ball_a_radius)
    r_b = float(args.ball_b_radius)
    m_a = float(args.ball_a_mass)
    m_b = float(args.ball_b_mass)
    e_eff = float(args.ball_a_restitution) * float(args.ball_b_restitution)
    gravity = float(args.gravity_z)

    heading = math.radians(float(args.launch_heading_deg))
    speed = float(args.launch_speed)
    velocity = (speed * math.cos(heading), speed * math.sin(heading), 0.0)
    # Launched already rolling rather than sliding: for a contact directly
    # underneath, rolling without slipping is omega = (z_hat x v) / r. From rest
    # the ball spends its first tenth of a second skidding while friction spins
    # it up, which reads as a shove and not a roll -- and on a shell, which takes
    # 2/3 m r^2 rather than 2/5 to spin up, it takes noticeably longer still.
    spin = (-velocity[1] / r_a, velocity[0] / r_a, 0.0)

    ball_a_start = (float(args.launch_x), float(args.launch_y), TABLE_TOP_Z + r_a)
    ball_b_start = (float(args.ball_b_x), float(args.ball_b_y), r_b)

    client = p.connect(p.DIRECT)
    try:
        p.resetSimulation(physicsClientId=client)
        p.setGravity(0.0, 0.0, gravity, physicsClientId=client)
        p.setTimeStep(dt, physicsClientId=client)
        p.setPhysicsEngineParameter(
            fixedTimeStep=dt,
            numSolverIterations=300,
            contactBreakingThreshold=0.001,
            deterministicOverlappingPairs=1,
            enableConeFriction=1,
            physicsClientId=client,
        )

        table_id = add_table(client, args)
        floor_id = add_floor(client, args)
        props_id = None
        if args.props_collider is not None:
            props_id = add_props(client, args.props_collider,
                                 float(args.props_friction),
                                 float(args.props_restitution))

        # ball_a is always created -- the whole simulation is centred on its
        # roll off the table. Only ball_b is optional.
        ball_b_active = bool(int(args.ball_b_active)) and not bool(args.disable_ball_b)

        ball_a = add_ball(client, r_a, m_a, ball_a_start,
                          float(args.ball_a_friction), float(args.ball_a_restitution),
                          float(args.ball_rolling_friction),
                          float(args.ball_spinning_friction))
        ball_b = None
        if ball_b_active:
            ball_b = add_ball(client, r_b, m_b, ball_b_start,
                              float(args.ball_b_friction),
                              float(args.ball_b_restitution),
                              float(args.ball_rolling_friction),
                              float(args.ball_spinning_friction))

        p.resetBaseVelocity(ball_a, linearVelocity=velocity, angularVelocity=spin,
                            physicsClientId=client)

        lip = None              # state the moment the ball leaves the table
        touchdown = None        # first floor contact, only meaningful without B
        hit = {"frame": None, "time": None, "point": None, "normal": None,
               "a_vel_in": None, "a_vel_out": None, "b_vel_out": None,
               "a_pos_in": None, "b_pos_in": None}
        touching = False
        prop_contact_frame = None
        ran_off = None

        frames = []
        settle_a = None
        settle_b = None
        prev_a_vel = velocity
        prev_a_pos = ball_a_start
        # Speed sampled at the launch and again just before the lip, so the
        # deceleration the table's rolling friction actually produces is measured
        # rather than asserted.
        roll_start_pos = ball_a_start
        # Substep-resolution clock. The frame index is far too coarse to time
        # either event with: at 24 fps the ball covers 44 mm between frames,
        # two thirds of its own diameter, and the whole free flight is seven and
        # a half frames long. Timing the lip and the impact to the frame would
        # put a 30 mm error into the flight check on its own.
        sim_time = 0.0

        for frame_index in range(1, frame_end + 1):
            for _ in range(substeps if frame_index > 1 else 0):
                p.stepSimulation(physicsClientId=client)
                sim_time += dt
                now = sim_time

                a_pos, _ = p.getBasePositionAndOrientation(ball_a, physicsClientId=client)
                a_vel, _ = p.getBaseVelocity(ball_a, physicsClientId=client)

                # The lip: the last substep at which the ball's contact point was
                # still over the top. Everything downstream is predicted from
                # here, so it is taken at substep resolution -- at 24 fps the ball
                # travels 44 mm in a frame, which is two thirds of its diameter.
                if lip is None and math.hypot(a_pos[0], a_pos[1]) > TABLE_RADIUS:
                    lip = {
                        "frame": frame_index,
                        # prev_a_* was sampled before this step, so that is when
                        # the ball was still on the table.
                        "time": now - dt,
                        "position": list(prev_a_pos),
                        "velocity": list(prev_a_vel),
                        "speed_h": horiz(prev_a_vel),
                        "roll_distance": math.dist(
                            (roll_start_pos[0], roll_start_pos[1]),
                            (prev_a_pos[0], prev_a_pos[1])),
                    }

                if touchdown is None and lip is not None:
                    if p.getContactPoints(bodyA=ball_a, bodyB=floor_id,
                                          physicsClientId=client):
                        touchdown = {"frame": frame_index, "time": now,
                                     "position": list(a_pos),
                                     "velocity": list(a_vel)}

                if ball_b is not None:
                    # Only the *first* touch is the scene's impact. The dropper
                    # bounces and can clip the target again a few frames later,
                    # and letting that overwrite the first reading turns a clean
                    # handover into a nudge.
                    points = p.getContactPoints(bodyA=ball_a, bodyB=ball_b,
                                                physicsClientId=client)
                    if points and not touching:
                        touching = True
                        if hit["frame"] is None:
                            cp = max(points, key=lambda c: -c[8])
                            b_pos, _ = p.getBasePositionAndOrientation(
                                ball_b, physicsClientId=client)
                            hit["frame"] = frame_index
                            hit["time"] = now - dt
                            hit["point"] = list(cp[5])
                            # cp[7] is the contact normal on B, which points from
                            # B toward A; the line of centres wanted here runs the
                            # other way.
                            hit["normal"] = [-c for c in cp[7]]
                            hit["a_vel_in"] = list(prev_a_vel)
                            hit["a_pos_in"] = list(prev_a_pos)
                            hit["b_pos_in"] = list(b_pos)
                    elif not points and touching:
                        touching = False
                        if hit["a_vel_out"] is None:
                            b_vel, _ = p.getBaseVelocity(ball_b, physicsClientId=client)
                            hit["a_vel_out"] = list(a_vel)
                            hit["b_vel_out"] = list(b_vel)

                if props_id is not None and prop_contact_frame is None:
                    for body in (ball_a, ball_b):
                        if body is None:
                            continue
                        if p.getContactPoints(bodyA=body, bodyB=props_id,
                                              physicsClientId=client):
                            prop_contact_frame = frame_index
                            break

                prev_a_vel = a_vel
                prev_a_pos = a_pos

            record = {"frame_index": frame_index,
                      "time_sec": (frame_index - 1) / float(fps)}
            spin_now = {}
            for name, body, radius in (("ball_a", ball_a, r_a),
                                       ("ball_b", ball_b, r_b)):
                if body is None:
                    continue
                pos, quat = p.getBasePositionAndOrientation(body, physicsClientId=client)
                lin, ang = p.getBaseVelocity(body, physicsClientId=client)
                on_table = (math.hypot(pos[0], pos[1]) <= TABLE_RIM_RADIUS
                            and pos[2] > TABLE_TOP_Z - radius)
                record[name] = {
                    "location": list(pos),
                    "quaternion_xyzw": list(quat),
                    "linear_velocity": list(lin),
                    "angular_velocity": list(ang),
                    "speed": norm3(lin),
                    "height_above_floor": pos[2] - radius,
                    "spin_z": ang[2],
                    "on_table": bool(on_table),
                    "airborne": bool(pos[2] > radius + 0.002 and not on_table),
                }
                spin_now[name] = norm3(ang)
                edge = off_floor(pos)
                if edge is not None and ran_off is None:
                    ran_off = {"ball": name, "frame": frame_index, "edge": edge,
                               "location": list(pos)}

            # "Settled" means stopped moving *and* stopped turning. Speed alone
            # is not enough: a ball can sit at 0.000 m/s and go on spinning about
            # the vertical indefinitely.
            a_still = (record["ball_a"]["speed"] < SETTLE_SPEED
                       and spin_now["ball_a"] < SETTLE_SPIN)
            settle_a = frame_index if (a_still and settle_a is None) else (
                settle_a if a_still else None)
            if ball_b is not None:
                b_still = (record["ball_b"]["speed"] < SETTLE_SPEED
                           and spin_now["ball_b"] < SETTLE_SPIN)
                if hit["frame"] is not None and b_still:
                    settle_b = frame_index if settle_b is None else settle_b
                elif not b_still:
                    settle_b = None
                record["ball_b"]["moving"] = bool(
                    hit["frame"] is not None and frame_index >= hit["frame"])

            record["ball_a"]["phase"] = (
                "rolling" if lip is None or frame_index < lip["frame"] else
                "falling" if hit["frame"] is None or frame_index < hit["frame"]
                else "after_impact"
            )
            frames.append(record)

        last = frames[-1]
        residual_spin = {
            name: norm3(last[name]["angular_velocity"])
            for name in ("ball_a", "ball_b") if name in last
        }

        # --- the two predictions, and what actually happened ------------------
        flight = {}
        flight_error = None
        if lip is not None:
            flight = predict_flight(lip["position"], lip["velocity"], r_a, gravity)
            if flight:
                if hit["frame"] is not None:
                    # With the target in the way the ball never gets a clean
                    # touchdown -- it is deflected first, and whatever it does
                    # with the floor afterwards says nothing about the parabola.
                    # So the parabola is checked where it still can be: against
                    # the ball's own position at the moment of contact.
                    t = hit["time"] - lip["time"]
                    predicted = [
                        lip["position"][i] + lip["velocity"][i] * t
                        + (0.5 * gravity * t * t if i == 2 else 0.0)
                        for i in range(3)
                    ]
                    flight_error = math.dist(
                        (predicted[0], predicted[1], predicted[2]),
                        tuple(hit["a_pos_in"]))
                    flight["predicted_position_at_contact"] = predicted
                elif touchdown is not None:
                    # No target: the ball reaches the floor on its own, and the
                    # predicted landing can be compared with a real one. This is
                    # the run that checks the projectile formula against
                    # something other than another application of itself, and it
                    # is how the target's position was set in the first place.
                    flight_error = math.dist(
                        (flight["landing"][0], flight["landing"][1]),
                        (touchdown["position"][0], touchdown["position"][1]))

        roll_deceleration = None
        if lip is not None and lip["roll_distance"] > 1e-6:
            roll_deceleration = ((speed ** 2 - lip["speed_h"] ** 2)
                                 / (2.0 * lip["roll_distance"]))

        impact = {}
        if hit["frame"] is not None and hit["normal"] is not None:
            impact = predict_impact(hit["a_vel_in"], hit["normal"], m_a, m_b,
                                    e_eff, float(args.floor_friction))
            if hit["b_vel_out"] is not None:
                impact["measured_b_velocity"] = hit["b_vel_out"]
                impact["measured_b_horizontal_speed"] = horiz(hit["b_vel_out"])
                impact["measured_b_heading_deg"] = heading_deg(hit["b_vel_out"])
                impact["measured_a_velocity"] = hit["a_vel_out"]
                impact["measured_a_horizontal_speed"] = horiz(hit["a_vel_out"])
                impact["measured_a_heading_deg"] = heading_deg(hit["a_vel_out"])
                impact["separation_deg"] = abs(
                    signed_turn(hit["a_vel_out"], hit["b_vel_out"]))

        travel = {}
        for name, start in (("ball_a", ball_a_start), ("ball_b", ball_b_start)):
            if name not in last:
                continue
            travel[name] = math.dist(
                (last[name]["location"][0], last[name]["location"][1]),
                (start[0], start[1]))

        return {
            "schema_version": 1,
            "simulator": "pybullet",
            "fps": fps,
            "frame_start": 1,
            "frame_end": frame_end,
            "duration_sec": float(args.duration_sec),
            "substeps_per_frame": substeps,
            "physics_dt": dt,
            "objects": {
                "ball_a": {
                    "radius": r_a, "mass": m_a, "diameter": 2.0 * r_a,
                    "inertia": shell_inertia(m_a, r_a),
                    "inertia_factor": SHELL_INERTIA_FACTOR,
                    "solid_sphere_inertia": 0.4 * m_a * r_a ** 2,
                    "initial_location": list(ball_a_start),
                    "initial_linear_velocity": list(velocity),
                    "initial_angular_velocity": list(spin),
                    "restitution": float(args.ball_a_restitution),
                    "friction": float(args.ball_a_friction),
                },
                "ball_b": {
                    "radius": r_b, "mass": m_b, "diameter": 2.0 * r_b,
                    "inertia": shell_inertia(m_b, r_b),
                    "initial_location": list(ball_b_start),
                    "restitution": float(args.ball_b_restitution),
                    "friction": float(args.ball_b_friction),
                    "present": ball_b is not None,
                },
                "table": {
                    "top_z": TABLE_TOP_Z,
                    "radius": TABLE_RADIUS,
                    "rim_radius": TABLE_RIM_RADIUS,
                    "plate_thickness": TABLE_PLATE,
                    "friction": float(args.table_friction),
                    "rolling_friction": float(args.table_rolling_friction),
                    "spinning_friction": float(args.table_spinning_friction),
                    "restitution": float(args.table_restitution),
                },
                "floor": {
                    "z": 0.0,
                    "friction": float(args.floor_friction),
                    "rolling_friction": float(args.floor_rolling_friction),
                    "restitution": float(args.floor_restitution),
                },
            },
            "launch": {
                "position": list(ball_a_start),
                "speed": speed,
                "heading_deg": float(args.launch_heading_deg),
                "lip": lip,
                "roll_deceleration": roll_deceleration,
                "predicted_flight": flight,
                "touchdown": touchdown,
                "flight_prediction_error": flight_error,
            },
            "collision": {
                "mass_ratio": m_a / m_b,
                "effective_restitution": e_eff,
                "centre_separation": r_a + r_b,
                "predicted": impact,
            },
            "quality": {
                "left_table": lip is not None,
                "lip_frame": None if lip is None else lip["frame"],
                "hit_ball_b": hit["frame"] is not None,
                "contact_frame": hit["frame"],
                "contact_time": hit["time"],
                "contact_point": hit["point"],
                "contact_normal": hit["normal"],
                "airborne_frames": sum(1 for fr in frames
                                       if fr["ball_a"]["airborne"]),
                "ball_a_final": last["ball_a"]["location"],
                "ball_b_final": None if ball_b is None else last["ball_b"]["location"],
                "ball_a_travel": travel.get("ball_a"),
                "ball_b_travel": travel.get("ball_b"),
                "ball_a_settled_frame": settle_a,
                "ball_b_settled_frame": settle_b,
                "residual_spin": residual_spin,
                "spinning_on_the_spot": any(v > RESIDUAL_SPIN_LIMIT
                                            for v in residual_spin.values()),
                "hit_table_dressing": prop_contact_frame is not None,
                "prop_contact_frame": prop_contact_frame,
                "ran_off_floor": ran_off,
            },
            "frames": frames,
        }
    finally:
        p.disconnect(client)


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    records = simulate(args)
    args.out.write_text(json.dumps(records, indent=2), encoding="utf-8")
    q = records["quality"]
    launch = records["launch"]
    imp = records["collision"]["predicted"]

    def fmt(value, spec="%.3f"):
        return "None" if value is None else spec % value

    lip = launch["lip"] or {}
    flight = launch["predicted_flight"] or {}
    print(
        "[SIM] lip f=%s v=%s run=%s decel=%s | fall t=%s reach=%s land=%s "
        "err=%s | hit f=%s n_below=%s deg | b_out %s (pred %s, free %s) head %s/%s | "
        "a_out %s (pred %s) rev=%s | sep=%s | ends A=%s B=%s settle=(%s, %s) "
        "spin_left=%s dressing=%s off_floor=%s" % (
            q["lip_frame"], fmt(lip.get("speed_h")), fmt(lip.get("roll_distance")),
            fmt(launch["roll_deceleration"]),
            fmt(flight.get("fall_time")), fmt(flight.get("reach")),
            "(%.3f, %.3f)" % (flight["landing"][0], flight["landing"][1])
            if flight.get("landing") else "None",
            fmt(launch["flight_prediction_error"], "%.4f"),
            q["contact_frame"], fmt(imp.get("normal_below_horizontal_deg"), "%.1f"),
            fmt(imp.get("measured_b_horizontal_speed")),
            fmt(imp.get("b_horizontal_speed")),
            fmt(imp.get("b_horizontal_speed_free")),
            fmt(imp.get("measured_b_heading_deg"), "%.0f"),
            fmt(imp.get("b_heading_deg"), "%.0f"),
            fmt(imp.get("measured_a_horizontal_speed")),
            fmt(imp.get("a_horizontal_speed")),
            imp.get("a_horizontal_reverses"),
            fmt(imp.get("separation_deg"), "%.1f"),
            "(%.3f, %.3f)" % (q["ball_a_final"][0], q["ball_a_final"][1]),
            None if q["ball_b_final"] is None else
            "(%.3f, %.3f)" % (q["ball_b_final"][0], q["ball_b_final"][1]),
            q["ball_a_settled_frame"], q["ball_b_settled_frame"],
            "/".join("%.2f" % v for v in q["residual_spin"].values()),
            q["prop_contact_frame"],
            None if q["ran_off_floor"] is None else
            f"{q['ran_off_floor']['ball']}@{q['ran_off_floor']['frame']}"
            f" {q['ran_off_floor']['edge']}",
        )
    )


if __name__ == "__main__":
    main()
