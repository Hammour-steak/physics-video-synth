from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pybullet as p


# A big glass marble is rolled the length of a kitchen bar table into a small
# one sitting at rest, and knocks it away.
#
# The scene is built around one fact: **the mass ratio is the cube of the size
# ratio.** The two balls are the same glass; the big one is twice the diameter,
# so it is eight times the mass, and that is the whole reason the small one
# leaves faster than the big one arrived while the big one barely slows down.
# Nothing about that is visible in the two balls other than how big they are,
# which is what makes it worth measuring.
#
# For a head-on impact between masses m_A and m_B with coefficient of
# restitution e, the struck ball leaves at
#
#     v_B = (1 + e) * m_A / (m_A + m_B) * v_A
#
# so at 8:1 and e = 0.76 the small ball leaves at 1.56 times the speed the big
# one came in at, and the big one keeps 0.81 of its own. Both of those are
# recorded against the measured values in the ground truth, because a solver
# that quietly disagrees with the closed form is the thing most likely to be
# wrong here.
#
# The impact is deliberately not *quite* head-on. Dead centre, the two balls
# come away on the same line and the small one simply runs away in front of the
# big one, which is a tidy demonstration and a poor picture: nothing in the
# frame separates. With the small ball set 12 mm off the big one's line the two
# leave on headings 12 deg apart -- the small one bending toward the table's
# front edge, the big one drifting back -- and the split is legible.
#
# Coordinates are metric and match render_table_marble_collision.py's world
# frame: z = 0 is the bar table's top surface, the origin is at the centre of
# that surface, +x runs east along the table's length and +y north, out of the
# wall the table stands against and toward the bar stools.

# The table top, measured off assets/models/modern_living_room.glb by
# raycasting rather than taken from a bounding box; see the render script's
# calibration block. It is one dead-flat rectangle -- 1701 probe points over the
# clear area all came back at the same z to four decimals.
TABLE_X = (-0.9152, 0.9152)
TABLE_Y = (-0.2897, 0.2897)
TABLE_PLATE = 0.0509        # thickness of the top plate

# The table stands against a wall along its whole south edge, so a ball drifting
# south stops rather than falling. The wall goes in as a static box whose face
# sits on the table's south edge; in the model it is about 8 mm further south
# than that, which is inside the ball radii and not worth modelling.
WALL_Y = TABLE_Y[0]

FLOOR_Z = -0.900            # the ceramic floor, one bar-table height down

SETTLE_SPEED = 0.02         # m/s below which a marble counts as stopped
SETTLE_SPIN = 0.30          # rad/s below which it counts as stopped *turning*
RESIDUAL_SPIN_LIMIT = 0.30

GLASS_DENSITY = 2500.0      # kg/m^3, soda-lime glass


def sphere_mass(radius: float, density: float = GLASS_DENSITY) -> float:
    return density * 4.0 / 3.0 * math.pi * radius ** 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--props-collider", type=Path, default=None,
        help="Metre-scale OBJ of the dressing already standing on the table -- "
        "the wooden tray and the two brass candlesticks -- written by "
        "render_table_marble_collision.py's --mode colliders pass. It is a "
        "separate body from the table so that touching it is reported as its "
        "own event: the hero take passes 0.12 m clear of it, and a ball that "
        "reaches it has gone somewhere the shot does not intend.",
    )
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-sec", type=float, default=3.0)
    parser.add_argument(
        "--substeps", type=int, default=60,
        help="Physics steps per rendered frame. The impact is resolved inside a "
        "single substep, and speeds either side of it are sampled at substep "
        "resolution: at 24 fps one frame is long enough for the table to take a "
        "measurable slice out of the struck ball before anything is recorded.",
    )
    parser.add_argument("--gravity-z", type=float, default=-9.8)

    # --- the big marble, the one that is rolled ---
    parser.add_argument("--ball-a-radius", type=float, default=0.050)
    parser.add_argument(
        "--ball-a-mass", type=float, default=None,
        help="Defaults to a solid glass sphere of --ball-a-radius at "
        f"{GLASS_DENSITY:.0f} kg/m^3. Pass a value only to break the link "
        "between the size you can see and the mass that does the work.",
    )
    parser.add_argument(
        "--ball-a-restitution", type=float, default=0.87,
        help="Bullet multiplies the two bodies' restitutions, so the impact "
        "actually sees this times --ball-b-restitution. Glass on glass is "
        "lively: 0.87 each gives an effective 0.76.",
    )
    parser.add_argument("--ball-a-friction", type=float, default=0.30)

    # --- the small marble, the one that is struck ---
    parser.add_argument("--ball-b-radius", type=float, default=0.025)
    parser.add_argument("--ball-b-mass", type=float, default=None)
    parser.add_argument("--ball-b-restitution", type=float, default=0.87)
    parser.add_argument("--ball-b-friction", type=float, default=0.30)
    parser.add_argument("--ball-b-x", type=float, default=0.060)
    parser.add_argument(
        "--ball-b-y", type=float, default=0.008,
        help="Where the struck marble waits. The gap between this and "
        "--launch-y is the impact parameter, and it is the only thing that "
        "decides how far apart the two balls leave: 12 mm on a 75 mm centre "
        "separation is 9.2 deg off the centre line.",
    )
    parser.add_argument(
        "--disable-ball-b", action="store_true",
        help="Leave the struck marble out. Used to calibrate the run-up: the "
        "speed at the impact point has to be known before there is any point "
        "predicting what the impact does.",
    )
    parser.add_argument(
        "--ball-rolling-friction", type=float, default=0.0,
        help="Kept at 0 on purpose so all rolling resistance comes from the "
        "table. Bullet does not combine rolling friction as a plain product: it "
        "uses rf_a * lateral_b + rf_b * lateral_a, so giving a ball rf = 1 -- "
        "the obvious way to let the surface's value pass through -- multiplies "
        "in the *table's lateral* friction instead and turns the roll into a "
        "skid. With the balls at 0 and lateral friction on the table, the "
        "effective rolling resistance is exactly the table's own value.",
    )
    parser.add_argument(
        "--ball-spinning-friction", type=float, default=0.0,
        help="Zero for the same reason, and it is the same trap: spinning "
        "friction combines as sf_a * lateral_b + sf_b * lateral_a too.",
    )

    # --- surfaces ---
    parser.add_argument(
        "--table-rolling-friction", type=float, default=0.0200,
        help="Rolling resistance of the table top, and this shot's clock. It "
        "sets how much speed the big marble still carries into the impact and "
        "how far each ball runs afterwards, and it is what makes the shot fit "
        "on the table at all -- see the README: 1.83 m is not much table, and at "
        "this value it is 0.83 m/s^2 of deceleration for the big marble and 1.65 "
        "for the small one. Bullet's rolling friction acts as a contact-offset "
        "length rather than a dimensionless coefficient, so the deceleration "
        "goes as 1/radius: the small marble slows twice as fast as the big one, "
        "which is exactly why it can leave at 1.5 times the speed and still "
        "come to rest 0.19 m short of the far end.",
    )
    parser.add_argument("--table-friction", type=float, default=0.42)
    parser.add_argument("--table-restitution", type=float, default=0.30)
    parser.add_argument(
        "--table-spinning-friction", type=float, default=0.006,
        help="Resistance to a ball turning on the spot about the vertical. An "
        "oblique impact puts real spin about that axis into both balls, and "
        "nothing else in Bullet opposes it -- rolling friction resists rolling, "
        "lateral friction resists sliding. Left at zero, a ball comes to a dead "
        "stop and goes on turning where it stands, forever, which is plainly "
        "visible in the render.",
    )
    parser.add_argument("--props-friction", type=float, default=0.55)
    parser.add_argument("--props-restitution", type=float, default=0.20)
    parser.add_argument("--wall-friction", type=float, default=0.50)
    parser.add_argument("--wall-restitution", type=float, default=0.30)
    parser.add_argument("--floor-friction", type=float, default=0.55)
    parser.add_argument("--floor-restitution", type=float, default=0.42)

    # --- the push ---
    parser.add_argument(
        "--launch-x", type=float, default=-0.620,
        help="Where the big marble is set rolling. Short of the table's west end "
        "on purpose: the chosen framing comes in close enough that a launch at "
        "-0.780 clipped the right edge of frame, and shortening the run-up was "
        "the cheaper fix than moving the camera back out.",
    )
    parser.add_argument("--launch-y", type=float, default=0.020)
    parser.add_argument(
        "--launch-speed", type=float, default=1.31,
        help="Initial speed in m/s: a firm hand roll. After 0.62 m of table the "
        "big marble arrives at the impact at 0.83 m/s. The default is about 5 "
        "per cent below the push that sends the struck marble off the far end, "
        "which is deliberate -- it is what makes a harder push a useful case "
        "rather than a repeat of the hero.",
    )
    parser.add_argument("--launch-heading-deg", type=float, default=0.0)
    # PCVE DELETE edit for the small marble: 0 removes ball_b, equivalent
    # to --disable-ball-b. Big marble is always present -- the whole
    # simulation is built around its roll.
    parser.add_argument("--ball-b-active", type=int, default=1)
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
        # Bullet's default 0.04 m collision margin would inflate a box whose
        # edges are the thing a ball is supposed to fall off by 40 mm on every
        # side, which is most of the small marble's diameter.
        collisionMargin=0.0005, physicsClientId=client,
    )
    return body


def add_table(client: int, args: argparse.Namespace) -> int:
    """The bar table's top plate, top face at z = 0.

    A box rather than the model's own triangles, and for once that is the
    honest choice rather than the cheap one: the top *is* a flat rectangle, and
    its extent was measured by raycasting the model at 0.2-unit steps until the
    surface dropped away. A trimesh would add the metal frame underneath, which
    no ball on the table can reach.
    """
    half = ((TABLE_X[1] - TABLE_X[0]) / 2.0, (TABLE_Y[1] - TABLE_Y[0]) / 2.0,
            TABLE_PLATE / 2.0)
    centre = ((TABLE_X[0] + TABLE_X[1]) / 2.0, (TABLE_Y[0] + TABLE_Y[1]) / 2.0,
              -TABLE_PLATE / 2.0)
    return add_box(client, half, centre, float(args.table_friction),
                   float(args.table_restitution),
                   float(args.table_rolling_friction),
                   float(args.table_spinning_friction))


def add_wall(client: int, args: argparse.Namespace) -> int:
    """The wall the table's south edge stands against."""
    thickness = 0.10
    height = 2.0
    return add_box(
        client, (2.0, thickness / 2.0, height / 2.0),
        (0.0, WALL_Y - thickness / 2.0, FLOOR_Z + height / 2.0),
        float(args.wall_friction), float(args.wall_restitution),
    )


def add_floor(client: int, args: argparse.Namespace) -> int:
    """The ceramic floor, so a ball that leaves the table lands on something."""
    return add_box(client, (4.0, 4.0, 0.25), (0.0, 0.0, FLOOR_Z - 0.25),
                   float(args.floor_friction), float(args.floor_restitution))


def add_props(client: int, obj_path: Path, friction: float, restitution: float) -> int:
    if not obj_path.exists():
        raise FileNotFoundError(
            f"Table dressing collision mesh not found: {obj_path}\n"
            "Generate it first with:\n"
            "  blender -b --python scripts/table_marble_collision/"
            "render_table_marble_collision.py -- --mode colliders "
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
    shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=client)
    body = p.createMultiBody(mass, shape, -1, list(position), physicsClientId=client)
    p.changeDynamics(
        body, -1,
        lateralFriction=friction, rollingFriction=rolling_friction,
        spinningFriction=spinning_friction, restitution=restitution,
        linearDamping=0.0, angularDamping=0.0,
        collisionMargin=0.0005, physicsClientId=client,
    )
    return body


def horiz(v) -> float:
    return math.hypot(v[0], v[1])


def heading_deg(v) -> float:
    return math.degrees(math.atan2(v[1], v[0])) % 360.0


def signed_turn(before, after) -> float:
    """Degrees the horizontal heading turned, positive counter-clockwise."""
    if horiz(before) < 1e-9 or horiz(after) < 1e-9:
        return float("nan")
    a = math.atan2(before[1], before[0])
    b = math.atan2(after[1], after[0])
    return math.degrees((b - a + math.pi) % (2.0 * math.pi) - math.pi)


def predict_impact(m_a: float, m_b: float, e: float, obliquity_deg: float,
                   side: float) -> dict:
    """The closed form this scene exists to check, for an oblique impact.

    Two smooth spheres exchange momentum only along the line joining their
    centres; the component across it is untouched. Working in a frame whose x
    axis is the big ball's approach, with the centre line at ``obliquity`` off
    it on the ``side`` the struck ball sits:

        n           = (cos t, side * sin t)
        v_B'        = nu * v cos t * n            nu = (1 + e) m_A / (m_A + m_B)
        v_A'        = v x_hat - lam * v cos t * n  lam = (1 + e) m_B / (m_A + m_B)

    The struck ball therefore leaves *exactly* along the centre line whatever
    the masses are -- ``obliquity`` off the approach, and that is a prediction
    with no free parameters in it at all. The big ball's outgoing direction does
    depend on the masses, and its along-approach component changes sign once
    ``lam cos^2 t`` passes 1: heavier struck ball, and the big one comes back.

    Doing this properly rather than as ``1 - lam cos t`` matters at the far end
    of the suite. Near head-on the two agree to a fraction of a percent, but for
    equal masses the approximation puts the big ball's outgoing speed at 0.12 of
    its approach when the correct value is 0.17, and for a struck ball 2.7 times
    heavier it gives a negative *speed*.
    """
    t = math.radians(obliquity_deg)
    cos_t, sin_t = math.cos(t), math.sin(t)
    nu = (1.0 + e) * m_a / (m_a + m_b)
    lam = (1.0 + e) * m_b / (m_a + m_b)

    b_along, b_across = nu * cos_t * cos_t, nu * cos_t * side * sin_t
    a_along = 1.0 - lam * cos_t * cos_t
    a_across = -lam * cos_t * side * sin_t

    a_turn = math.degrees(math.atan2(a_across, a_along))
    b_turn = math.degrees(math.atan2(b_across, b_along))
    return {
        "b_speed_ratio": math.hypot(b_along, b_across),
        "a_speed_ratio": math.hypot(a_along, a_across),
        # Signed component along the approach: negative means the big ball is
        # sent back the way it came, which a magnitude cannot say.
        "a_along_approach_ratio": a_along,
        "a_reverses": a_along < 0.0,
        "a_turn_deg": a_turn,
        "b_turn_deg": b_turn,
        "separation_deg": abs(a_turn - b_turn),
    }


def off_table(position, radius: float):
    """Which table edge a ball's centre has crossed, if any."""
    x, y = position[0], position[1]
    if x < TABLE_X[0]:
        # Not a fall: the table's west end is flush with the kitchen worktop and
        # the surface simply carries on. Reported anyway, because the simulation
        # stops modelling a surface here and the render does not.
        return "west (onto the kitchen worktop)"
    if x > TABLE_X[1]:
        return "east"
    if y > TABLE_Y[1]:
        return "north (the open side, over the bar stools)"
    if y < TABLE_Y[0]:
        return "south (into the wall)"
    return None


def simulate(args: argparse.Namespace) -> dict:
    fps = int(args.fps)
    frame_end = max(2, int(round(float(args.duration_sec) * fps)))
    substeps = int(args.substeps)
    dt = 1.0 / float(fps * substeps)

    r_a = float(args.ball_a_radius)
    r_b = float(args.ball_b_radius)
    m_a = sphere_mass(r_a) if args.ball_a_mass is None else float(args.ball_a_mass)
    m_b = sphere_mass(r_b) if args.ball_b_mass is None else float(args.ball_b_mass)
    e_eff = float(args.ball_a_restitution) * float(args.ball_b_restitution)

    heading = math.radians(float(args.launch_heading_deg))
    speed = float(args.launch_speed)
    velocity = (speed * math.cos(heading), speed * math.sin(heading), 0.0)

    # Launched already rolling rather than sliding. For a contact directly
    # underneath, rolling without slipping means omega = (z_hat x v) / r; from
    # rest the marble spends its first tenth of a second skidding while friction
    # spins it up, which reads as a shove and not a roll.
    spin = (-velocity[1] / r_a, velocity[0] / r_a, 0.0)

    ball_a_start = (float(args.launch_x), float(args.launch_y), r_a)
    ball_b_start = (float(args.ball_b_x), float(args.ball_b_y), r_b)
    impact_parameter = abs(ball_a_start[1] - ball_b_start[1])
    separation = r_a + r_b
    obliquity_deg = math.degrees(math.asin(min(1.0, impact_parameter / separation)))

    client = p.connect(p.DIRECT)
    try:
        p.resetSimulation(physicsClientId=client)
        p.setGravity(0.0, 0.0, float(args.gravity_z), physicsClientId=client)
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
        add_wall(client, args)
        add_floor(client, args)
        props_id = None
        if args.props_collider is not None:
            props_id = add_props(client, args.props_collider,
                                 float(args.props_friction),
                                 float(args.props_restitution))

        ball_a = add_ball(client, r_a, m_a, ball_a_start,
                          float(args.ball_a_friction), float(args.ball_a_restitution),
                          float(args.ball_rolling_friction),
                          float(args.ball_spinning_friction))
        ball_b = None
        if not args.disable_ball_b and bool(int(args.ball_b_active)):
            ball_b = add_ball(client, r_b, m_b, ball_b_start,
                              float(args.ball_b_friction),
                              float(args.ball_b_restitution),
                              float(args.ball_rolling_friction),
                              float(args.ball_spinning_friction))

        p.resetBaseVelocity(ball_a, linearVelocity=velocity, angularVelocity=spin,
                            physicsClientId=client)

        hit = {"frame": None, "time": None, "point": None, "normal": None,
               "a_speed_in": None, "a_speed_out": None, "b_speed_out": None,
               "a_vel_in": None, "a_vel_out": None, "b_vel_out": None}
        touching = False
        prop_contact_frame = None
        left_table = None

        frames = []
        settle_a = None
        settle_b = None
        prev_a_vel = velocity

        for frame_index in range(1, frame_end + 1):
            for _ in range(substeps if frame_index > 1 else 0):
                p.stepSimulation(physicsClientId=client)
                now = (frame_index - 1) / float(fps)

                a_vel, _ = p.getBaseVelocity(ball_a, physicsClientId=client)

                if ball_b is not None:
                    # Only the *first* touch is the scene's impact. In the
                    # softer cases the big marble catches the small one up again
                    # later, and letting that overwrite the first reading turns a
                    # 1.5x speed transfer into a nudge.
                    points = p.getContactPoints(bodyA=ball_a, bodyB=ball_b,
                                                physicsClientId=client)
                    if points and not touching:
                        touching = True
                        if hit["frame"] is None:
                            cp = max(points, key=lambda c: -c[8])
                            hit["frame"] = frame_index
                            hit["time"] = now
                            hit["point"] = list(cp[5])
                            hit["normal"] = list(cp[7])
                            hit["a_speed_in"] = horiz(prev_a_vel)
                            hit["a_vel_in"] = list(prev_a_vel)
                    elif not points and touching:
                        touching = False
                        if hit["a_speed_out"] is None:
                            b_vel, _ = p.getBaseVelocity(ball_b, physicsClientId=client)
                            hit["a_speed_out"] = horiz(a_vel)
                            hit["b_speed_out"] = horiz(b_vel)
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

            record = {"frame_index": frame_index,
                      "time_sec": (frame_index - 1) / float(fps)}
            spin_now = {}
            for name, body, radius in (("ball_a", ball_a, r_a),
                                       ("ball_b", ball_b, r_b)):
                if body is None:
                    continue
                pos, quat = p.getBasePositionAndOrientation(body, physicsClientId=client)
                lin, ang = p.getBaseVelocity(body, physicsClientId=client)
                record[name] = {
                    "location": list(pos),
                    "quaternion_xyzw": list(quat),
                    "linear_velocity": list(lin),
                    "angular_velocity": list(ang),
                    "speed": math.sqrt(sum(v ** 2 for v in lin)),
                    # Spin about the vertical: the component the table's
                    # spinning friction is there to remove, kept per frame so a
                    # ball turning on the spot cannot go unnoticed.
                    "spin_z": ang[2],
                    "on_table": pos[2] > -0.5 * radius,
                }
                spin_now[name] = math.sqrt(sum(v ** 2 for v in ang))
                edge = off_table(pos, radius)
                if edge is not None and left_table is None:
                    left_table = {"ball": name, "frame": frame_index, "edge": edge,
                                  "location": list(pos)}

            # "Settled" means stopped moving *and* stopped turning. Speed alone
            # is not enough: a marble can sit at 0.000 m/s and go on spinning
            # about the vertical axis indefinitely.
            a_still = (record["ball_a"]["speed"] < SETTLE_SPEED
                       and spin_now["ball_a"] < SETTLE_SPIN)
            if a_still:
                settle_a = frame_index if settle_a is None else settle_a
            else:
                settle_a = None
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
                "approach" if hit["frame"] is None or frame_index < hit["frame"]
                else "after_impact"
            )
            frames.append(record)

        last = frames[-1]
        residual_spin = {
            name: math.sqrt(sum(v ** 2 for v in last[name]["angular_velocity"]))
            for name in ("ball_a", "ball_b") if name in last
        }
        approach_run = None
        if hit["frame"] is not None:
            contact = frames[hit["frame"] - 1]["ball_a"]["location"]
            approach_run = math.dist((ball_a_start[0], ball_a_start[1]),
                                     (contact[0], contact[1]))
        travel = {}
        for name, start in (("ball_a", ball_a_start), ("ball_b", ball_b_start)):
            if name not in last:
                continue
            travel[name] = math.dist((last[name]["location"][0], last[name]["location"][1]),
                                     (start[0], start[1]))

        prediction = predict_impact(m_a, m_b, e_eff, obliquity_deg,
                                    1.0 if ball_b_start[1] >= ball_a_start[1] else -1.0)
        measured_b = (None if hit["b_speed_out"] is None or not hit["a_speed_in"]
                      else hit["b_speed_out"] / hit["a_speed_in"])
        measured_a = (None if hit["a_speed_out"] is None or not hit["a_speed_in"]
                      else hit["a_speed_out"] / hit["a_speed_in"])

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
                    "initial_location": list(ball_a_start),
                    "initial_linear_velocity": list(velocity),
                    "initial_angular_velocity": list(spin),
                    "restitution": float(args.ball_a_restitution),
                    "friction": float(args.ball_a_friction),
                },
                "ball_b": {
                    "radius": r_b, "mass": m_b, "diameter": 2.0 * r_b,
                    "initial_location": list(ball_b_start),
                    "restitution": float(args.ball_b_restitution),
                    "friction": float(args.ball_b_friction),
                    "present": ball_b is not None,
                },
                "table": {
                    "footprint_x": list(TABLE_X),
                    "footprint_y": list(TABLE_Y),
                    "plate_thickness": TABLE_PLATE,
                    "friction": float(args.table_friction),
                    "rolling_friction": float(args.table_rolling_friction),
                    "spinning_friction": float(args.table_spinning_friction),
                    "restitution": float(args.table_restitution),
                },
                "floor_z": FLOOR_Z,
            },
            "collision": {
                "mass_ratio": m_a / m_b,
                "size_ratio": r_a / r_b,
                "effective_restitution": e_eff,
                "impact_parameter": impact_parameter,
                "centre_separation": separation,
                "obliquity_deg": obliquity_deg,
                "predicted": prediction,
                "measured_b_speed_ratio": measured_b,
                "measured_a_speed_ratio": measured_a,
            },
            "quality": {
                "hit_ball_b": hit["frame"] is not None,
                "contact_frame": hit["frame"],
                "contact_time": hit["time"],
                "contact_point": hit["point"],
                "contact_normal": hit["normal"],
                "approach_run": approach_run,
                "a_speed_in": hit["a_speed_in"],
                "a_speed_out": hit["a_speed_out"],
                "b_speed_out": hit["b_speed_out"],
                "a_heading_in_deg": (None if hit["a_vel_in"] is None
                                     else heading_deg(hit["a_vel_in"])),
                "a_turn_deg": (None if hit["a_vel_out"] is None
                               else signed_turn(hit["a_vel_in"], hit["a_vel_out"])),
                "b_heading_deg": (None if hit["b_vel_out"] is None
                                  else heading_deg(hit["b_vel_out"])),
                "b_turn_from_a_deg": (None if hit["b_vel_out"] is None
                                      else signed_turn(hit["a_vel_in"], hit["b_vel_out"])),
                "separation_deg": (
                    None if hit["a_vel_out"] is None or hit["b_vel_out"] is None
                    else abs(signed_turn(hit["a_vel_out"], hit["b_vel_out"]))),
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
                "left_table": left_table,
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
    c = records["collision"]

    def fmt(value, spec="%.3f"):
        return "None" if value is None else spec % value

    pred = c["predicted"]
    print(
        "[SIM] hit=%s f=%s run=%s | a_in=%s a_out=%s b_out=%s | "
        "ratio b=%s/%s a=%s/%s | obliq=%s sep=%s/%s | travel a=%s b=%s | "
        "A_end=(%.3f, %.3f) B_end=%s settle=(%s, %s) spin_left=%s "
        "dressing=%s off_table=%s" % (
            q["hit_ball_b"], q["contact_frame"], fmt(q["approach_run"]),
            fmt(q["a_speed_in"]), fmt(q["a_speed_out"]), fmt(q["b_speed_out"]),
            fmt(c["measured_b_speed_ratio"]), fmt(pred["b_speed_ratio"]),
            fmt(c["measured_a_speed_ratio"]), fmt(pred["a_speed_ratio"]),
            fmt(c["obliquity_deg"], "%.1f"), fmt(q["separation_deg"], "%.1f"),
            fmt(pred["separation_deg"], "%.1f"),
            fmt(q["ball_a_travel"]), fmt(q["ball_b_travel"]),
            q["ball_a_final"][0], q["ball_a_final"][1],
            None if q["ball_b_final"] is None else
            "(%.3f, %.3f)" % (q["ball_b_final"][0], q["ball_b_final"][1]),
            q["ball_a_settled_frame"], q["ball_b_settled_frame"],
            "/".join("%.2f" % v for v in q["residual_spin"].values()),
            q["prop_contact_frame"],
            None if q["left_table"] is None else
            f"{q['left_table']['ball']}@{q['left_table']['frame']}"
            f" {q['left_table']['edge']}",
        )
    )


if __name__ == "__main__":
    main()
