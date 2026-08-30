from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pybullet as p


# A ball is rolled across a playroom mat into the side of a wooden toy chest,
# rebounds off it, and on the way back knocks a second, stationary ball away.
#
# Three events, one push. Nothing after the launch is scripted: the ball is
# given a single initial velocity and a matching roll, and the rest is the
# chest's restitution and one ball-ball impulse.
#
# The chest is not a box in the simulation -- it is the room model's own mesh,
# exported to metres by render_ball_box_rebound.py and loaded here as a static
# concave triangle mesh. That matters more than it sounds. The chest's north
# side is not a flat slab: it stands on corner feet, its body begins 50 mm off
# the mat, and its outer surface is a framed panel whose recess sits ~16 mm
# behind the frame. Which of those two planes the ball meets depends on the
# ball's radius, because a rolling ball touches a vertical wall at its equator.
# A hand-fitted box collider would have to guess; the mesh does not. It also
# means the physics and the render cannot drift apart -- both read the same
# geometry through the same transform.
#
# The chest is also yawed about 3.7 deg relative to the room's axes, so the
# rebound is not a mirror about the world axes. The reflection is measured off
# the contact normal Bullet reports rather than assumed, and written into the
# ground truth.
#
# Coordinates are metric and match render_ball_box_rebound.py's world frame:
# z = 0 is the play mat's top surface, and the origin is on the chest's north
# face at the mid-point of its width.

GROUND_SPAN = 6.0            # half-width of the static floor slab
GROUND_DEPTH = 0.5           # how far the floor slab reaches below its top face

# The room's printed play mat is taken out of the set, so the balls roll on one
# uniform hardwood floor from launch to rest and there is no surface change
# anywhere in the shot. That is the point of taking it out: the only things that
# act on the ball are the floor's rolling resistance, the chest, and the other
# ball.
#
# The extent of the room the set is actually dressed for -- a corner, with a
# west wall, a south wall and two open sides. This is NOT the edge of the
# rendered floor: render_ball_box_rebound.py hides the model's own floorboard
# quad and lays in a 9 m tiling hardwood plane, so a ball past this line is
# still on solid, rendered floor. What it has left is the framed set, and at
# this camera it is on its way out of shot -- which is worth a note, not a
# failure.
FLOOR_X = (-1.892, 1.221)
FLOOR_Y = (-1.252, 1.348)

SETTLE_SPEED = 0.02          # m/s below which a ball counts as stopped
SETTLE_SPIN = 0.30           # rad/s below which a ball counts as stopped *turning*

# A ball at rest that is still turning is the failure this scene walked into
# once and must not walk into again, so the settle test checks both. See
# --floor-spinning-friction.
RESIDUAL_SPIN_LIMIT = 0.30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--chest-collider", type=Path, required=True,
        help="Metre-scale OBJ of the toy chest's own geometry, written by "
        "render_ball_box_rebound.py's --mode colliders pass.",
    )
    parser.add_argument(
        "--props-collider", type=Path, default=None,
        help="Metre-scale OBJ of the rest of the room's reachable static "
        "geometry -- the open lid, the toy basket, the teddy bear. Kept as a "
        "separate body from the chest so that 'the ball bounced off the chest' "
        "and 'the ball fetched up against the toy basket' cannot be confused "
        "for each other in the metrics; the first is the scene, the second is a "
        "failed take.",
    )
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-sec", type=float, default=3.0)
    parser.add_argument("--substeps", type=int, default=60)
    parser.add_argument("--gravity-z", type=float, default=-9.8)

    # --- the rolling ball ---
    parser.add_argument("--ball-a-radius", type=float, default=0.080)
    parser.add_argument("--ball-a-mass", type=float, default=0.120)
    parser.add_argument(
        "--ball-a-restitution", type=float, default=0.88,
        help="Bullet multiplies the two bodies' restitutions, so what the "
        "rebound actually sees is this times --chest-restitution. Both are kept "
        "separate because only one of them is a property of the chest.",
    )
    parser.add_argument("--ball-a-friction", type=float, default=1.0)
    parser.add_argument(
        "--ball-spinning-friction", type=float, default=0.0,
        help="Kept at 0 for the same reason as --ball-rolling-friction, and it "
        "is the same trap: Bullet combines spinning friction as "
        "sf_a * lateral_b + sf_b * lateral_a, so setting the *ball* to 1.0 -- "
        "the obvious way to let the surface's value pass through -- multiplies "
        "in the floor's lateral friction instead and yields an effective 0.6. "
        "That is enough to bleed the ball's linear speed as well as its spin: "
        "it arrived at the target ball at 0.78 m/s instead of 1.10 and moved it "
        "0.06 m instead of 0.30. With the ball at 0 and lateral friction 1.0 the "
        "effective value is the floor's own.",
    )
    parser.add_argument(
        "--ball-rolling-friction", type=float, default=0.0,
        help="Kept at 0 on purpose, so all rolling resistance comes from the "
        "surface. Bullet does not combine rolling friction as a plain product: "
        "it uses rf_a * lateral_b + rf_b * lateral_a, so giving the ball rf = 1 "
        "multiplies in the *mat's lateral* friction instead and turns the roll "
        "into a skid. With the ball at rf = 0 and lateral friction 1.0 the "
        "effective rolling resistance is exactly the mat's own value.",
    )

    parser.add_argument(
        "--ball-a-rolling-friction", type=float, default=None,
        help="Rolling resistance carried by the rolling ball itself, rather "
        "than by the floor. Defaults to --ball-rolling-friction. Note the "
        "combination rule above: this is multiplied by the *floor's lateral* "
        "friction, so an effective resistance of 0.030 (against the floor's own "
        "0.006 baseline) wants roughly 0.04 here, not 0.030.",
    )
    parser.add_argument(
        "--ball-b-rolling-friction", type=float, default=None,
        help="Same, for the target ball. Defaults to --ball-rolling-friction.",
    )

    # --- the target ball ---
    parser.add_argument("--ball-b-radius", type=float, default=0.065)
    parser.add_argument("--ball-b-mass", type=float, default=0.090)
    parser.add_argument("--ball-b-restitution", type=float, default=0.80)
    parser.add_argument("--ball-b-friction", type=float, default=1.0)
    parser.add_argument("--ball-b-x", type=float, default=-0.340)
    parser.add_argument("--ball-b-y", type=float, default=0.330)
    parser.add_argument(
        "--disable-ball-b", action="store_true",
        help="Leave the target ball out entirely. Used when aiming the shot: "
        "the rebound line has to be measured before there is anywhere sensible "
        "to put the ball it is supposed to hit.",
    )

    # --- surfaces ---
    parser.add_argument(
        "--floor-rolling-friction", type=float, default=0.0060,
        help="Rolling resistance of the hardwood floor, and the scene's only "
        "surface. This is the shot's clock: it sets how much speed the ball "
        "still carries into the chest and how far both balls run afterwards. "
        "The room is 3.1 x 2.6 m, so it has to be low enough that the ball "
        "makes it back off the chest and high enough that neither ball runs "
        "out past the boards.",
    )
    parser.add_argument("--floor-friction", type=float, default=0.60)
    parser.add_argument("--floor-restitution", type=float, default=0.45)
    parser.add_argument(
        "--floor-spinning-friction", type=float, default=0.008,
        help="Resistance to a ball spinning about the vertical axis, i.e. to it "
        "turning on the spot like a top. This is not a refinement -- with it at "
        "zero the scene is visibly wrong. Bouncing off the chest at an angle "
        "gives the ball a real 10.5 rad/s of spin about the vertical (the "
        "chest's surface friction acts tangentially, and that is what tangential "
        "friction does), and nothing else in Bullet opposes it: rolling friction "
        "resists rolling, and lateral friction resists sliding. The ball "
        "therefore came to a dead stop on the floor and went on turning at 7.7 "
        "rad/s, forever. At 0.008 -- an 8 mm contact patch, the same order as the "
        "rolling-friction figure and just as plausible for a soft rubber ball "
        "-- the spin bleeds away over about a third of a second, which is what "
        "a ball on boards actually does. It costs nothing elsewhere: spinning "
        "friction acts only about the contact normal, so it leaves both the "
        "roll and the linear speed untouched to the millimetre.",
    )

    parser.add_argument(
        "--chest-restitution", type=float, default=0.80,
        help="The parameter this scene is built around. Multiplied by the "
        "ball's own restitution it gives the fraction of the *normal* speed "
        "component that survives the bounce -- so it changes the rebound's "
        "direction as well as its speed, because the tangential component is "
        "untouched. Lower values send the ball off flatter along the chest and "
        "past the target ball entirely.",
    )
    parser.add_argument("--chest-friction", type=float, default=0.55)

    # --- the push ---
    parser.add_argument("--launch-x", type=float, default=0.750)
    parser.add_argument("--launch-y", type=float, default=0.550)
    parser.add_argument(
        "--launch-speed", type=float, default=1.95,
        help="Initial speed in m/s (a firm hand roll).",
    )
    parser.add_argument(
        "--launch-heading-deg", type=float, default=215.3,
        help="Direction of the push, degrees CCW from +X. The default aims the "
        "ball at the chest's north panel a little east of its centre, which is "
        "the impact point that leaves the most clear mat on the rebound side.",
    )
    return parser.parse_args()


def add_ground(client: int, args: argparse.Namespace) -> int:
    """The hardwood floor, with its top face at z = 0."""
    shape = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[GROUND_SPAN, GROUND_SPAN, GROUND_DEPTH / 2.0],
        physicsClientId=client,
    )
    body = p.createMultiBody(0.0, shape, -1, [0.0, 0.0, -GROUND_DEPTH / 2.0],
                             physicsClientId=client)
    p.changeDynamics(
        body, -1,
        lateralFriction=float(args.floor_friction),
        rollingFriction=float(args.floor_rolling_friction),
        spinningFriction=float(args.floor_spinning_friction),
        restitution=float(args.floor_restitution),
        collisionMargin=0.0005, physicsClientId=client,
    )
    return body


def add_room(client: int, obj_path: Path, friction: float, restitution: float) -> int:
    if not obj_path.exists():
        raise FileNotFoundError(
            f"Room collision mesh not found: {obj_path}\n"
            "Generate it first with:\n"
            "  blender -b --python scripts/ball_box_rebound/render_ball_box_rebound.py"
            " -- --mode colliders --out-dir /tmp/colliders"
        )
    shape = p.createCollisionShape(
        p.GEOM_MESH, fileName=str(obj_path), meshScale=[1.0, 1.0, 1.0],
        flags=p.GEOM_FORCE_CONCAVE_TRIMESH, physicsClientId=client,
    )
    body = p.createMultiBody(0.0, shape, -1, [0.0, 0.0, 0.0], physicsClientId=client)
    p.changeDynamics(
        body, -1,
        lateralFriction=friction, rollingFriction=0.0, spinningFriction=0.0,
        restitution=restitution, collisionMargin=0.0005, physicsClientId=client,
    )
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


def angle_from_normal(velocity, normal) -> float:
    """Angle in degrees between a horizontal velocity and a wall normal.

    Measured the way a reflection is usually quoted: 0 deg is straight into (or
    straight back out of) the wall, 90 deg is parallel to it.
    """
    vx, vy = velocity[0], velocity[1]
    nx, ny = normal[0], normal[1]
    vn = math.hypot(vx, vy)
    nn = math.hypot(nx, ny)
    if vn < 1e-9 or nn < 1e-9:
        return float("nan")
    cos = abs((vx * nx + vy * ny) / (vn * nn))
    return math.degrees(math.acos(min(1.0, cos)))


def simulate(args: argparse.Namespace) -> dict:
    fps = int(args.fps)
    frame_end = max(2, int(round(float(args.duration_sec) * fps)))
    substeps = int(args.substeps)
    dt = 1.0 / float(fps * substeps)

    r_a = float(args.ball_a_radius)
    r_b = float(args.ball_b_radius)
    heading = math.radians(float(args.launch_heading_deg))
    speed = float(args.launch_speed)
    velocity = (speed * math.cos(heading), speed * math.sin(heading), 0.0)

    # Launch the ball already rolling rather than sliding. For a contact
    # directly underneath, rolling without slipping means omega = (z_hat x v)/r;
    # launched with zero spin the ball spends its first tenth of a second
    # skidding while friction spins it up, which reads as a shove, not a roll.
    spin = (-velocity[1] / r_a, velocity[0] / r_a, 0.0)

    ball_a_start = (float(args.launch_x), float(args.launch_y), r_a)
    ball_b_start = (float(args.ball_b_x), float(args.ball_b_y), r_b)

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

        add_ground(client, args)
        chest_id = add_room(client, args.chest_collider, float(args.chest_friction),
                            float(args.chest_restitution))
        props_id = None
        if args.props_collider is not None:
            props_id = add_room(client, args.props_collider, 0.75, 0.15)
        # Per-ball rolling resistance, falling back to the shared knob so the
        # existing defaults and every existing caller are unchanged.
        shared_rf = float(args.ball_rolling_friction)
        rf_a = shared_rf if args.ball_a_rolling_friction is None else float(args.ball_a_rolling_friction)
        rf_b = shared_rf if args.ball_b_rolling_friction is None else float(args.ball_b_rolling_friction)

        ball_a = add_ball(client, r_a, float(args.ball_a_mass), ball_a_start,
                          float(args.ball_a_friction), float(args.ball_a_restitution),
                          rf_a,
                          float(args.ball_spinning_friction))
        ball_b = None
        if not args.disable_ball_b:
            ball_b = add_ball(client, r_b, float(args.ball_b_mass), ball_b_start,
                              float(args.ball_b_friction), float(args.ball_b_restitution),
                              rf_b,
                              float(args.ball_spinning_friction))

        p.resetBaseVelocity(ball_a, linearVelocity=velocity, angularVelocity=spin,
                            physicsClientId=client)

        # Both impacts are resolved inside a single substep, so speeds are
        # sampled at substep resolution rather than at frame boundaries -- at
        # 24 fps a frame is long enough for the mat to eat a measurable slice of
        # the rebound before anything gets recorded.
        wall = {"frame": None, "time": None, "normal": None,
                "speed_in": None, "speed_out": None,
                "incidence_deg": None, "reflection_deg": None, "point": None}
        hit = {"frame": None, "time": None,
               "a_speed_in": None, "a_speed_out": None, "b_speed_out": None,
               "point": None, "b_heading_deg": None}
        prev_a_vel = velocity
        wall_touching = False
        hit_touching = False
        prop_contact_frame = None

        frames = []
        settle_a = None
        settle_b = None
        left_floor = None

        for frame_index in range(1, frame_end + 1):
            for _ in range(substeps if frame_index > 1 else 0):
                p.stepSimulation(physicsClientId=client)
                now = (frame_index - 1) / float(fps)

                a_vel, _ = p.getBaseVelocity(ball_a, physicsClientId=client)

                # Only the *first* touch on the chest is the scene's bounce. The
                # ball can and does come back to the chest later in the softer
                # scenarios, and an earlier version of this loop let the second
                # touch overwrite the first, which quietly reported a 1.2 m/s
                # rebound out of a 0.8 m/s approach.
                wall_points = p.getContactPoints(bodyA=ball_a, bodyB=chest_id,
                                                 physicsClientId=client)
                if wall_points and not wall_touching:
                    wall_touching = True
                    if wall["frame"] is None:
                        cp = max(wall_points, key=lambda c: c[9])
                        wall["frame"] = frame_index
                        wall["time"] = now
                        wall["point"] = list(cp[5])
                        # contactNormalOnB points from the chest toward the ball.
                        wall["normal"] = list(cp[7])
                        wall["speed_in"] = horiz(prev_a_vel)
                        wall["incidence_deg"] = angle_from_normal(prev_a_vel, cp[7])
                elif not wall_points and wall_touching:
                    wall_touching = False
                    if wall["speed_out"] is None:
                        wall["speed_out"] = horiz(a_vel)
                        wall["reflection_deg"] = angle_from_normal(a_vel, wall["normal"])

                if props_id is not None and prop_contact_frame is None:
                    for body in (ball_a, ball_b):
                        if body is None:
                            continue
                        if p.getContactPoints(bodyA=body, bodyB=props_id,
                                              physicsClientId=client):
                            prop_contact_frame = frame_index
                            break

                if ball_b is not None:
                    ball_points = p.getContactPoints(bodyA=ball_a, bodyB=ball_b,
                                                     physicsClientId=client)
                    if ball_points and not hit_touching:
                        hit_touching = True
                        if hit["frame"] is None:
                            hit["frame"] = frame_index
                            hit["time"] = now
                            hit["point"] = list(ball_points[0][5])
                            hit["a_speed_in"] = horiz(prev_a_vel)
                    elif not ball_points and hit_touching:
                        hit_touching = False
                        if hit["a_speed_out"] is None:
                            b_vel, _ = p.getBaseVelocity(ball_b, physicsClientId=client)
                            hit["a_speed_out"] = horiz(a_vel)
                            hit["b_speed_out"] = horiz(b_vel)
                            hit["b_heading_deg"] = math.degrees(
                                math.atan2(b_vel[1], b_vel[0])) % 360.0

                prev_a_vel = a_vel

            record = {"frame_index": frame_index,
                      "time_sec": (frame_index - 1) / float(fps)}
            spin_now = {}
            for name, body in (("ball_a", ball_a), ("ball_b", ball_b)):
                if body is None:
                    continue
                pos, quat = p.getBasePositionAndOrientation(body, physicsClientId=client)
                lin, ang = p.getBaseVelocity(body, physicsClientId=client)
                record[name] = {
                    "location": list(pos),
                    "quaternion_xyzw": list(quat),
                    "linear_velocity": list(lin),
                    "angular_velocity": list(ang),
                    "speed": math.sqrt(lin[0] ** 2 + lin[1] ** 2 + lin[2] ** 2),
                    # Spin about the vertical: the component the floor's
                    # spinning friction is there to remove, kept per-frame so a
                    # ball turning on the spot cannot go unnoticed again.
                    "spin_z": ang[2],
                }
                spin_now[name] = math.sqrt(ang[0] ** 2 + ang[1] ** 2 + ang[2] ** 2)
                if not (FLOOR_X[0] <= pos[0] <= FLOOR_X[1]
                        and FLOOR_Y[0] <= pos[1] <= FLOOR_Y[1]) and left_floor is None:
                    left_floor = {"ball": name, "frame": frame_index,
                                  "location": list(pos)}

            # "Settled" means stopped moving *and* stopped turning. Speed alone
            # is not enough: a ball can sit at 0.000 m/s and go on spinning
            # about the vertical axis indefinitely, which is exactly what this
            # scene did before the floor was given any spinning friction.
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
                    hit["frame"] is not None and frame_index >= hit["frame"]
                )

            record["ball_a"]["phase"] = (
                "approach" if wall["frame"] is None or frame_index < wall["frame"]
                else "rebound" if hit["frame"] is None or frame_index < hit["frame"]
                else "after_impact"
            )
            frames.append(record)

        last = frames[-1]
        residual_spin = {
            name: math.sqrt(sum(v ** 2 for v in last[name]["angular_velocity"]))
            for name in ("ball_a", "ball_b") if name in last
        }
        b_travel = 0.0 if ball_b is None else math.dist(
            (last["ball_b"]["location"][0], last["ball_b"]["location"][1]),
            (ball_b_start[0], ball_b_start[1]),
        )
        rebound_run = None
        if wall["frame"] is not None and hit["frame"] is not None:
            wf = frames[wall["frame"] - 1]["ball_a"]["location"]
            hf = frames[hit["frame"] - 1]["ball_a"]["location"]
            rebound_run = math.dist((wf[0], wf[1]), (hf[0], hf[1]))

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
                    "radius": r_a,
                    "mass": float(args.ball_a_mass),
                    "initial_location": list(ball_a_start),
                    "initial_linear_velocity": list(velocity),
                    "initial_angular_velocity": list(spin),
                    "restitution": float(args.ball_a_restitution),
                    "friction": float(args.ball_a_friction),
                },
                "ball_b": {
                    "radius": r_b,
                    "mass": float(args.ball_b_mass),
                    "initial_location": list(ball_b_start),
                    "restitution": float(args.ball_b_restitution),
                    "friction": float(args.ball_b_friction),
                    "rolling_friction": rf_b,
                },
                "chest": {
                    "collider": str(args.chest_collider),
                    "restitution": float(args.chest_restitution),
                    "friction": float(args.chest_friction),
                    "effective_restitution": float(args.chest_restitution)
                                             * float(args.ball_a_restitution),
                },
                "floor": {
                    "friction": float(args.floor_friction),
                    "rolling_friction": float(args.floor_rolling_friction),
                    "restitution": float(args.floor_restitution),
                    "footprint_x": list(FLOOR_X),
                    "footprint_y": list(FLOOR_Y),
                },
            },
            "quality": {
                "hit_chest": wall["frame"] is not None,
                "chest_contact_frame": wall["frame"],
                "chest_contact_time": wall["time"],
                "chest_contact_point": wall["point"],
                "chest_contact_normal": wall["normal"],
                "approach_speed": wall["speed_in"],
                "rebound_speed": wall["speed_out"],
                "incidence_deg": wall["incidence_deg"],
                "reflection_deg": wall["reflection_deg"],
                "rebound_run": rebound_run,
                "hit_ball_b": hit["frame"] is not None,
                "ball_contact_frame": hit["frame"],
                "ball_contact_time": hit["time"],
                "ball_contact_point": hit["point"],
                "ball_a_speed_into_b": hit["a_speed_in"],
                "ball_a_speed_after_b": hit["a_speed_out"],
                "ball_b_speed_after": hit["b_speed_out"],
                "ball_b_heading_deg": hit["b_heading_deg"],
                "ball_b_travel": b_travel,
                "ball_a_final": last["ball_a"]["location"],
                "ball_b_final": None if ball_b is None else last["ball_b"]["location"],
                "ball_a_settled_frame": settle_a,
                "ball_b_settled_frame": settle_b,
                # Both should be at solver noise. Anything near the launch spin
                # means a ball is still turning after it has stopped moving.
                "residual_spin": residual_spin,
                "spinning_on_the_spot": any(v > RESIDUAL_SPIN_LIMIT
                                            for v in residual_spin.values()),
                "hit_room_props": prop_contact_frame is not None,
                "prop_contact_frame": prop_contact_frame,
                "left_floor": left_floor,
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

    def fmt(value, spec="%.3f"):
        return "None" if value is None else spec % value

    print(
        "[SIM] chest=%s f=%s in=%s out=%s inc=%s refl=%s | run=%s | "
        "ballB=%s f=%s a_in=%s a_out=%s b_out=%s b_travel=%s | "
        "A_end=(%.3f, %.3f) B_end=%s settle=(%s, %s) spin_left=%s props=%s off_floor=%s" % (
            q["hit_chest"], q["chest_contact_frame"], fmt(q["approach_speed"]),
            fmt(q["rebound_speed"]), fmt(q["incidence_deg"], "%.1f"),
            fmt(q["reflection_deg"], "%.1f"), fmt(q["rebound_run"]),
            q["hit_ball_b"], q["ball_contact_frame"], fmt(q["ball_a_speed_into_b"]),
            fmt(q["ball_a_speed_after_b"]), fmt(q["ball_b_speed_after"]),
            fmt(q["ball_b_travel"]),
            q["ball_a_final"][0], q["ball_a_final"][1],
            None if q["ball_b_final"] is None else
            "(%.3f, %.3f)" % (q["ball_b_final"][0], q["ball_b_final"][1]),
            q["ball_a_settled_frame"], q["ball_b_settled_frame"],
            "/".join("%.2f" % v for v in q["residual_spin"].values()),
            q["prop_contact_frame"],
            None if q["left_floor"] is None else
            f"{q['left_floor']['ball']}@{q['left_floor']['frame']}",
        )
    )


if __name__ == "__main__":
    main()
