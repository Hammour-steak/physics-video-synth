from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pybullet as p


# A toy ball is rolled across a living room's bare floor onto a square area rug
# and is brought to a stop by the rug's much higher rolling resistance.
#
# The whole scene is one continuous roll -- there is no thrower, no scripted
# stop, and no keyframed slowdown. Exactly one thing changes under the ball:
# rolling resistance. The floor's is very low (the ball barely loses speed
# crossing it) and the rug's is roughly twenty times higher, and that ratio
# alone is what stops the ball.
#
# The rug is flat: carpet_thickness defaults to 0, so its surface is coplanar
# with the bare floor and the ball's path has no step in it at all. It rolls
# across one continuous flat plane and simply starts shedding speed the moment
# it crosses the rug's boundary. This is deliberate -- it isolates friction as
# the only variable. (The parameter is still live: a positive carpet_thickness
# raises the rug into a real step the ball has to climb, which costs it speed
# on its own and muddies that isolation.)
#
# Because the rug has to be a *coplanar* friction patch rather than a slab
# sitting on top of something, the bare floor cannot be one infinite plane: a
# plane under the rug would keep taking the ball's contacts and the rug's
# friction would never apply. The floor is therefore tiled as four boxes that
# surround the rug's footprint without overlapping it, all with their top faces
# at exactly z = 0, so the ball crosses from tile to rug on a continuous
# surface and only the surface properties change.
#
# Coordinates are metric and match render_ball_carpet_climb.py's world frame:
# the rug is centred on the origin, its top face at z = carpet_thickness, and
# the bare floor is z = 0.

BALL_RADIUS = 0.105          # FIVB volleyball, 0.21 m diameter
CARPET_HALF = 0.7657         # half-width of the room model's square rug
FLOOR_Z = 0.0
FLOOR_SPAN = 4.0             # half-width of the tiled bare floor, comfortably past the room
SLAB_DEPTH = 0.5             # how far the static floor/rug boxes reach below their top face

# The living room's glass coffee table stands on the rug. Its four corner legs
# are simulated as static boxes so a ball launched harder than the hero case
# (the PCVE suite does exactly this) bumps into a leg the way it visibly would,
# instead of rolling straight through the rendered frame. The hero ball's lane
# passes west of the table and never touches them.
TABLE_LEG_X = (-0.380, 0.530)
TABLE_LEG_Y = (-0.520, 0.380)
TABLE_LEG_HALF = 0.016
TABLE_LEG_TOP = 0.40

# The sectional sofa's upholstered base, standing on the rug's far half. Only
# the low-friction rug cases in the PCVE suite send the ball this far, and
# without it those balls roll straight through the rendered sofa; with it they
# bump its front and come to rest against it, which is what would really
# happen. Measured off the room model's own furniture footprint.
SOFA_FRONT_Y = 0.47
SOFA_HALF_WIDTH = 1.70
SOFA_DEPTH = 1.05
SOFA_HEIGHT = 0.45


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-sec", type=float, default=2.0)
    parser.add_argument("--substeps", type=int, default=60)

    parser.add_argument("--ball-mass", type=float, default=0.27)
    parser.add_argument("--ball-friction", type=float, default=1.0)
    parser.add_argument("--ball-restitution", type=float, default=0.45)
    parser.add_argument(
        "--ball-rolling-friction",
        type=float,
        default=0.0,
        help="Kept at 0 on purpose, so that all rolling resistance comes from "
        "whichever surface the ball is on. Bullet does NOT combine rolling "
        "friction as a plain product: it uses rf_a * lateral_b + rf_b * "
        "lateral_a. Giving the ball rf=1.0 (the obvious way to make surface "
        "values pass straight through) therefore multiplies in the floor's "
        "*lateral* friction instead, which killed the ball's spin on the first "
        "contact and turned the whole roll into a skid. With the ball at rf=0 "
        "and lateral friction 1.0, the effective rolling resistance is exactly "
        "the surface's own rolling-friction value -- which is what lets the "
        "floor be slick and the rug grippy in one continuous roll.",
    )
    parser.add_argument("--ball-spinning-friction", type=float, default=0.0)

    parser.add_argument(
        "--floor-rolling-friction",
        type=float,
        default=0.0015,
        help="Rolling resistance of the bare floor (hard, smooth). Low enough "
        "that the ball crosses the approach almost without slowing.",
    )
    parser.add_argument("--floor-friction", type=float, default=0.45)
    parser.add_argument("--floor-restitution", type=float, default=0.7)

    parser.add_argument(
        "--carpet-rolling-friction",
        type=float,
        default=0.060,
        help="Rolling resistance of the rug -- the parameter this scene is "
        "built around, and with a flush rug the only thing that changes under "
        "the ball at all. Higher values stop it nearer the near edge; low "
        "enough values let it coast the full width into the sofa.",
    )
    parser.add_argument("--carpet-friction", type=float, default=0.9)
    parser.add_argument("--carpet-restitution", type=float, default=0.1)
    parser.add_argument(
        "--carpet-thickness",
        type=float,
        default=0.0,
        help="Height of the rug's top face above the bare floor, in metres. At "
        "the default 0 the rug is a flat, coplanar friction patch and the ball "
        "rolls over one continuous plane, so rolling resistance is the only "
        "thing that changes under it. A positive value turns the rug's boundary "
        "into a real step the ball has to climb, which costs it speed by itself. "
        "Must stay in sync with the rendered carpet slab.",
    )

    parser.add_argument(
        "--launch-x",
        type=float,
        default=-0.57,
        help="The ball's lane, held constant through the roll. Chosen to thread "
        "the ~0.39 m gap between the rug's west edge and the coffee table's legs.",
    )
    parser.add_argument("--launch-y", type=float, default=-1.35)
    parser.add_argument(
        "--launch-speed",
        type=float,
        default=2.05,
        help="Initial speed along +Y in m/s (a firm hand roll).",
    )
    parser.add_argument("--gravity-z", type=float, default=-9.8)
    return parser.parse_args()


def add_static_box(client: int, half_extents, position, friction: float,
                   restitution: float, rolling_friction: float) -> int:
    shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=list(half_extents), physicsClientId=client)
    body = p.createMultiBody(0.0, shape, -1, list(position), physicsClientId=client)
    p.changeDynamics(
        body, -1,
        lateralFriction=friction,
        rollingFriction=rolling_friction,
        spinningFriction=0.0,
        restitution=restitution,
        collisionMargin=0.0005,
        physicsClientId=client,
    )
    return body


def simulate(args: argparse.Namespace) -> dict:
    fps = int(args.fps)
    frame_end = max(2, int(round(float(args.duration_sec) * fps)))
    substeps = int(args.substeps)
    dt = 1.0 / float(fps * substeps)

    thickness = float(args.carpet_thickness)
    launch_x = float(args.launch_x)
    launch_y = float(args.launch_y)
    speed = float(args.launch_speed)
    ball_start = (launch_x, launch_y, FLOOR_Z + BALL_RADIUS)

    # Start the ball already rolling rather than sliding: for travel along +Y
    # with contact underneath, v = omega x r gives omega = (-v/r, 0, 0). A ball
    # launched with zero spin spends its first tenth of a second skidding while
    # friction spins it up, which reads as a shove rather than a roll.
    spin = (-speed / BALL_RADIUS, 0.0, 0.0)

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

        # The bare floor, tiled as four boxes around the rug's footprint rather
        # than laid down as one infinite plane. A plane would run underneath the
        # rug, and with a flat (zero-thickness) rug the ball would rest on both
        # at once -- in practice the plane wins the contact and the rug's
        # friction never applies at all, which is exactly the bug this tiling
        # avoids. Every tile's top face is at z = 0, so the ball never meets a
        # seam it can feel.
        outer = FLOOR_SPAN
        floor_tiles = (
            # (x_min, x_max, y_min, y_max)
            (-outer, outer, -outer, -CARPET_HALF),      # near side, the run-up
            (-outer, outer, CARPET_HALF, outer),        # far side, past the rug
            (-outer, -CARPET_HALF, -CARPET_HALF, CARPET_HALF),   # west of the rug
            (CARPET_HALF, outer, -CARPET_HALF, CARPET_HALF),     # east of the rug
        )
        for x_min, x_max, y_min, y_max in floor_tiles:
            add_static_box(
                client,
                ((x_max - x_min) / 2.0, (y_max - y_min) / 2.0, SLAB_DEPTH / 2.0),
                ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0, FLOOR_Z - SLAB_DEPTH / 2.0),
                float(args.floor_friction),
                float(args.floor_restitution),
                float(args.floor_rolling_friction),
            )

        # The rug, grown downward from its top face so that thickness = 0 is a
        # valid flush patch rather than a degenerate zero-height box.
        add_static_box(
            client,
            (CARPET_HALF, CARPET_HALF, (SLAB_DEPTH + thickness) / 2.0),
            (0.0, 0.0, FLOOR_Z + thickness - (SLAB_DEPTH + thickness) / 2.0),
            float(args.carpet_friction),
            float(args.carpet_restitution),
            float(args.carpet_rolling_friction),
        )

        leg_ids = []
        leg_half_height = (TABLE_LEG_TOP - thickness) / 2.0
        for lx in TABLE_LEG_X:
            for ly in TABLE_LEG_Y:
                leg_ids.append(add_static_box(
                    client,
                    (TABLE_LEG_HALF, TABLE_LEG_HALF, leg_half_height),
                    (lx, ly, thickness + leg_half_height),
                    0.6, 0.2, 0.001,
                ))

        add_static_box(
            client,
            (SOFA_HALF_WIDTH, SOFA_DEPTH / 2.0, SOFA_HEIGHT / 2.0),
            (0.0, SOFA_FRONT_Y + SOFA_DEPTH / 2.0, SOFA_HEIGHT / 2.0),
            0.7, 0.15, 0.05,
        )

        ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=BALL_RADIUS, physicsClientId=client)
        ball_id = p.createMultiBody(
            baseMass=float(args.ball_mass),
            baseCollisionShapeIndex=ball_shape,
            baseVisualShapeIndex=-1,
            basePosition=ball_start,
            physicsClientId=client,
        )
        p.resetBaseVelocity(
            ball_id,
            linearVelocity=(0.0, speed, 0.0),
            angularVelocity=spin,
            physicsClientId=client,
        )
        p.changeDynamics(
            ball_id, -1,
            lateralFriction=float(args.ball_friction),
            rollingFriction=float(args.ball_rolling_friction),
            spinningFriction=float(args.ball_spinning_friction),
            restitution=float(args.ball_restitution),
            linearDamping=0.0,
            angularDamping=0.0,
            collisionMargin=0.0005,
            physicsClientId=client,
        )

        frames = []
        on_carpet_frame = None
        edge_entry_speed = None
        edge_entry_xy = None
        max_lift_over_carpet = 0.0
        carpet_travel = 0.0
        leg_contact_frame = None
        settle_frame = None

        for frame_index in range(1, frame_end + 1):
            if frame_index > 1:
                for _ in range(substeps):
                    p.stepSimulation(physicsClientId=client)

            pos, quat = p.getBasePositionAndOrientation(ball_id, physicsClientId=client)
            lin, ang = p.getBaseVelocity(ball_id, physicsClientId=client)
            speed_now = math.sqrt(lin[0] ** 2 + lin[1] ** 2 + lin[2] ** 2)

            # For a flat rug this is just "the contact patch is inside the rug".
            # The height test only bites when carpet_thickness is positive, where
            # the ball can be over the footprint while still down on the floor
            # part-way up the step; the 2 mm slack keeps it from tripping on
            # solver jitter when the rug is flush.
            over_carpet = abs(pos[0]) <= CARPET_HALF and abs(pos[1]) <= CARPET_HALF
            resting_on_carpet = (
                over_carpet
                and pos[2] >= FLOOR_Z + BALL_RADIUS + thickness * 0.4 - 0.002
            )
            if resting_on_carpet and on_carpet_frame is None:
                on_carpet_frame = frame_index
                edge_entry_speed = speed_now
                edge_entry_xy = (pos[0], pos[1])
            if resting_on_carpet and edge_entry_xy is not None:
                max_lift_over_carpet = max(
                    max_lift_over_carpet, pos[2] - (FLOOR_Z + BALL_RADIUS + thickness),
                )
                carpet_travel = math.dist((pos[0], pos[1]), edge_entry_xy)

            # First frame the ball has effectively come to rest and stays there.
            if speed_now < 0.02:
                if settle_frame is None:
                    settle_frame = frame_index
            else:
                settle_frame = None

            if leg_contact_frame is None:
                for leg in leg_ids:
                    if p.getContactPoints(bodyA=ball_id, bodyB=leg, physicsClientId=client):
                        leg_contact_frame = frame_index
                        break

            frames.append({
                "frame_index": frame_index,
                "time_sec": (frame_index - 1) / float(fps),
                "ball": {
                    "location": list(pos),
                    "quaternion_xyzw": list(quat),
                    "linear_velocity": list(lin),
                    "angular_velocity": list(ang),
                    "speed": speed_now,
                    "on_carpet": bool(resting_on_carpet),
                },
            })

        final = frames[-1]["ball"]
        final_speed = final["speed"]

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
                "ball": {
                    "radius": BALL_RADIUS,
                    "mass": float(args.ball_mass),
                    "initial_location": list(ball_start),
                    "initial_linear_velocity": [0.0, speed, 0.0],
                    "initial_angular_velocity": list(spin),
                    "friction": float(args.ball_friction),
                    "rolling_friction": float(args.ball_rolling_friction),
                    "restitution": float(args.ball_restitution),
                },
                "carpet": {
                    "half_extent": CARPET_HALF,
                    "thickness": thickness,
                    "friction": float(args.carpet_friction),
                    "rolling_friction": float(args.carpet_rolling_friction),
                    "restitution": float(args.carpet_restitution),
                },
                "floor": {
                    "friction": float(args.floor_friction),
                    "rolling_friction": float(args.floor_rolling_friction),
                    "restitution": float(args.floor_restitution),
                },
                "table_legs": {
                    "x": list(TABLE_LEG_X),
                    "y": list(TABLE_LEG_Y),
                    "half_width": TABLE_LEG_HALF,
                    "top_z": TABLE_LEG_TOP,
                },
            },
            "quality": {
                "reached_carpet": on_carpet_frame is not None,
                "carpet_entry_frame": on_carpet_frame,
                "carpet_entry_speed": edge_entry_speed,
                "carpet_travel": carpet_travel,
                "carpet_step_height": thickness,
                # Doubles as a flatness check: with a flush rug this should stay
                # at solver noise (< 1 mm). Anything larger means the ball is
                # being kicked upward at the rug boundary, i.e. the surface the
                # ball actually rolls on is not flat after all.
                "max_lift_over_carpet": max_lift_over_carpet,
                "final_speed": final_speed,
                "final_location": final["location"],
                "settled_frame": settle_frame,
                "hit_table_leg": leg_contact_frame is not None,
                "table_leg_contact_frame": leg_contact_frame,
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
    print(
        "[SIM] reached=%s entry_frame=%s entry_speed=%s carpet_travel=%.3f "
        "max_lift=%.4f final=(%.3f, %.3f, %.3f) final_speed=%.4f "
        "settled_frame=%s hit_leg=%s" % (
            q["reached_carpet"], q["carpet_entry_frame"],
            ("%.3f" % q["carpet_entry_speed"]) if q["carpet_entry_speed"] else None,
            q["carpet_travel"], q["max_lift_over_carpet"],
            *q["final_location"], q["final_speed"],
            q["settled_frame"], q["hit_table_leg"],
        )
    )


if __name__ == "__main__":
    main()
