from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pybullet as p


# A picnic scene: an apple falls from a low branch overhead and lands an
# off-center hit on the top-side of a soccer ball resting on the grass next
# to a spread-out picnic blanket. The oblique hit (the apple's start XY is
# offset from the ball's center, not dead-center-above it) is what makes the
# ball roll away afterward instead of just being squashed straight down --
# a purely vertical, centered hit has no lever arm to torque the ball into
# rotation. Real-world sizes: FIFA size-5 ball, 0.22 m diameter; a medium
# apple, ~0.075 m diameter.
BALL_RADIUS = 0.11
APPLE_RADIUS = 0.0375

GROUND_Z = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-sec", type=float, default=4.5)
    parser.add_argument("--substeps", type=int, default=60)
    parser.add_argument("--ball-mass", type=float, default=0.43)
    parser.add_argument("--ball-friction", type=float, default=0.25)
    parser.add_argument("--ball-restitution", type=float, default=0.35)
    parser.add_argument("--ball-rolling-friction", type=float, default=0.015)
    parser.add_argument("--ball-spinning-friction", type=float, default=0.01)
    parser.add_argument("--apple-mass", type=float, default=0.15)
    parser.add_argument("--apple-friction", type=float, default=0.5)
    parser.add_argument("--apple-restitution", type=float, default=0.25)
    parser.add_argument("--grass-friction", type=float, default=0.25)
    parser.add_argument("--ball-location", nargs=2, type=float, default=(0.0, 0.0))
    parser.add_argument(
        "--apple-offset-x",
        type=float,
        default=0.105,
        help="Apple's start-XY offset from the ball's center along +X (the "
        "lever arm that turns the vertical drop into a rolling hit). Contact "
        "point on a sphere always lies on the center-to-center line, so a "
        "force through the ball's center can impart no spin; this offset is "
        "what makes the hit oblique enough for the ball to actually roll "
        "instead of just being pressed straight down into the grass.",
    )
    parser.add_argument("--apple-offset-y", type=float, default=0.0)
    parser.add_argument("--drop-height", type=float, default=1.3)
    parser.add_argument("--apple-drift-x", type=float, default=0.0)
    parser.add_argument("--apple-drift-y", type=float, default=0.0)
    parser.add_argument("--gravity-z", type=float, default=-9.8)
    # Per-object active flags for PCVE DELETE edits. A 0 removes that body
    # from the simulation entirely; its frame slot still appears in the
    # output (frozen at its start pose, present=false) so downstream
    # consumers keep a fixed two-object layout.
    parser.add_argument("--apple-active", type=int, default=1)
    parser.add_argument("--ball-active", type=int, default=1)
    return parser.parse_args()


def simulate(args: argparse.Namespace) -> dict:
    fps = int(args.fps)
    frame_end = max(2, int(round(float(args.duration_sec) * fps)))
    substeps = int(args.substeps)
    dt = 1.0 / float(fps * substeps)

    ball_x, ball_y = float(args.ball_location[0]), float(args.ball_location[1])
    ball_start = (ball_x, ball_y, GROUND_Z + BALL_RADIUS)

    apple_start = (
        ball_x + float(args.apple_offset_x),
        ball_y + float(args.apple_offset_y),
        GROUND_Z + BALL_RADIUS * 2.0 + APPLE_RADIUS + float(args.drop_height),
    )
    apple_initial_velocity = (float(args.apple_drift_x), float(args.apple_drift_y), 0.0)

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

        ground_shape = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=client)
        ground_id = p.createMultiBody(0.0, ground_shape, -1, (0.0, 0.0, GROUND_Z), physicsClientId=client)
        p.changeDynamics(
            ground_id,
            -1,
            lateralFriction=float(args.grass_friction),
            restitution=0.15,
            physicsClientId=client,
        )

        ball_active = bool(int(args.ball_active))
        apple_active = bool(int(args.apple_active))

        ball_id = None
        if ball_active:
            ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=BALL_RADIUS, physicsClientId=client)
            ball_id = p.createMultiBody(
                baseMass=float(args.ball_mass),
                baseCollisionShapeIndex=ball_shape,
                baseVisualShapeIndex=-1,
                basePosition=ball_start,
                physicsClientId=client,
            )
            p.changeDynamics(
                ball_id,
                -1,
                lateralFriction=float(args.ball_friction),
                spinningFriction=float(args.ball_spinning_friction),
                rollingFriction=float(args.ball_rolling_friction),
                restitution=float(args.ball_restitution),
                linearDamping=0.0,
                angularDamping=0.02,
                collisionMargin=0.0005,
                physicsClientId=client,
            )

        apple_id = None
        if apple_active:
            apple_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=APPLE_RADIUS, physicsClientId=client)
            apple_id = p.createMultiBody(
                baseMass=float(args.apple_mass),
                baseCollisionShapeIndex=apple_shape,
                baseVisualShapeIndex=-1,
                basePosition=apple_start,
                physicsClientId=client,
            )
            p.resetBaseVelocity(
                apple_id, linearVelocity=apple_initial_velocity, angularVelocity=(0.0, 0.0, 0.0),
                physicsClientId=client,
            )
            p.changeDynamics(
                apple_id,
                -1,
                lateralFriction=float(args.apple_friction),
                spinningFriction=0.01,
                rollingFriction=0.01,
                restitution=float(args.apple_restitution),
                linearDamping=0.0,
                angularDamping=0.05,
                collisionMargin=0.0005,
                physicsClientId=client,
            )

        frames = []
        ball_start_xy = (ball_x, ball_y)
        ball_roll_distance = 0.0
        min_apple_ball_gap = float("inf")

        identity_quat = [0.0, 0.0, 0.0, 1.0]
        zero_vec = [0.0, 0.0, 0.0]
        for frame_index in range(1, frame_end + 1):
            if frame_index > 1:
                for _ in range(substeps):
                    p.stepSimulation(physicsClientId=client)

            if ball_id is not None:
                ball_pos, ball_quat = p.getBasePositionAndOrientation(ball_id, physicsClientId=client)
                ball_lin, ball_ang = p.getBaseVelocity(ball_id, physicsClientId=client)
                ball_present = True
            else:
                ball_pos, ball_quat = ball_start, identity_quat
                ball_lin, ball_ang = zero_vec, zero_vec
                ball_present = False

            if apple_id is not None:
                apple_pos, apple_quat = p.getBasePositionAndOrientation(apple_id, physicsClientId=client)
                apple_lin, apple_ang = p.getBaseVelocity(apple_id, physicsClientId=client)
                apple_present = True
            else:
                apple_pos, apple_quat = apple_start, identity_quat
                apple_lin, apple_ang = zero_vec, zero_vec
                apple_present = False

            if ball_present and apple_present:
                gap = math.dist(ball_pos, apple_pos) - (BALL_RADIUS + APPLE_RADIUS)
                min_apple_ball_gap = min(min_apple_ball_gap, gap)
            if ball_present:
                ball_roll_distance = math.dist(ball_start_xy, (ball_pos[0], ball_pos[1]))

            frames.append({
                "frame_index": frame_index,
                "time_sec": (frame_index - 1) / float(fps),
                "soccer_ball": {
                    "present": ball_present,
                    "location": list(ball_pos),
                    "quaternion_xyzw": list(ball_quat),
                    "linear_velocity": list(ball_lin),
                    "angular_velocity": list(ball_ang),
                },
                "apple": {
                    "present": apple_present,
                    "location": list(apple_pos),
                    "quaternion_xyzw": list(apple_quat),
                    "linear_velocity": list(apple_lin),
                    "angular_velocity": list(apple_ang),
                },
            })

        final_ball_speed = math.sqrt(sum(v * v for v in frames[-1]["soccer_ball"]["linear_velocity"]))

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
                "soccer_ball": {
                    "radius": BALL_RADIUS,
                    "mass": float(args.ball_mass),
                    "initial_location": list(ball_start),
                    "friction": float(args.ball_friction),
                    "rolling_friction": float(args.ball_rolling_friction),
                    "restitution": float(args.ball_restitution),
                    "active": int(ball_active),
                },
                "apple": {
                    "radius": APPLE_RADIUS,
                    "mass": float(args.apple_mass),
                    "initial_location": list(apple_start),
                    "initial_linear_velocity": list(apple_initial_velocity),
                    "friction": float(args.apple_friction),
                    "restitution": float(args.apple_restitution),
                    "active": int(apple_active),
                },
                "ground": {
                    "friction": float(args.grass_friction),
                },
            },
            "quality": {
                "min_apple_ball_gap": min_apple_ball_gap,
                "ball_roll_distance": ball_roll_distance,
                "ball_final_speed": final_ball_speed,
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


if __name__ == "__main__":
    main()
