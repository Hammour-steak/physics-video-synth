from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pybullet as p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-sec", type=float, default=3.0)
    parser.add_argument("--substeps", type=int, default=12)
    parser.add_argument("--ball-radius", type=float, default=0.033)
    parser.add_argument("--ball-mass", type=float, default=0.057)
    parser.add_argument("--ball-friction", type=float, default=0.5)
    parser.add_argument("--ball-restitution", type=float, default=0.05)
    parser.add_argument("--ball-rolling-friction", type=float, default=0.015)
    parser.add_argument("--floor-friction", type=float, default=0.6)
    parser.add_argument("--launch-x", type=float, default=-4.0)
    parser.add_argument("--launch-y", type=float, default=0.0)
    parser.add_argument("--launch-z", type=float, default=1.5)
    parser.add_argument("--launch-vx", type=float, default=4.5)
    parser.add_argument("--launch-vy", type=float, default=0.0)
    parser.add_argument("--launch-vz", type=float, default=4.5)
    parser.add_argument("--gravity-z", type=float, default=-9.8)
    return parser.parse_args()


def simulate(args: argparse.Namespace) -> dict:
    fps = int(args.fps)
    frame_end = max(2, int(round(float(args.duration_sec) * fps)))
    substeps = int(args.substeps)
    dt = 1.0 / float(fps * substeps)
    radius = float(args.ball_radius)
    gravity_z = float(args.gravity_z)

    ball_initial_location = (
        float(args.launch_x),
        float(args.launch_y),
        float(args.launch_z),
    )
    ball_initial_velocity = (
        float(args.launch_vx),
        float(args.launch_vy),
        float(args.launch_vz),
    )

    client = p.connect(p.DIRECT)
    try:
        p.resetSimulation(physicsClientId=client)
        p.setGravity(0.0, 0.0, gravity_z, physicsClientId=client)
        p.setTimeStep(dt, physicsClientId=client)
        p.setPhysicsEngineParameter(
            fixedTimeStep=dt,
            numSolverIterations=500,
            contactBreakingThreshold=0.0002,
            deterministicOverlappingPairs=1,
            enableConeFriction=1,
            physicsClientId=client,
        )

        floor_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=(10.0, 10.0, 0.1),
            physicsClientId=client,
        )
        floor_id = p.createMultiBody(
            0.0,
            floor_shape,
            -1,
            (0.0, 0.0, -0.1),
            physicsClientId=client,
        )
        p.changeDynamics(
            floor_id,
            -1,
            lateralFriction=float(args.floor_friction),
            restitution=0.3,
            collisionMargin=0.001,
            physicsClientId=client,
        )

        ball_shape = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=radius,
            physicsClientId=client,
        )
        ball_id = p.createMultiBody(
            baseMass=float(args.ball_mass),
            baseCollisionShapeIndex=ball_shape,
            baseVisualShapeIndex=-1,
            basePosition=ball_initial_location,
            baseOrientation=(0.0, 0.0, 0.0, 1.0),
            baseInertialFramePosition=(0.0, 0.0, 0.0),
            baseInertialFrameOrientation=(0.0, 0.0, 0.0, 1.0),
            physicsClientId=client,
        )
        p.resetBaseVelocity(
            ball_id,
            linearVelocity=ball_initial_velocity,
            angularVelocity=(0.0, 0.0, 0.0),
            physicsClientId=client,
        )
        p.changeDynamics(
            ball_id,
            -1,
            lateralFriction=float(args.ball_friction),
            spinningFriction=0.02,
            rollingFriction=float(args.ball_rolling_friction),
            restitution=float(args.ball_restitution),
            linearDamping=0.01,
            angularDamping=0.05,
            collisionMargin=0.001,
            physicsClientId=client,
        )

        frames = []
        max_height = ball_initial_location[2]

        for frame_index in range(1, frame_end + 1):
            if frame_index > 1:
                for _ in range(substeps):
                    p.stepSimulation(physicsClientId=client)

            ball_pos, ball_quat = p.getBasePositionAndOrientation(
                ball_id, physicsClientId=client
            )
            ball_lin, ball_ang = p.getBaseVelocity(ball_id, physicsClientId=client)
            ball_floor_gap = ball_pos[2] - radius

            if ball_pos[2] > max_height:
                max_height = ball_pos[2]

            frames.append(
                {
                    "frame_index": frame_index,
                    "time_sec": (frame_index - 1) / float(fps),
                    "ball_location": list(ball_pos),
                    "ball_quaternion_xyzw": list(ball_quat),
                    "ball_linear_velocity": list(ball_lin),
                    "ball_angular_velocity": list(ball_ang),
                    "ball_floor_gap": ball_floor_gap,
                }
            )

        return {
            "schema_version": 2,
            "simulator": "pybullet",
            "fps": fps,
            "frame_start": 1,
            "frame_end": frame_end,
            "duration_sec": float(args.duration_sec),
            "substeps_per_frame": substeps,
            "physics_dt": dt,
            "objects": {
                "ball": {
                    "radius": radius,
                    "mass": float(args.ball_mass),
                    "initial_location": list(ball_initial_location),
                    "initial_velocity": list(ball_initial_velocity),
                    "friction": float(args.ball_friction),
                    "restitution": float(args.ball_restitution),
                },
                "floor": {
                    "friction": float(args.floor_friction),
                },
            },
            "quality": {
                "max_height": max_height,
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