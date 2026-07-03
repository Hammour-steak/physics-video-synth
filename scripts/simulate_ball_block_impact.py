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
    parser.add_argument("--duration-sec", type=float, default=8.0)
    parser.add_argument("--substeps", type=int, default=12)
    parser.add_argument("--ball-radius", type=float, default=0.34)
    parser.add_argument("--ball-initial-location", nargs=3, type=float, default=(-3.05, -0.12, 0.341))
    parser.add_argument("--block-location", nargs=3, type=float, default=(0.23, -0.02, 0.35))
    parser.add_argument("--block-yaw-deg", type=float, default=0.0)
    parser.add_argument("--ball-initial-velocity", nargs=3, type=float, default=(6.0, 0.0, 0.0))
    parser.add_argument("--ball-mass", type=float, default=0.58)
    parser.add_argument("--block-mass", type=float, default=0.65)
    parser.add_argument("--floor-friction", type=float, default=0.82)
    parser.add_argument("--ball-friction", type=float, default=0.38)
    parser.add_argument("--ball-restitution", type=float, default=0.78)
    parser.add_argument("--block-friction", type=float, default=0.32)
    parser.add_argument("--block-restitution", type=float, default=0.55)
    parser.add_argument("--ramp-enabled", action="store_true")
    parser.add_argument("--ramp-location", nargs=3, type=float, default=(-0.35, 0.0, 0.77))
    parser.add_argument("--ramp-dimensions", nargs=3, type=float, default=(2.5, 1.1, 0.08))
    parser.add_argument("--ramp-pitch-deg", type=float, default=18.0)
    parser.add_argument("--ramp-friction", type=float, default=0.56)
    parser.add_argument("--ramp-restitution", type=float, default=0.06)
    return parser.parse_args()


def sphere_box_gap(
    sphere_center: tuple[float, float, float],
    radius: float,
    box_center: tuple[float, float, float],
    box_quat: tuple[float, float, float, float],
    half_extents: tuple[float, float, float],
) -> float:
    inv_pos, inv_quat = p.invertTransform(box_center, box_quat)
    local_center, _ = p.multiplyTransforms(inv_pos, inv_quat, sphere_center, (0.0, 0.0, 0.0, 1.0))
    clamped = tuple(
        max(-extent, min(extent, coord))
        for coord, extent in zip(local_center, half_extents)
    )
    delta = tuple(coord - clamp for coord, clamp in zip(local_center, clamped))
    distance = math.sqrt(sum(value * value for value in delta))
    if distance > 1e-9:
        return distance - radius

    inside_clearance = min(
        extent - abs(coord)
        for coord, extent in zip(local_center, half_extents)
    )
    return -(radius + max(0.0, inside_clearance))


def simulate(args: argparse.Namespace) -> dict:
    fps = int(args.fps)
    frame_end = max(2, int(round(float(args.duration_sec) * fps)))
    substeps = int(args.substeps)
    dt = 1.0 / float(fps * substeps)
    radius = float(args.ball_radius)
    block_half_extents = (0.46, 0.29, 0.35)
    ball_initial_location = tuple(float(value) for value in args.ball_initial_location)
    block_location = tuple(float(value) for value in args.block_location)
    block_yaw = math.radians(float(args.block_yaw_deg))
    block_orientation = p.getQuaternionFromEuler((0.0, 0.0, block_yaw))
    ball_initial_velocity = tuple(float(value) for value in args.ball_initial_velocity)
    ramp_enabled = bool(args.ramp_enabled)
    ramp_location = tuple(float(value) for value in args.ramp_location)
    ramp_dimensions = tuple(float(value) for value in args.ramp_dimensions)
    ramp_half_extents = tuple(0.5 * value for value in ramp_dimensions)
    ramp_pitch = math.radians(float(args.ramp_pitch_deg))
    ramp_orientation = p.getQuaternionFromEuler((0.0, ramp_pitch, 0.0))

    client = p.connect(p.DIRECT)
    try:
        p.resetSimulation(physicsClientId=client)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=client)
        p.setTimeStep(dt, physicsClientId=client)
        p.setPhysicsEngineParameter(
            fixedTimeStep=dt,
            numSolverIterations=180,
            contactBreakingThreshold=0.002,
            deterministicOverlappingPairs=1,
            physicsClientId=client,
        )

        floor_shape = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=client)
        floor_id = p.createMultiBody(0.0, floor_shape, -1, (0.0, 0.0, 0.0), physicsClientId=client)
        p.changeDynamics(
            floor_id,
            -1,
            lateralFriction=float(args.floor_friction),
            restitution=0.0,
            physicsClientId=client,
        )

        ramp_id = None
        if ramp_enabled:
            ramp_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=ramp_half_extents,
                physicsClientId=client,
            )
            ramp_id = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=ramp_shape,
                baseVisualShapeIndex=-1,
                basePosition=ramp_location,
                baseOrientation=ramp_orientation,
                physicsClientId=client,
            )
            p.changeDynamics(
                ramp_id,
                -1,
                lateralFriction=float(args.ramp_friction),
                restitution=float(args.ramp_restitution),
                physicsClientId=client,
            )

        block_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=block_half_extents,
            physicsClientId=client,
        )
        block_id = p.createMultiBody(
            baseMass=float(args.block_mass),
            baseCollisionShapeIndex=block_shape,
            baseVisualShapeIndex=-1,
            basePosition=block_location,
            baseOrientation=block_orientation,
            physicsClientId=client,
        )
        p.changeDynamics(
            block_id,
            -1,
            lateralFriction=float(args.block_friction),
            spinningFriction=0.02,
            rollingFriction=0.006,
            restitution=float(args.block_restitution),
            linearDamping=0.08,
            angularDamping=0.08,
            physicsClientId=client,
        )

        ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=client)
        ball_id = p.createMultiBody(
            baseMass=float(args.ball_mass),
            baseCollisionShapeIndex=ball_shape,
            baseVisualShapeIndex=-1,
            basePosition=ball_initial_location,
            baseOrientation=(0.0, 0.0, 0.0, 1.0),
            physicsClientId=client,
        )
        p.resetBaseVelocity(
            ball_id,
            linearVelocity=ball_initial_velocity,
            angularVelocity=(
                ball_initial_velocity[1] / radius,
                -ball_initial_velocity[0] / radius,
                0.0,
            ),
            physicsClientId=client,
        )
        p.changeDynamics(
            ball_id,
            -1,
            lateralFriction=float(args.ball_friction),
            spinningFriction=0.018,
            rollingFriction=0.0015,
            restitution=float(args.ball_restitution),
            linearDamping=0.006,
            angularDamping=0.006,
            physicsClientId=client,
        )

        frames = []
        min_ball_block_gap = float("inf")
        min_ball_floor_gap = float("inf")
        min_ball_ramp_gap = float("inf")
        for frame_index in range(1, frame_end + 1):
            if frame_index > 1:
                for _ in range(substeps):
                    p.stepSimulation(physicsClientId=client)

            ball_pos, ball_quat = p.getBasePositionAndOrientation(ball_id, physicsClientId=client)
            block_pos, block_quat = p.getBasePositionAndOrientation(block_id, physicsClientId=client)
            ball_lin, ball_ang = p.getBaseVelocity(ball_id, physicsClientId=client)
            block_lin, block_ang = p.getBaseVelocity(block_id, physicsClientId=client)

            ball_floor_gap = ball_pos[2] - radius
            ball_block_gap = sphere_box_gap(ball_pos, radius, block_pos, block_quat, block_half_extents)
            ball_ramp_gap = None
            if ramp_enabled:
                ball_ramp_gap = sphere_box_gap(
                    ball_pos,
                    radius,
                    ramp_location,
                    ramp_orientation,
                    ramp_half_extents,
                )
            min_ball_floor_gap = min(min_ball_floor_gap, ball_floor_gap)
            min_ball_block_gap = min(min_ball_block_gap, ball_block_gap)
            if ball_ramp_gap is not None:
                min_ball_ramp_gap = min(min_ball_ramp_gap, ball_ramp_gap)

            frames.append(
                {
                    "frame_index": frame_index,
                    "time_sec": (frame_index - 1) / float(fps),
                    "ball_location": list(ball_pos),
                    "ball_quaternion_xyzw": list(ball_quat),
                    "ball_linear_velocity": list(ball_lin),
                    "ball_angular_velocity": list(ball_ang),
                    "wood_block_location": list(block_pos),
                    "wood_block_quaternion_xyzw": list(block_quat),
                    "wood_block_linear_velocity": list(block_lin),
                    "wood_block_angular_velocity": list(block_ang),
                    "ball_floor_gap": ball_floor_gap,
                    "ball_block_gap": ball_block_gap,
                    "ball_ramp_gap": ball_ramp_gap,
                }
            )

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
                    "radius": radius,
                    "mass": float(args.ball_mass),
                    "initial_location": list(ball_initial_location),
                    "initial_linear_velocity": list(ball_initial_velocity),
                    "friction": float(args.ball_friction),
                    "restitution": float(args.ball_restitution),
                },
                "wood_block": {
                    "dimensions": [2.0 * value for value in block_half_extents],
                    "mass": float(args.block_mass),
                    "initial_location": list(block_location),
                    "initial_yaw_deg": float(args.block_yaw_deg),
                    "friction": float(args.block_friction),
                    "restitution": float(args.block_restitution),
                },
                "floor": {
                    "friction": float(args.floor_friction),
                },
                "ramp": {
                    "enabled": ramp_enabled,
                    "dimensions": list(ramp_dimensions),
                    "initial_location": list(ramp_location),
                    "pitch_deg": float(args.ramp_pitch_deg),
                    "friction": float(args.ramp_friction),
                    "restitution": float(args.ramp_restitution),
                },
            },
            "quality": {
                "min_ball_floor_gap": min_ball_floor_gap,
                "min_ball_block_gap": min_ball_block_gap,
                "min_ball_ramp_gap": min_ball_ramp_gap if ramp_enabled else None,
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
