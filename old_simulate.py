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
    parser.add_argument("--ball-radius", type=float, default=0.015)
    parser.add_argument("--ramp-angle-deg", type=float, default=27.0)
    parser.add_argument("--ramp-length", type=float, default=0.28)
    parser.add_argument("--ramp-thickness", type=float, default=0.028)
    parser.add_argument("--ramp-width", type=float, default=0.08)
    parser.add_argument("--ball-mass", type=float, default=0.022)
    parser.add_argument("--marble-mass", type=float, default=0.018)
    parser.add_argument("--marble-radius", type=float, default=0.012)
    parser.add_argument("--floor-friction", type=float, default=0.35)
    parser.add_argument("--ramp-friction", type=float, default=0.18)
    parser.add_argument("--ball-friction", type=float, default=0.12)
    parser.add_argument("--ball-restitution", type=float, default=0.75)
    parser.add_argument("--marble-friction", type=float, default=0.15)
    parser.add_argument("--marble-restitution", type=float, default=0.70)
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
    ramp_angle = math.radians(float(args.ramp_angle_deg))
    ramp_length = float(args.ramp_length)
    ramp_thickness = float(args.ramp_thickness)
    ramp_width = float(args.ramp_width)
    marble_radius = float(args.marble_radius)

    cos_a = math.cos(ramp_angle)
    sin_a = math.sin(ramp_angle)

    ramp_center_z = ramp_length / 2 * sin_a + ramp_thickness / 2 * cos_a

    # Ball at the HIGH end of the ramp (local -X, resting on top surface)
    ball_local_x = -(ramp_length / 2 - radius - 0.008)
    ball_local_z = ramp_thickness / 2 + radius + 0.0015
    ball_initial_x = ball_local_x * cos_a + ball_local_z * sin_a
    ball_initial_z = -ball_local_x * sin_a + ball_local_z * cos_a + ramp_center_z

    # Rightmost point of the ramp at the low end (top corner, local +X,+Z)
    ramp_low_top_x = ramp_length / 2 * cos_a + ramp_thickness / 2 * sin_a

    ball_initial_location = (ball_initial_x, 0.0, ball_initial_z)

    # Marbles on the floor at the low end, safely to the right of the ramp body
    marble_base_x = ramp_low_top_x + marble_radius + 0.5
    marble_offsets = [0.0, 0.06, 0.12, 0.18]
    marble_y_offsets = [0.0, -0.04, 0.04, -0.02]
    marble_locations = []
    for i, dx in enumerate(marble_offsets):
        y_off = marble_y_offsets[i]
        marble_locations.append((marble_base_x + dx, y_off, marble_radius))

    client = p.connect(p.DIRECT)
    try:
        p.resetSimulation(physicsClientId=client)
        p.setGravity(0.0, 0.0, -1.0, physicsClientId=client)
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
            restitution=0.05,
            collisionMargin=0.001,
            physicsClientId=client,
        )

        ramp_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=(ramp_length / 2, ramp_width / 2, ramp_thickness / 2),
            physicsClientId=client,
        )
        ramp_orientation = p.getQuaternionFromEuler((0.0, ramp_angle, 0.0))
        ramp_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=ramp_shape,
            baseVisualShapeIndex=-1,
            basePosition=(0.0, 0.0, ramp_center_z),
            baseOrientation=ramp_orientation,
            physicsClientId=client,
        )
        p.changeDynamics(
            ramp_id,
            -1,
            lateralFriction=float(args.ramp_friction),
            restitution=0.05,
            collisionMargin=0.001,
            physicsClientId=client,
        )

        marble_shape = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=marble_radius,
            physicsClientId=client,
        )

        marble_ids = []
        for ml in marble_locations:
            marble_id = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=marble_shape,
                baseVisualShapeIndex=-1,
                basePosition=ml,
                baseOrientation=(0.0, 0.0, 0.0, 1.0),
                physicsClientId=client,
            )
            p.changeDynamics(
                marble_id,
                -1,
                lateralFriction=float(args.marble_friction),
                spinningFriction=0.025,
                rollingFriction=0.0015,
                restitution=float(args.marble_restitution),
                linearDamping=0.01,
                angularDamping=0.01,
                collisionMargin=0.001,
                physicsClientId=client,
            )
            marble_ids.append(marble_id)

        ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=client)
        ball_id = p.createMultiBody(
            baseMass=float(args.ball_mass),
            baseCollisionShapeIndex=ball_shape,
            baseVisualShapeIndex=-1,
            basePosition=ball_initial_location,
            baseOrientation=(0.0, 0.0, 0.0, 1.0),
            physicsClientId=client,
        )
        p.changeDynamics(
            ball_id,
            -1,
            lateralFriction=float(args.ball_friction),
            spinningFriction=0.02,
            rollingFriction=0.0012,
            restitution=float(args.ball_restitution),
            linearDamping=0.008,
            angularDamping=0.008,
            collisionMargin=0.001,
            physicsClientId=client,
        )

        frames = []
        min_ball_marble_gap = float("inf")

        for frame_index in range(1, frame_end + 1):
            if frame_index > 1:
                for _ in range(substeps):
                    p.stepSimulation(physicsClientId=client)

            ball_pos, ball_quat = p.getBasePositionAndOrientation(ball_id, physicsClientId=client)
            ball_lin, ball_ang = p.getBaseVelocity(ball_id, physicsClientId=client)
            ball_floor_gap = ball_pos[2] - radius

            marble_data = []
            for ml_id in marble_ids:
                mpos, mquat = p.getBasePositionAndOrientation(ml_id, physicsClientId=client)
                mlin, mang = p.getBaseVelocity(ml_id, physicsClientId=client)
                dx = ball_pos[0] - mpos[0]
                dy = ball_pos[1] - mpos[1]
                dz = ball_pos[2] - mpos[2]
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                gap = dist - radius - marble_radius
                min_ball_marble_gap = min(min_ball_marble_gap, gap)
                marble_data.append({
                    "location": list(mpos),
                    "quaternion_xyzw": list(mquat),
                    "linear_velocity": list(mlin),
                    "angular_velocity": list(mang),
                    "gap_to_ball": gap,
                })

            frames.append(
                {
                    "frame_index": frame_index,
                    "time_sec": (frame_index - 1) / float(fps),
                    "ball_location": list(ball_pos),
                    "ball_quaternion_xyzw": list(ball_quat),
                    "ball_linear_velocity": list(ball_lin),
                    "ball_angular_velocity": list(ball_ang),
                    "ball_floor_gap": ball_floor_gap,
                    "marbles": marble_data,
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
            "ramp": {
                "angle_deg": float(args.ramp_angle_deg),
                "length": ramp_length,
                "width": ramp_width,
                "thickness": ramp_thickness,
                "friction": float(args.ramp_friction),
            },
            "objects": {
                "ball": {
                    "radius": radius,
                    "mass": float(args.ball_mass),
                    "initial_location": list(ball_initial_location),
                    "friction": float(args.ball_friction),
                    "restitution": float(args.ball_restitution),
                },
                "marbles": {
                    "radius": marble_radius,
                    "count": len(marble_locations),
                    "mass": float(args.marble_mass),
                    "initial_locations": [list(ml) for ml in marble_locations],
                    "friction": float(args.marble_friction),
                    "restitution": float(args.marble_restitution),
                },
                "floor": {
                    "friction": float(args.floor_friction),
                },
            },
            "quality": {
                "min_ball_marble_gap": min_ball_marble_gap,
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
