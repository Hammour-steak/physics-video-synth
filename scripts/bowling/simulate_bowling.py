from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pybullet as p


# Scene geometry matches render_bowling.py so the PyBullet trajectory can be
# applied directly as Blender keyframes.  The bowling ball and pin dimensions
# are taken from assets/models/bowling_club.glb.
FLOOR_SIZE = 40.0
FLOOR_THICKNESS = 0.1
FLOOR_Z = 0.0

BALL_RADIUS = 0.12
PIN_RADIUS = 0.075
PIN_HEIGHT = 0.495


def _quat_multiply(a: tuple, b: tuple) -> tuple:
    """Hamilton product a * b for scalar-last quaternions (x, y, z, w)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _rolling_quaternions(locations: list[tuple[float, float, float]], radius: float) -> list[tuple]:
    """Reconstruct rolling-without-slipping orientations from a position path."""
    identity = (0.0, 0.0, 0.0, 1.0)
    quaternions = [identity]
    for i in range(1, len(locations)):
        x0, y0, _ = locations[i - 1]
        x1, y1, _ = locations[i]
        dx = x1 - x0
        dy = y1 - y0
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1e-9:
            quaternions.append(quaternions[-1])
            continue
        # Rotation axis: velocity direction cross up (0, 0, 1).
        axis_x = dy / dist
        axis_y = -dx / dist
        half_angle = dist / (2.0 * radius)
        s = math.sin(half_angle)
        c = math.cos(half_angle)
        delta_q = (axis_x * s, axis_y * s, 0.0, c)
        quaternions.append(_quat_multiply(delta_q, quaternions[-1]))
    return quaternions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-sec", type=float, default=3.0)
    parser.add_argument("--substeps", type=int, default=12)
    parser.add_argument("--ball-radius", type=float, default=BALL_RADIUS)
    parser.add_argument("--ball-mass", type=float, default=3.0)
    parser.add_argument("--ball-friction", type=float, default=0.4)
    parser.add_argument("--ball-restitution", type=float, default=0.5)
    parser.add_argument("--pin-radius", type=float, default=PIN_RADIUS)
    parser.add_argument("--pin-height", type=float, default=PIN_HEIGHT)
    parser.add_argument("--pin-mass", type=float, default=0.8)
    parser.add_argument("--pin-friction", type=float, default=0.4)
    parser.add_argument("--pin-restitution", type=float, default=0.3)
    parser.add_argument("--floor-friction", type=float, default=0.5)
    parser.add_argument("--ball-initial-location", nargs=3, type=float, default=(-2.0, 0.0, 0.12))
    parser.add_argument("--ball-initial-velocity", nargs=3, type=float, default=(5.0, 0.0, 0.0))
    parser.add_argument("--pin-spacing", type=float, default=0.28)
    parser.add_argument("--gravity-z", type=float, default=-9.81)
    parser.add_argument("--floor-z", type=float, default=FLOOR_Z)
    parser.add_argument("--scene-offset-x", type=float, default=0.0)
    parser.add_argument("--scene-offset-y", type=float, default=0.0)
    return parser.parse_args()


def simulate(args: argparse.Namespace) -> dict:
    fps = int(args.fps)
    frame_end = max(2, int(round(float(args.duration_sec) * fps)))
    substeps = int(args.substeps)
    dt = 1.0 / float(fps * substeps)
    ball_radius = float(args.ball_radius)
    pin_radius = float(args.pin_radius)
    pin_height = float(args.pin_height)
    pin_half_height = pin_height / 2.0
    pin_spacing = float(args.pin_spacing)
    floor_z = float(args.floor_z)
    offset_x = float(args.scene_offset_x)
    offset_y = float(args.scene_offset_y)

    raw_ball_initial_location = tuple(float(value) for value in args.ball_initial_location)
    ball_initial_velocity = tuple(float(value) for value in args.ball_initial_velocity)
    ball_initial_location = (
        raw_ball_initial_location[0] + offset_x,
        raw_ball_initial_location[1] + offset_y,
        raw_ball_initial_location[2] + floor_z,
    )

    # Three pins in a tight triangle, facing the incoming ball.
    pin_base_positions = [
        (0.0, 0.0),
        (pin_spacing * 0.866, pin_spacing * 0.5),
        (pin_spacing * 0.866, -pin_spacing * 0.5),
    ]
    pin_initial_locations = [
        (x + offset_x, y + offset_y, floor_z + pin_half_height)
        for x, y in pin_base_positions
    ]

    client = p.connect(p.DIRECT)
    try:
        p.resetSimulation(physicsClientId=client)
        p.setGravity(0.0, 0.0, float(args.gravity_z), physicsClientId=client)
        p.setTimeStep(dt, physicsClientId=client)
        p.setPhysicsEngineParameter(
            fixedTimeStep=dt,
            numSolverIterations=200,
            contactBreakingThreshold=0.001,
            deterministicOverlappingPairs=1,
            enableConeFriction=1,
            physicsClientId=client,
        )

        floor_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=(FLOOR_SIZE / 2, FLOOR_SIZE / 2, FLOOR_THICKNESS / 2),
            physicsClientId=client,
        )
        floor_id = p.createMultiBody(
            0.0,
            floor_shape,
            -1,
            (offset_x, offset_y, floor_z - FLOOR_THICKNESS / 2),
            physicsClientId=client,
        )
        p.changeDynamics(
            floor_id,
            -1,
            lateralFriction=float(args.floor_friction),
            restitution=1.0,
            physicsClientId=client,
        )

        ball_shape = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=ball_radius,
            physicsClientId=client,
        )
        ball_id = p.createMultiBody(
            baseMass=float(args.ball_mass),
            baseCollisionShapeIndex=ball_shape,
            baseVisualShapeIndex=-1,
            basePosition=ball_initial_location,
            baseOrientation=(0.0, 0.0, 0.0, 1.0),
            physicsClientId=client,
        )
        # Rolling initial spin matching the forward velocity.
        p.resetBaseVelocity(
            ball_id,
            linearVelocity=ball_initial_velocity,
            angularVelocity=(
                0.0,
                -ball_initial_velocity[0] / ball_radius,
                0.0,
            ),
            physicsClientId=client,
        )
        p.changeDynamics(
            ball_id,
            -1,
            lateralFriction=float(args.ball_friction),
            spinningFriction=0.02,
            rollingFriction=0.0015,
            restitution=float(args.ball_restitution),
            linearDamping=0.005,
            angularDamping=0.005,
            physicsClientId=client,
        )

        pin_shape = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=pin_radius,
            height=pin_height,
            physicsClientId=client,
        )
        pin_ids = []
        for idx, pin_location in enumerate(pin_initial_locations):
            pin_id = p.createMultiBody(
                baseMass=float(args.pin_mass),
                baseCollisionShapeIndex=pin_shape,
                baseVisualShapeIndex=-1,
                basePosition=pin_location,
                baseOrientation=(0.0, 0.0, 0.0, 1.0),
                physicsClientId=client,
            )
            p.changeDynamics(
                pin_id,
                -1,
                lateralFriction=float(args.pin_friction),
                spinningFriction=0.01,
                rollingFriction=0.001,
                restitution=float(args.pin_restitution),
                linearDamping=0.01,
                angularDamping=0.01,
                physicsClientId=client,
            )
            pin_ids.append(pin_id)

        frames = []
        min_ball_floor_gap = float("inf")
        min_ball_pin_gap = float("inf")

        for frame_index in range(1, frame_end + 1):
            if frame_index > 1:
                for _ in range(substeps):
                    p.stepSimulation(physicsClientId=client)

            ball_pos, ball_quat = p.getBasePositionAndOrientation(ball_id, physicsClientId=client)
            ball_lin, ball_ang = p.getBaseVelocity(ball_id, physicsClientId=client)
            ball_floor_gap = ball_pos[2] - ball_radius
            min_ball_floor_gap = min(min_ball_floor_gap, ball_floor_gap)

            pin_data = []
            for pin_id in pin_ids:
                ppos, pquat = p.getBasePositionAndOrientation(pin_id, physicsClientId=client)
                plin, pang = p.getBaseVelocity(pin_id, physicsClientId=client)
                dx = ball_pos[0] - ppos[0]
                dy = ball_pos[1] - ppos[1]
                dz = ball_pos[2] - ppos[2]
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                gap = dist - ball_radius - pin_radius
                min_ball_pin_gap = min(min_ball_pin_gap, gap)
                pin_data.append({
                    "location": list(ppos),
                    "quaternion_xyzw": list(pquat),
                    "linear_velocity": list(plin),
                    "angular_velocity": list(pang),
                    "gap_to_ball": gap,
                })

            frames.append({
                "frame_index": frame_index,
                "time_sec": (frame_index - 1) / float(fps),
                "ball_location": list(ball_pos),
                "ball_quaternion_xyzw": list(ball_quat),
                "ball_linear_velocity": list(ball_lin),
                "ball_angular_velocity": list(ball_ang),
                "ball_floor_gap": ball_floor_gap,
                "pins": pin_data,
            })

        # PyBullet does not always generate rolling rotation for spheres on a
        # plane, so we reconstruct a rolling-without-slipping orientation from the
        # ball path to keep the texture aligned with the motion.
        ball_quats = _rolling_quaternions(
            [tuple(f["ball_location"]) for f in frames], ball_radius
        )
        for f, bq in zip(frames, ball_quats):
            f["ball_quaternion_xyzw"] = list(bq)

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
                    "radius": ball_radius,
                    "mass": float(args.ball_mass),
                    "initial_location": list(ball_initial_location),
                    "initial_linear_velocity": list(ball_initial_velocity),
                    "friction": float(args.ball_friction),
                    "restitution": float(args.ball_restitution),
                },
                "pins": {
                    "radius": pin_radius,
                    "height": pin_height,
                    "count": len(pin_initial_locations),
                    "mass": float(args.pin_mass),
                    "initial_locations": [list(loc) for loc in pin_initial_locations],
                    "friction": float(args.pin_friction),
                    "restitution": float(args.pin_restitution),
                },
                "floor": {
                    "friction": float(args.floor_friction),
                    "z": floor_z,
                },
                "scene_offset": {
                    "x": offset_x,
                    "y": offset_y,
                },
            },
            "quality": {
                "min_ball_floor_gap": min_ball_floor_gap,
                "min_ball_pin_gap": min_ball_pin_gap,
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
