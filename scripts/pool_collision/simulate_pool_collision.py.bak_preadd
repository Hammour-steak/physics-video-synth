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
    parser.add_argument("--duration-sec", type=float, default=2.5)
    parser.add_argument("--substeps", type=int, default=12)
    parser.add_argument("--ball-radius", type=float, default=0.05715)
    parser.add_argument("--ball-mass", type=float, default=0.17)
    parser.add_argument("--ball-friction", type=float, default=0.15)
    parser.add_argument("--ball-restitution", type=float, default=0.90)
    parser.add_argument("--ball-rolling-friction", type=float, default=0.02)
    parser.add_argument("--ball-spinning-friction", type=float, default=0.02)
    parser.add_argument("--table-friction", type=float, default=0.08)
    parser.add_argument("--table-restitution", type=float, default=0.10)
    parser.add_argument("--gravity-z", type=float, default=-9.81)
    parser.add_argument("--surface-z", type=float, default=0.8246)
    parser.add_argument("--cue-x", type=float, default=0.0)
    parser.add_argument("--cue-y", type=float, default=-0.6)
    parser.add_argument("--cue-z", type=float, default=0.0)
    parser.add_argument("--target-x", type=float, default=0.0)
    parser.add_argument("--target-y", type=float, default=0.0)
    parser.add_argument("--target-z", type=float, default=0.0)
    parser.add_argument("--cue-vx", type=float, default=0.0)
    parser.add_argument("--cue-vy", type=float, default=1.0)
    parser.add_argument("--cue-vz", type=float, default=0.0)

    # ---- Per-ball overrides (the PCVE edit surface) -----------------------
    # Left at None each falls back to the corresponding global flag above,
    # which reproduces the pre-PCVE behaviour exactly.
    for prefix in ("cue", "target"):
        parser.add_argument(f"--{prefix}-mass", type=float, default=None)
        parser.add_argument(f"--{prefix}-friction", type=float, default=None)
        parser.add_argument(f"--{prefix}-restitution", type=float, default=None)
        parser.add_argument(f"--{prefix}-rolling-friction", type=float, default=None)
        parser.add_argument(f"--{prefix}-spinning-friction", type=float, default=None)
    # 0 removes that ball from the sim entirely; its frame slot still
    # appears in the output (frozen at its start pose, present=false).
    parser.add_argument("--cue-active", type=int, default=1)
    parser.add_argument("--target-active", type=int, default=1)
    return parser.parse_args()


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


def simulate(args: argparse.Namespace) -> dict:
    fps = int(args.fps)
    frame_end = max(2, int(round(float(args.duration_sec) * fps)))
    substeps = int(args.substeps)
    dt = 1.0 / float(fps * substeps)
    radius = float(args.ball_radius)
    surface_z = float(args.surface_z)

    cue_initial_location = (
        float(args.cue_x),
        float(args.cue_y),
        surface_z + radius + float(args.cue_z),
    )
    target_initial_location = (
        float(args.target_x),
        float(args.target_y),
        surface_z + radius + float(args.target_z),
    )
    cue_initial_velocity = (
        float(args.cue_vx),
        float(args.cue_vy),
        float(args.cue_vz),
    )

    client = p.connect(p.DIRECT)
    try:
        p.resetSimulation(physicsClientId=client)
        p.setGravity(0.0, 0.0, float(args.gravity_z), physicsClientId=client)
        p.setTimeStep(dt, physicsClientId=client)
        p.setPhysicsEngineParameter(
            fixedTimeStep=dt,
            numSolverIterations=500,
            contactBreakingThreshold=0.0002,
            deterministicOverlappingPairs=1,
            enableConeFriction=1,
            physicsClientId=client,
        )

        # Static table surface represented by a large thin box.
        table_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=(5.0, 5.0, 0.01),
            physicsClientId=client,
        )
        table_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=table_shape,
            baseVisualShapeIndex=-1,
            basePosition=(0.0, 0.0, surface_z - 0.01),
            baseOrientation=(0.0, 0.0, 0.0, 1.0),
            physicsClientId=client,
        )
        p.changeDynamics(
            table_id,
            -1,
            lateralFriction=float(args.table_friction),
            restitution=float(args.table_restitution),
            collisionMargin=0.0005,
            physicsClientId=client,
        )

        # Low cushion walls around the playing surface to keep balls on the table.
        play_half_x = 0.467
        play_half_y = 1.062
        cushion_offset = 0.005
        wall_x = play_half_x + cushion_offset
        wall_y = play_half_y + cushion_offset
        wall_height = 0.050
        wall_thickness = 0.020
        wall_z = surface_z + wall_height / 2.0

        wall_half_z = wall_height / 2.0
        short_wall = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=(wall_thickness / 2.0, wall_y + wall_thickness / 2.0, wall_half_z),
            physicsClientId=client,
        )
        long_wall = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=(wall_x + wall_thickness / 2.0, wall_thickness / 2.0, wall_half_z),
            physicsClientId=client,
        )

        wall_ids = []
        for wx, wy in ((wall_x, 0.0), (-wall_x, 0.0)):
            wall_id = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=short_wall,
                baseVisualShapeIndex=-1,
                basePosition=(wx, wy, wall_z),
                baseOrientation=(0.0, 0.0, 0.0, 1.0),
                physicsClientId=client,
            )
            wall_ids.append(wall_id)
        for wx, wy in ((0.0, wall_y), (0.0, -wall_y)):
            wall_id = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=long_wall,
                baseVisualShapeIndex=-1,
                basePosition=(wx, wy, wall_z),
                baseOrientation=(0.0, 0.0, 0.0, 1.0),
                physicsClientId=client,
            )
            wall_ids.append(wall_id)

        for wall_id in wall_ids:
            p.changeDynamics(
                wall_id,
                -1,
                lateralFriction=0.1,
                restitution=0.85,
                collisionMargin=0.0005,
                physicsClientId=client,
            )

        def resolved(prefix: str, key: str):
            v = getattr(args, f"{prefix}_{key}", None)
            return float(v) if v is not None else float(getattr(args, f"ball_{key}"))

        ball_shape = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=radius,
            physicsClientId=client,
        )

        cue_active = bool(int(args.cue_active))
        target_active = bool(int(args.target_active))

        cue_id = None
        if cue_active:
            cue_id = p.createMultiBody(
                baseMass=resolved("cue", "mass"),
                baseCollisionShapeIndex=ball_shape,
                baseVisualShapeIndex=-1,
                basePosition=cue_initial_location,
                baseOrientation=(0.0, 0.0, 0.0, 1.0),
                physicsClientId=client,
            )
            p.changeDynamics(
                cue_id,
                -1,
                lateralFriction=resolved("cue", "friction"),
                spinningFriction=resolved("cue", "spinning_friction"),
                rollingFriction=resolved("cue", "rolling_friction"),
                restitution=resolved("cue", "restitution"),
                linearDamping=0.0,
                angularDamping=0.0,
                collisionMargin=0.0005,
                physicsClientId=client,
            )
            p.resetBaseVelocity(
                cue_id,
                linearVelocity=cue_initial_velocity,
                physicsClientId=client,
            )

        target_id = None
        if target_active:
            target_id = p.createMultiBody(
                baseMass=resolved("target", "mass"),
                baseCollisionShapeIndex=ball_shape,
                baseVisualShapeIndex=-1,
                basePosition=target_initial_location,
                baseOrientation=(0.0, 0.0, 0.0, 1.0),
                physicsClientId=client,
            )
            p.changeDynamics(
                target_id,
                -1,
                lateralFriction=resolved("target", "friction"),
                spinningFriction=resolved("target", "spinning_friction"),
                rollingFriction=resolved("target", "rolling_friction"),
                restitution=resolved("target", "restitution"),
                linearDamping=0.0,
                angularDamping=0.0,
                collisionMargin=0.0005,
                physicsClientId=client,
            )

        frames = []
        min_ball_ball_gap = float("inf")

        identity_quat = (0.0, 0.0, 0.0, 1.0)
        zero_vec = (0.0, 0.0, 0.0)
        for frame_index in range(1, frame_end + 1):
            if frame_index > 1:
                for _ in range(substeps):
                    p.stepSimulation(physicsClientId=client)

            if cue_id is not None:
                cue_pos, cue_quat = p.getBasePositionAndOrientation(cue_id, physicsClientId=client)
                cue_lin, cue_ang = p.getBaseVelocity(cue_id, physicsClientId=client)
                cue_present = True
            else:
                cue_pos, cue_quat = cue_initial_location, identity_quat
                cue_lin, cue_ang = zero_vec, zero_vec
                cue_present = False

            if target_id is not None:
                target_pos, target_quat = p.getBasePositionAndOrientation(target_id, physicsClientId=client)
                target_lin, target_ang = p.getBaseVelocity(target_id, physicsClientId=client)
                target_present = True
            else:
                target_pos, target_quat = target_initial_location, identity_quat
                target_lin, target_ang = zero_vec, zero_vec
                target_present = False

            if cue_present and target_present:
                dx = cue_pos[0] - target_pos[0]
                dy = cue_pos[1] - target_pos[1]
                dz = cue_pos[2] - target_pos[2]
                ball_ball_gap = math.sqrt(dx * dx + dy * dy + dz * dz) - 2.0 * radius
                min_ball_ball_gap = min(min_ball_ball_gap, ball_ball_gap)
            else:
                ball_ball_gap = float("inf")

            frames.append(
                {
                    "frame_index": frame_index,
                    "time_sec": (frame_index - 1) / float(fps),
                    "cue_ball_present": cue_present,
                    "cue_ball_location": list(cue_pos),
                    "cue_ball_quaternion_xyzw": list(cue_quat),
                    "cue_ball_linear_velocity": list(cue_lin),
                    "cue_ball_angular_velocity": list(cue_ang),
                    "target_ball_present": target_present,
                    "target_ball_location": list(target_pos),
                    "target_ball_quaternion_xyzw": list(target_quat),
                    "target_ball_linear_velocity": list(target_lin),
                    "target_ball_angular_velocity": list(target_ang),
                    "cue_ball_table_gap": cue_pos[2] - surface_z - radius,
                    "target_ball_table_gap": target_pos[2] - surface_z - radius,
                    "ball_ball_gap": ball_ball_gap,
                }
            )

        # PyBullet does not always generate rolling rotation for spheres on a plane,
        # so we reconstruct a rolling-without-slipping orientation from each ball path.
        cue_quats = _rolling_quaternions(
            [tuple(f["cue_ball_location"]) for f in frames], radius
        )
        target_quats = _rolling_quaternions(
            [tuple(f["target_ball_location"]) for f in frames], radius
        )
        for f, cq, tq in zip(frames, cue_quats, target_quats):
            f["cue_ball_quaternion_xyzw"] = list(cq)
            f["target_ball_quaternion_xyzw"] = list(tq)

        return {
            "schema_version": 2,
            "simulator": "pybullet",
            "fps": fps,
            "frame_start": 1,
            "frame_end": frame_end,
            "duration_sec": float(args.duration_sec),
            "substeps_per_frame": substeps,
            "physics_dt": dt,
            "surface_z": surface_z,
            "gravity_z": float(args.gravity_z),
            "objects": {
                "cue_ball": {
                    "radius": radius,
                    "mass": resolved("cue", "mass"),
                    "initial_location": list(cue_initial_location),
                    "initial_velocity": list(cue_initial_velocity),
                    "friction": resolved("cue", "friction"),
                    "restitution": resolved("cue", "restitution"),
                    "rolling_friction": resolved("cue", "rolling_friction"),
                    "spinning_friction": resolved("cue", "spinning_friction"),
                    "active": int(cue_active),
                },
                "target_ball": {
                    "radius": radius,
                    "mass": resolved("target", "mass"),
                    "initial_location": list(target_initial_location),
                    "friction": resolved("target", "friction"),
                    "restitution": resolved("target", "restitution"),
                    "rolling_friction": resolved("target", "rolling_friction"),
                    "spinning_friction": resolved("target", "spinning_friction"),
                    "active": int(target_active),
                },
                "table": {
                    "surface_z": surface_z,
                    "friction": float(args.table_friction),
                    "restitution": float(args.table_restitution),
                },
            },
            "quality": {
                "min_ball_ball_gap": min_ball_ball_gap,
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
