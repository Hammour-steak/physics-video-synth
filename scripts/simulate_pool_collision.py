from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pybullet as p

# The shared timed-edit plumbing lives one directory up, next to the DSL.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pcve_timed_edits as timed  # noqa: E402


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
    # A third ball, off by default. The scene's PCVE ADD edit turns it on and
    # names its position; nothing else in the pipeline puts a ball here.
    parser.add_argument("--extra-x", type=float, default=0.0)
    parser.add_argument("--extra-y", type=float, default=-0.3)
    parser.add_argument("--extra-z", type=float, default=0.0)

    # ---- Per-ball overrides (the PCVE edit surface) -----------------------
    # Left at None each falls back to the corresponding global flag above,
    # which reproduces the pre-PCVE behaviour exactly.
    for prefix in ("cue", "target", "extra"):
        parser.add_argument(f"--{prefix}-mass", type=float, default=None)
        parser.add_argument(f"--{prefix}-friction", type=float, default=None)
        parser.add_argument(f"--{prefix}-restitution", type=float, default=None)
        parser.add_argument(f"--{prefix}-rolling-friction", type=float, default=None)
        parser.add_argument(f"--{prefix}-spinning-friction", type=float, default=None)
    # 0 removes that ball from the sim entirely; its frame slot still
    # appears in the output (frozen at its start pose, present=false).
    parser.add_argument("--cue-active", type=int, default=1)
    parser.add_argument("--target-active", type=int, default=1)
    # 0 by default: the extra ball is the one body the baseline scene does not
    # have, so its slot is the mirror of the other two -- an ADD edit sets it
    # to 1 the way a DELETE edit sets one of theirs to 0.
    parser.add_argument("--extra-active", type=int, default=0)
    # Edits that land partway through the clip, as
    # [{"frame": n, "params": {physics_key: new_value, ...}}]. Everything runs
    # on the flags above until frame n, where params are written into the live
    # simulation and the run carries on from the state it had reached.
    parser.add_argument("--timed-edits-json", type=Path, default=None)
    return parser.parse_args()


def apply_timed_params(client: int, params: dict, *, balls: list) -> None:
    """Write one frame's worth of edited physics into the live simulation.

    ``balls`` is [cue, target, extra] and is edited in place. Only what this
    scene's edit vocabulary can produce is handled; anything else raises,
    because an edit that is silently dropped renders as a video that looks like
    the baseline.
    """
    slots = {"cue": 0, "target": 1, "extra": 2}
    fields = {"mass": "mass", "friction": "lateralFriction",
              "rolling_friction": "rollingFriction",
              "spinning_friction": "spinningFriction",
              "restitution": "restitution"}
    for key, value in params.items():
        if key == "active":
            timed.remove_bodies(p, client, balls, value)
            continue
        prefix, _, suffix = key.partition("_")
        if prefix in slots and suffix in fields:
            timed.set_one(p, client, balls[slots[prefix]], fields[suffix], value)
            continue
        raise timed.unknown_param(key)


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
    timed_edits = timed.load_timed_edits(args.timed_edits_json)
    timed.check_horizon(timed_edits, frame_end)
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
    extra_initial_location = (
        float(args.extra_x),
        float(args.extra_y),
        surface_z + radius + float(args.extra_z),
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

        def make_ball(prefix: str, location):
            body = p.createMultiBody(
                baseMass=resolved(prefix, "mass"),
                baseCollisionShapeIndex=ball_shape,
                baseVisualShapeIndex=-1,
                basePosition=location,
                baseOrientation=(0.0, 0.0, 0.0, 1.0),
                physicsClientId=client,
            )
            p.changeDynamics(
                body,
                -1,
                lateralFriction=resolved(prefix, "friction"),
                spinningFriction=resolved(prefix, "spinning_friction"),
                rollingFriction=resolved(prefix, "rolling_friction"),
                restitution=resolved(prefix, "restitution"),
                linearDamping=0.0,
                angularDamping=0.0,
                collisionMargin=0.0005,
                physicsClientId=client,
            )
            return body

        cue_active = bool(int(args.cue_active))
        target_active = bool(int(args.target_active))
        extra_active = bool(int(args.extra_active))

        cue_id = None
        if cue_active:
            cue_id = make_ball("cue", cue_initial_location)
            p.resetBaseVelocity(
                cue_id,
                linearVelocity=cue_initial_velocity,
                physicsClientId=client,
            )

        target_id = None
        if target_active:
            target_id = make_ball("target", target_initial_location)

        extra_id = None
        if extra_active:
            extra_id = make_ball("extra", extra_initial_location)

        # Where each ball was last seen, for the frames after a timed delete
        # takes one away: a ball removed mid-roll has a real pose to freeze at,
        # while one that was never built keeps its start placement.
        last_pose = {
            "cue": (cue_initial_location, (0.0, 0.0, 0.0, 1.0)),
            "target": (target_initial_location, (0.0, 0.0, 0.0, 1.0)),
            "extra": (extra_initial_location, (0.0, 0.0, 0.0, 1.0)),
        }

        frames = []
        min_ball_ball_gap = float("inf")
        min_cue_extra_gap = float("inf")
        min_extra_target_gap = float("inf")

        identity_quat = (0.0, 0.0, 0.0, 1.0)
        zero_vec = (0.0, 0.0, 0.0)

        def surface_gap(a, b) -> float:
            """Clearance between two ball surfaces; 0 is exactly touching."""
            return math.dist(a, b) - 2.0 * radius
        for frame_index in range(1, frame_end + 1):
            if frame_index > 1:
                for _ in range(substeps):
                    p.stepSimulation(physicsClientId=client)

            # The edit lands at the top of its frame: this frame is the first
            # one that shows it, and every frame before it is the source video.
            if frame_index in timed_edits:
                balls = [cue_id, target_id, extra_id]
                apply_timed_params(client, timed_edits[frame_index], balls=balls)
                cue_id, target_id, extra_id = balls

            if cue_id is not None:
                cue_pos, cue_quat = p.getBasePositionAndOrientation(cue_id, physicsClientId=client)
                cue_lin, cue_ang = p.getBaseVelocity(cue_id, physicsClientId=client)
                cue_present = True
                last_pose["cue"] = (cue_pos, cue_quat)
            else:
                cue_pos, cue_quat = last_pose["cue"]
                cue_lin, cue_ang = zero_vec, zero_vec
                cue_present = False

            if target_id is not None:
                target_pos, target_quat = p.getBasePositionAndOrientation(target_id, physicsClientId=client)
                target_lin, target_ang = p.getBaseVelocity(target_id, physicsClientId=client)
                target_present = True
                last_pose["target"] = (target_pos, target_quat)
            else:
                target_pos, target_quat = last_pose["target"]
                target_lin, target_ang = zero_vec, zero_vec
                target_present = False

            if extra_id is not None:
                extra_pos, extra_quat = p.getBasePositionAndOrientation(extra_id, physicsClientId=client)
                extra_lin, extra_ang = p.getBaseVelocity(extra_id, physicsClientId=client)
                extra_present = True
                last_pose["extra"] = (extra_pos, extra_quat)
            else:
                extra_pos, extra_quat = last_pose["extra"]
                extra_lin, extra_ang = zero_vec, zero_vec
                extra_present = False

            if cue_present and target_present:
                ball_ball_gap = surface_gap(cue_pos, target_pos)
                min_ball_ball_gap = min(min_ball_ball_gap, ball_ball_gap)
            else:
                ball_ball_gap = float("inf")

            # Gaps against the ADDed ball, so a parameter sweep can see which
            # of the two contacts an inserted ball actually makes without
            # replaying the whole path.
            if cue_present and extra_present:
                cue_extra_gap = surface_gap(cue_pos, extra_pos)
                min_cue_extra_gap = min(min_cue_extra_gap, cue_extra_gap)
            else:
                cue_extra_gap = float("inf")
            if extra_present and target_present:
                extra_target_gap = surface_gap(extra_pos, target_pos)
                min_extra_target_gap = min(min_extra_target_gap, extra_target_gap)
            else:
                extra_target_gap = float("inf")

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
                    "extra_ball_present": extra_present,
                    "extra_ball_location": list(extra_pos),
                    "extra_ball_quaternion_xyzw": list(extra_quat),
                    "extra_ball_linear_velocity": list(extra_lin),
                    "extra_ball_angular_velocity": list(extra_ang),
                    "cue_ball_table_gap": cue_pos[2] - surface_z - radius,
                    "target_ball_table_gap": target_pos[2] - surface_z - radius,
                    "extra_ball_table_gap": extra_pos[2] - surface_z - radius,
                    "ball_ball_gap": ball_ball_gap,
                    "cue_extra_gap": cue_extra_gap,
                    "extra_target_gap": extra_target_gap,
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
        extra_quats = _rolling_quaternions(
            [tuple(f["extra_ball_location"]) for f in frames], radius
        )
        for f, cq, tq, eq in zip(frames, cue_quats, target_quats, extra_quats):
            f["cue_ball_quaternion_xyzw"] = list(cq)
            f["target_ball_quaternion_xyzw"] = list(tq)
            f["extra_ball_quaternion_xyzw"] = list(eq)

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
                    "active": int(cue_present),
                },
                "target_ball": {
                    "radius": radius,
                    "mass": resolved("target", "mass"),
                    "initial_location": list(target_initial_location),
                    "friction": resolved("target", "friction"),
                    "restitution": resolved("target", "restitution"),
                    "rolling_friction": resolved("target", "rolling_friction"),
                    "spinning_friction": resolved("target", "spinning_friction"),
                    "active": int(target_present),
                },
                "extra_ball": {
                    "radius": radius,
                    "mass": resolved("extra", "mass"),
                    "initial_location": list(extra_initial_location),
                    "friction": resolved("extra", "friction"),
                    "restitution": resolved("extra", "restitution"),
                    "rolling_friction": resolved("extra", "rolling_friction"),
                    "spinning_friction": resolved("extra", "spinning_friction"),
                    "active": int(extra_present),
                },
                "table": {
                    "surface_z": surface_z,
                    "friction": float(args.table_friction),
                    "restitution": float(args.table_restitution),
                },
            },
            "timed_edits": [
            {"frame": frame, "params": params}
            for frame, params in sorted(timed_edits.items())
        ],
        "quality": {
                "min_ball_ball_gap": min_ball_ball_gap,
                "min_cue_extra_gap": min_cue_extra_gap,
                "min_extra_target_gap": min_extra_target_gap,
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
