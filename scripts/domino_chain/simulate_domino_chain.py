from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pybullet as p


# Scene geometry matches render_domino_chain.py so the PyBullet trajectory
# can be applied directly as Blender keyframes.  The domino dimensions are
# taken from assets/models/domino_test.glb (a single upright tile, local
# X = thickness/row axis, Y = width, Z = height).
FLOOR_Z = -0.0322
DOMINO_THICKNESS = 0.20
DOMINO_WIDTH = 0.70
DOMINO_HEIGHT = 1.30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-sec", type=float, default=3.0)
    parser.add_argument("--substeps", type=int, default=12)
    parser.add_argument("--domino-count", type=int, default=4)
    parser.add_argument("--domino-spacing", type=float, default=0.55)
    parser.add_argument("--domino-thickness", type=float, default=DOMINO_THICKNESS)
    parser.add_argument("--domino-width", type=float, default=DOMINO_WIDTH)
    parser.add_argument("--domino-height", type=float, default=DOMINO_HEIGHT)
    parser.add_argument("--domino-mass", type=float, default=0.12)
    parser.add_argument("--domino-friction", type=float, default=0.6)
    parser.add_argument("--domino-restitution", type=float, default=0.05)
    parser.add_argument("--floor-friction", type=float, default=0.6)
    parser.add_argument("--push-angle-deg", type=float, default=12.0)
    parser.add_argument("--gravity-z", type=float, default=-9.81)
    parser.add_argument("--floor-z", type=float, default=FLOOR_Z)
    parser.add_argument("--scene-offset-x", type=float, default=0.0)
    parser.add_argument("--scene-offset-y", type=float, default=0.0)

    # ---- Per-domino overrides (the PCVE edit surface) ---------------------
    # Each list has one entry per tile in row order (index 0 = the pushed
    # tile). Left at None the corresponding global (--domino-mass etc.) is
    # broadcast across the row, which reproduces the pre-PCVE behaviour.
    parser.add_argument(
        "--domino-masses", nargs="+", type=float, default=None,
        help="Per-tile mass in kg, in row order (must match --domino-count).",
    )
    parser.add_argument(
        "--domino-frictions-list", nargs="+", type=float, default=None,
        help="Per-tile lateral friction, in row order (must match --domino-count).",
    )
    parser.add_argument(
        "--domino-restitutions-list", nargs="+", type=float, default=None,
        help="Per-tile restitution, in row order (must match --domino-count).",
    )
    parser.add_argument(
        "--domino-active", nargs="+", type=int, default=None,
        help="Which tiles exist, in row order. A 0 removes that tile from the "
        "simulation entirely; its frame slot still appears in the output "
        "(frozen at its start pose, present=false) so consumers keep a fixed "
        "layout. Must match --domino-count.",
    )
    return parser.parse_args()


def _resolve_lists(args, count):
    def broadcast(list_arg, scalar_arg, label):
        if list_arg is None:
            return [float(scalar_arg)] * count
        if len(list_arg) != count:
            raise ValueError(f"{label} must have {count} entries, got {len(list_arg)}")
        return [float(v) for v in list_arg]

    masses = broadcast(args.domino_masses, args.domino_mass, "--domino-masses")
    frictions = broadcast(args.domino_frictions_list, args.domino_friction, "--domino-frictions-list")
    restitutions = broadcast(args.domino_restitutions_list, args.domino_restitution, "--domino-restitutions-list")
    if args.domino_active is None:
        active = [True] * count
    else:
        if len(args.domino_active) != count:
            raise ValueError(f"--domino-active must have {count} entries, got {len(args.domino_active)}")
        active = [bool(int(v)) for v in args.domino_active]
    return masses, frictions, restitutions, active


def simulate(args: argparse.Namespace) -> dict:
    fps = int(args.fps)
    frame_end = max(2, int(round(float(args.duration_sec) * fps)))
    substeps = int(args.substeps)
    dt = 1.0 / float(fps * substeps)

    count = int(args.domino_count)
    if count < 2:
        raise ValueError("--domino-count must be at least 2 for a chain reaction.")
    spacing = float(args.domino_spacing)
    thickness = float(args.domino_thickness)
    width = float(args.domino_width)
    height = float(args.domino_height)
    floor_z = float(args.floor_z)
    offset_x = float(args.scene_offset_x)
    offset_y = float(args.scene_offset_y)
    half_height = height / 2.0

    masses, frictions, restitutions, active = _resolve_lists(args, count)

    row_start = -(count - 1) * spacing / 2.0
    base_positions = [(row_start + i * spacing, 0.0) for i in range(count)]
    initial_locations = [
        (x + offset_x, y + offset_y, floor_z + half_height) for x, y in base_positions
    ]

    identity_quat = (0.0, 0.0, 0.0, 1.0)
    push_angle_deg = float(args.push_angle_deg)
    push_quat = p.getQuaternionFromEuler((0.0, math.radians(push_angle_deg), 0.0))
    initial_orientations = [push_quat] + [identity_quat] * (count - 1)

    client = p.connect(p.DIRECT)
    try:
        p.resetSimulation(physicsClientId=client)
        p.setGravity(0.0, 0.0, float(args.gravity_z), physicsClientId=client)
        p.setTimeStep(dt, physicsClientId=client)
        p.setPhysicsEngineParameter(
            fixedTimeStep=dt,
            numSolverIterations=400,
            contactBreakingThreshold=0.0005,
            deterministicOverlappingPairs=1,
            enableConeFriction=1,
            physicsClientId=client,
        )

        floor_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=(12.0, 12.0, 0.05),
            physicsClientId=client,
        )
        floor_id = p.createMultiBody(
            0.0,
            floor_shape,
            -1,
            (offset_x, offset_y, floor_z - 0.05),
            physicsClientId=client,
        )
        p.changeDynamics(
            floor_id,
            -1,
            lateralFriction=float(args.floor_friction),
            restitution=0.1,
            physicsClientId=client,
        )

        domino_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=(thickness / 2.0, width / 2.0, half_height),
            physicsClientId=client,
        )
        domino_ids: list[int | None] = []
        for idx, (location, orientation) in enumerate(zip(initial_locations, initial_orientations)):
            if not active[idx]:
                domino_ids.append(None)
                continue
            domino_id = p.createMultiBody(
                baseMass=masses[idx],
                baseCollisionShapeIndex=domino_shape,
                baseVisualShapeIndex=-1,
                basePosition=location,
                baseOrientation=orientation,
                physicsClientId=client,
            )
            p.changeDynamics(
                domino_id,
                -1,
                lateralFriction=frictions[idx],
                spinningFriction=0.01,
                rollingFriction=0.0005,
                restitution=restitutions[idx],
                linearDamping=0.02,
                angularDamping=0.02,
                physicsClientId=client,
            )
            domino_ids.append(domino_id)

        frames = []
        max_tilt_deg = [0.0] * count

        for frame_index in range(1, frame_end + 1):
            if frame_index > 1:
                for _ in range(substeps):
                    p.stepSimulation(physicsClientId=client)

            domino_data = []
            for idx, domino_id in enumerate(domino_ids):
                if domino_id is None:
                    loc = initial_locations[idx]
                    quat = initial_orientations[idx]
                    domino_data.append({
                        "present": False,
                        "location": list(loc),
                        "quaternion_xyzw": list(quat),
                        "linear_velocity": [0.0, 0.0, 0.0],
                        "angular_velocity": [0.0, 0.0, 0.0],
                        "tilt_deg": 0.0,
                    })
                    continue
                dpos, dquat = p.getBasePositionAndOrientation(domino_id, physicsClientId=client)
                dlin, dang = p.getBaseVelocity(domino_id, physicsClientId=client)

                rot_matrix = p.getMatrixFromQuaternion(dquat)
                local_z_world_z = rot_matrix[8]
                tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, local_z_world_z))))
                max_tilt_deg[idx] = max(max_tilt_deg[idx], tilt_deg)

                domino_data.append({
                    "present": True,
                    "location": list(dpos),
                    "quaternion_xyzw": list(dquat),
                    "linear_velocity": list(dlin),
                    "angular_velocity": list(dang),
                    "tilt_deg": tilt_deg,
                })

            frames.append({
                "frame_index": frame_index,
                "time_sec": (frame_index - 1) / float(fps),
                "dominoes": domino_data,
            })

        toppled_count = sum(1 for i, tilt in enumerate(max_tilt_deg) if active[i] and tilt > 45.0)

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
                "dominoes": {
                    "count": count,
                    "spacing": spacing,
                    "thickness": thickness,
                    "width": width,
                    "height": height,
                    "initial_locations": [list(loc) for loc in initial_locations],
                    "initial_orientations_xyzw": [list(o) for o in initial_orientations],
                    "masses": masses,
                    "frictions": frictions,
                    "restitutions": restitutions,
                    "active": [int(a) for a in active],
                    "push_angle_deg": push_angle_deg,
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
                "max_tilt_deg_per_domino": max_tilt_deg,
                "toppled_count": toppled_count,
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
    print(f"[sim] toppled={q['toppled_count']}/{int(args.domino_count)}  "
          f"tilts={['%.1f' % t for t in q['max_tilt_deg_per_domino']]}")


if __name__ == "__main__":
    main()
