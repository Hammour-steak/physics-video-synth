from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pybullet as p


# Scene geometry matches render_mahjong_dice.py so the PyBullet trajectory
# can be applied directly as Blender keyframes.  Dimensions are taken from
# assets/models/riichi_mahjong.glb: the two decorative dice sitting in the
# table's center tray are perfect 0.0833-unit cubes resting at world Z
# 0.9382 (both at the exact same height, confirming a flat tray surface).
DIE_EDGE = 0.0833
TRAY_Z = 0.9382
FLOOR_Z = TRAY_Z - DIE_EDGE / 2.0

# Original decorative resting spots (die center), used as the drop targets.
DIE_0_XY = (-0.1612, -0.0052)
DIE_1_XY = (-0.1991, 0.1425)

# Dice fall purely vertically -- no lateral push, no spin, no tilt. Each is
# thrown with its own downward speed on top of gravity, so die_1 lands
# sooner and bounces more than die_2. The physical difference between edits
# then shows up as bounce height and landing time (restitution / initial
# speed) rather than as sliding around the tray.
DIE_0_SPEED = 1.5
DIE_1_SPEED = 0.8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-sec", type=float, default=3.0)
    parser.add_argument("--substeps", type=int, default=12)
    parser.add_argument("--die-edge", type=float, default=DIE_EDGE)

    # Scene-wide (fallback) values -- kept for back-compat with old callers.
    parser.add_argument("--die-mass", type=float, default=0.006)
    parser.add_argument("--die-friction", type=float, default=0.5)
    parser.add_argument("--die-restitution", type=float, default=0.72)
    parser.add_argument("--floor-friction", type=float, default=0.55)
    parser.add_argument(
        "--floor-restitution", type=float, default=None,
        help="Restitution of the tray floor. Defaults to --die-restitution "
        "for back-compat (pre-PCVE the two were locked together).",
    )
    parser.add_argument("--drop-height", type=float, default=0.6)
    parser.add_argument("--gravity-z", type=float, default=-9.8)
    parser.add_argument("--floor-z", type=float, default=FLOOR_Z)
    parser.add_argument("--die-0-xy", nargs=2, type=float, default=list(DIE_0_XY))
    parser.add_argument("--die-1-xy", nargs=2, type=float, default=list(DIE_1_XY))
    # Per-die initial speed magnitude. Direction is fixed per die
    # (DIE_0_VEL_DIR / DIE_1_VEL_DIR) so the PCVE `initial_velocity` edit is
    # a pure scalar knob on how hard each die is thrown.
    parser.add_argument("--die-initial-speeds", nargs=2, type=float,
                        default=(DIE_0_SPEED, DIE_1_SPEED),
                        help="Per-die initial speed magnitude (die_1, die_2) in m/s.")

    # ---- Per-die overrides (the PCVE edit surface) ------------------------
    # Order is always (die_1 = index 0, die_2 = index 1). Left at None each
    # list is broadcast from the corresponding scalar above.
    parser.add_argument(
        "--die-masses", nargs=2, type=float, default=None,
        help="Per-die mass in kg (die_1, die_2). Overrides --die-mass.",
    )
    parser.add_argument(
        "--die-frictions-list", nargs=2, type=float, default=None,
        help="Per-die lateral friction (die_1, die_2). Overrides --die-friction.",
    )
    parser.add_argument(
        "--die-restitutions-list", nargs=2, type=float, default=None,
        help="Per-die restitution (die_1, die_2). Overrides --die-restitution "
        "for the dice; the floor's own restitution stays at --floor-restitution.",
    )
    parser.add_argument(
        "--die-active", nargs=2, type=int, default=(1, 1),
        help="Which dice exist (die_1, die_2). A 0 removes that die from the "
        "simulation entirely; its frame slot still appears (frozen at its start "
        "pose, present=false).",
    )
    return parser.parse_args()


def _resolve_lists(args):
    def broadcast(lst, scalar, label):
        if lst is None:
            return [float(scalar), float(scalar)]
        return [float(v) for v in lst]

    masses = broadcast(args.die_masses, args.die_mass, "--die-masses")
    frictions = broadcast(args.die_frictions_list, args.die_friction, "--die-frictions-list")
    restitutions = broadcast(args.die_restitutions_list, args.die_restitution, "--die-restitutions-list")
    active = [bool(int(v)) for v in args.die_active]
    floor_rest = float(args.floor_restitution) if args.floor_restitution is not None else float(args.die_restitution)
    return masses, frictions, restitutions, active, floor_rest


def simulate(args: argparse.Namespace) -> dict:
    fps = int(args.fps)
    frame_end = max(2, int(round(float(args.duration_sec) * fps)))
    substeps = int(args.substeps)
    dt = 1.0 / float(fps * substeps)

    edge = float(args.die_edge)
    half_edge = edge / 2.0
    floor_z = float(args.floor_z)
    drop_height = float(args.drop_height)

    masses, frictions, restitutions, active, floor_rest = _resolve_lists(args)

    die_xy = [tuple(args.die_0_xy), tuple(args.die_1_xy)]
    initial_locations = [
        (x, y, floor_z + half_edge + drop_height + 0.05 * idx)
        for idx, (x, y) in enumerate(die_xy)
    ]
    # Pure vertical drop: no spin, no tilt, no lateral velocity. What varies
    # between the two dice (and between edits) is the initial downward speed
    # and their bounce coefficients.
    initial_angular_velocities = [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    ]
    speeds = [float(v) for v in args.die_initial_speeds]
    initial_linear_velocities = [
        (0.0, 0.0, -speeds[0]),
        (0.0, 0.0, -speeds[1]),
    ]
    identity_quat = (0.0, 0.0, 0.0, 1.0)
    initial_orientations = [identity_quat, identity_quat]

    client = p.connect(p.DIRECT)
    try:
        p.resetSimulation(physicsClientId=client)
        p.setGravity(0.0, 0.0, float(args.gravity_z), physicsClientId=client)
        p.setTimeStep(dt, physicsClientId=client)
        p.setPhysicsEngineParameter(
            fixedTimeStep=dt,
            numSolverIterations=300,
            contactBreakingThreshold=0.0005,
            deterministicOverlappingPairs=1,
            enableConeFriction=1,
            physicsClientId=client,
        )

        floor_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=(1.2, 1.2, 0.05),
            physicsClientId=client,
        )
        floor_id = p.createMultiBody(
            0.0,
            floor_shape,
            -1,
            (die_xy[0][0], die_xy[0][1], floor_z - 0.05),
            physicsClientId=client,
        )
        p.changeDynamics(
            floor_id,
            -1,
            lateralFriction=float(args.floor_friction),
            restitution=floor_rest,
            physicsClientId=client,
        )

        die_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=(half_edge, half_edge, half_edge),
            physicsClientId=client,
        )
        die_ids: list[int | None] = []
        for idx in range(2):
            if not active[idx]:
                die_ids.append(None)
                continue
            die_id = p.createMultiBody(
                baseMass=masses[idx],
                baseCollisionShapeIndex=die_shape,
                baseVisualShapeIndex=-1,
                basePosition=initial_locations[idx],
                baseOrientation=initial_orientations[idx],
                physicsClientId=client,
            )
            p.resetBaseVelocity(
                die_id,
                linearVelocity=initial_linear_velocities[idx],
                angularVelocity=initial_angular_velocities[idx],
                physicsClientId=client,
            )
            p.changeDynamics(
                die_id,
                -1,
                lateralFriction=frictions[idx],
                spinningFriction=0.01,
                rollingFriction=0.0008,
                restitution=restitutions[idx],
                linearDamping=0.02,
                angularDamping=0.05,
                physicsClientId=client,
            )
            die_ids.append(die_id)

        frames = []

        for frame_index in range(1, frame_end + 1):
            if frame_index > 1:
                for _ in range(substeps):
                    p.stepSimulation(physicsClientId=client)

            die_data = []
            for idx, die_id in enumerate(die_ids):
                if die_id is None:
                    loc = initial_locations[idx]
                    quat = initial_orientations[idx]
                    die_data.append({
                        "present": False,
                        "location": list(loc),
                        "quaternion_xyzw": list(quat),
                        "linear_velocity": [0.0, 0.0, 0.0],
                        "angular_velocity": [0.0, 0.0, 0.0],
                    })
                    continue
                dpos, dquat = p.getBasePositionAndOrientation(die_id, physicsClientId=client)
                dlin, dang = p.getBaseVelocity(die_id, physicsClientId=client)
                die_data.append({
                    "present": True,
                    "location": list(dpos),
                    "quaternion_xyzw": list(dquat),
                    "linear_velocity": list(dlin),
                    "angular_velocity": list(dang),
                })

            frames.append({
                "frame_index": frame_index,
                "time_sec": (frame_index - 1) / float(fps),
                "dice": die_data,
            })

        def up_face_axis(quat: tuple) -> str:
            rot = p.getMatrixFromQuaternion(quat)
            axes = {
                "+X": (rot[0], rot[3], rot[6]),
                "-X": (-rot[0], -rot[3], -rot[6]),
                "+Y": (rot[1], rot[4], rot[7]),
                "-Y": (-rot[1], -rot[4], -rot[7]),
                "+Z": (rot[2], rot[5], rot[8]),
                "-Z": (-rot[2], -rot[5], -rot[8]),
            }
            return max(axes, key=lambda k: axes[k][2])

        final_frame = frames[-1]["dice"]
        settled = []
        up_faces = []
        for idx, die_data in enumerate(final_frame):
            if not active[idx]:
                settled.append(None)
                up_faces.append(None)
                continue
            lin_speed = math.sqrt(sum(v * v for v in die_data["linear_velocity"]))
            ang_speed = math.sqrt(sum(v * v for v in die_data["angular_velocity"]))
            settled.append(bool(lin_speed < 0.02 and ang_speed < 0.05))
            up_faces.append(up_face_axis(tuple(die_data["quaternion_xyzw"])))

        # Per-die displacement in the XY plane -- useful for the sweep to
        # tell "settled where it landed" apart from "slid across the tray".
        xy_displacement = []
        final_z = []
        for idx, die_data in enumerate(final_frame):
            if not active[idx]:
                xy_displacement.append(None)
                final_z.append(None)
                continue
            fx, fy, fz = die_data["location"]
            sx, sy = die_xy[idx]
            xy_displacement.append(math.hypot(fx - sx, fy - sy))
            final_z.append(fz)

        return {
            "schema_version": 3,
            "simulator": "pybullet",
            "fps": fps,
            "frame_start": 1,
            "frame_end": frame_end,
            "duration_sec": float(args.duration_sec),
            "substeps_per_frame": substeps,
            "physics_dt": dt,
            "objects": {
                "dice": {
                    "edge": edge,
                    "count": 2,
                    "masses": masses,
                    "initial_locations": [list(loc) for loc in initial_locations],
                    "initial_speeds": speeds,
                    "frictions": frictions,
                    "restitutions": restitutions,
                    "active": [int(a) for a in active],
                },
                "floor": {
                    "friction": float(args.floor_friction),
                    "restitution": floor_rest,
                    "z": floor_z,
                },
            },
            "quality": {
                "settled": settled,
                "up_face_local_axis": up_faces,
                "xy_displacement": xy_displacement,
                "final_z": final_z,
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
    print(f"[sim] settled={q['settled']}  up_faces={q['up_face_local_axis']}  "
          f"xy_disp={[None if v is None else round(v, 3) for v in q['xy_displacement']]}")


if __name__ == "__main__":
    main()
