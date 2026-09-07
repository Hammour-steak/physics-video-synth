from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pybullet as p


# Scene geometry matches render_curling_collision.py so the PyBullet
# trajectory can be applied directly as Blender keyframes.  Two curling
# stones slide toward each other along a single line (a pure head-on
# collision, no glancing offset) and, with equal mass, equal-and-opposite
# speed, and low restitution, both come to rest at the point of impact:
# the common post-collision velocity for a perfectly inelastic collision of
# equal masses with opposite momentum is (m*v + m*(-v)) / (2m) = 0.
STONE_RADIUS = 0.145
STONE_HEIGHT = 0.114
FLOOR_Z = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-sec", type=float, default=4.0)
    parser.add_argument("--substeps", type=int, default=60)
    parser.add_argument("--stone-radius", type=float, default=STONE_RADIUS)
    parser.add_argument("--stone-height", type=float, default=STONE_HEIGHT)
    parser.add_argument("--stone-mass", type=float, default=20.0)
    parser.add_argument("--stone-2-mass", type=float, default=20.0)
    parser.add_argument("--stone-friction", type=float, default=0.15)
    parser.add_argument("--stone-restitution", type=float, default=0.0)
    parser.add_argument("--ice-friction", type=float, default=0.015)
    parser.add_argument("--launch-speed", type=float, default=0.9,
        help="Symmetric magnitude applied to both stones (opposite signs). "
        "Overridden per-stone by --stone-1-launch-speed / --stone-2-launch-speed if given.")
    parser.add_argument("--stone-1-launch-speed", type=float, default=None,
        help="Magnitude for the red (stone_1) stone's initial speed along +x. "
        "Defaults to --launch-speed if unset.")
    parser.add_argument("--stone-2-launch-speed", type=float, default=None,
        help="Magnitude for the yellow (stone_2) stone's initial speed along -x. "
        "Defaults to --launch-speed if unset.")
    parser.add_argument("--start-separation", type=float, default=5.0)
    parser.add_argument("--gravity-z", type=float, default=-9.8)
    return parser.parse_args()


def simulate(args: argparse.Namespace) -> dict:
    fps = int(args.fps)
    frame_end = max(2, int(round(float(args.duration_sec) * fps)))
    substeps = int(args.substeps)
    dt = 1.0 / float(fps * substeps)

    radius = float(args.stone_radius)
    height = float(args.stone_height)
    half_height = height / 2.0
    separation = float(args.start_separation)
    speed = float(args.launch_speed)
    speed_1 = float(args.stone_1_launch_speed) if args.stone_1_launch_speed is not None else speed
    speed_2 = float(args.stone_2_launch_speed) if args.stone_2_launch_speed is not None else speed

    initial_locations = [
        (-separation / 2.0, 0.0, FLOOR_Z + half_height),
        (separation / 2.0, 0.0, FLOOR_Z + half_height),
    ]
    initial_velocities = [
        (speed_1, 0.0, 0.0),
        (-speed_2, 0.0, 0.0),
    ]
    masses = [float(args.stone_mass), float(args.stone_2_mass)]

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

        ice_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=(6.0, 1.5, 0.05),
            physicsClientId=client,
        )
        ice_id = p.createMultiBody(
            0.0, ice_shape, -1, (0.0, 0.0, FLOOR_Z - 0.05),
            physicsClientId=client,
        )
        p.changeDynamics(
            ice_id, -1, lateralFriction=float(args.ice_friction),
            restitution=0.1, physicsClientId=client,
        )

        stone_shape = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=radius, height=height, physicsClientId=client,
        )
        stone_ids = []
        for location, velocity, mass in zip(initial_locations, initial_velocities, masses):
            stone_id = p.createMultiBody(
                baseMass=mass,
                baseCollisionShapeIndex=stone_shape,
                baseVisualShapeIndex=-1,
                basePosition=location,
                baseOrientation=(0.0, 0.0, 0.0, 1.0),
                physicsClientId=client,
            )
            p.resetBaseVelocity(
                stone_id, linearVelocity=velocity, angularVelocity=(0.0, 0.0, 0.0),
                physicsClientId=client,
            )
            p.changeDynamics(
                stone_id, -1,
                lateralFriction=float(args.stone_friction),
                spinningFriction=0.01,
                rollingFriction=0.0005,
                restitution=float(args.stone_restitution),
                linearDamping=0.0,
                angularDamping=0.02,
                collisionMargin=0.0005,
                physicsClientId=client,
            )
            stone_ids.append(stone_id)

        frames = []
        min_gap = float("inf")

        for frame_index in range(1, frame_end + 1):
            if frame_index > 1:
                for _ in range(substeps):
                    p.stepSimulation(physicsClientId=client)

            stone_data = []
            positions = []
            for stone_id in stone_ids:
                pos, quat = p.getBasePositionAndOrientation(stone_id, physicsClientId=client)
                lin, ang = p.getBaseVelocity(stone_id, physicsClientId=client)
                positions.append(pos)
                stone_data.append({
                    "location": list(pos),
                    "quaternion_xyzw": list(quat),
                    "linear_velocity": list(lin),
                    "angular_velocity": list(ang),
                })

            gap = math.dist(positions[0][:2], positions[1][:2]) - 2 * radius
            min_gap = min(min_gap, gap)

            frames.append({
                "frame_index": frame_index,
                "time_sec": (frame_index - 1) / float(fps),
                "stones": stone_data,
            })

        final_frame = frames[-1]["stones"]
        final_speeds = [
            math.sqrt(sum(v * v for v in s["linear_velocity"])) for s in final_frame
        ]
        both_at_rest = all(speed_val < 0.03 for speed_val in final_speeds)

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
                "stones": {
                    "count": 2,
                    "radius": radius,
                    "height": height,
                    "masses": masses,
                    "initial_locations": [list(loc) for loc in initial_locations],
                    "initial_velocities": [list(v) for v in initial_velocities],
                    "friction": float(args.stone_friction),
                    "restitution": float(args.stone_restitution),
                },
                "ice": {
                    "friction": float(args.ice_friction),
                    "z": FLOOR_Z,
                },
            },
            "quality": {
                "min_gap": min_gap,
                "final_speeds": final_speeds,
                "both_at_rest": both_at_rest,
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
