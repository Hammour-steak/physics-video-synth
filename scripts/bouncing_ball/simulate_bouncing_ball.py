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


# Scene geometry matches render_bouncing_ball.py so the PyBullet trajectory
# can be applied directly as Blender keyframes.
FLOOR_SIZE = 10.0
FLOOR_THICKNESS = 0.1
FLOOR_Z = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-sec", type=float, default=3.0)
    parser.add_argument("--substeps", type=int, default=12)
    parser.add_argument("--ball-radius", type=float, default=0.05)
    parser.add_argument("--ball-mass", type=float, default=0.2)
    parser.add_argument("--ball-initial-location", nargs=3, type=float, default=(0.0, 0.0, 1.0))
    parser.add_argument("--ball-initial-velocity", nargs=3, type=float, default=(0.5, 0.0, 0.0))
    parser.add_argument("--floor-friction", type=float, default=0.6)
    parser.add_argument("--ball-friction", type=float, default=0.4)
    parser.add_argument("--ball-restitution", type=float, default=0.75)
    parser.add_argument("--gravity-z", type=float, default=-9.81)
    # Edits that land partway through the clip, as
    # [{"frame": n, "params": {physics_key: new_value, ...}}]. Everything runs
    # on the CLI values until frame n, where params are written into the live
    # simulation and the run carries on from the state it had reached.
    parser.add_argument("--timed-edits-json", type=Path, default=None)
    return parser.parse_args()


def apply_timed_params(client: int, params: dict, *, bodies: dict) -> None:
    """Write one frame's worth of edited physics into the live simulation.

    ``bodies`` is the ball. Anything this scene's edit vocabulary cannot
    produce raises: an edit that is silently dropped renders as a video that
    looks like the baseline and nothing downstream would catch it.
    """
    fields = {"ball_mass": ("ball", "mass"),
              "ball_friction": ("ball", "lateralFriction"),
              "ball_restitution": ("ball", "restitution")}
    for key, value in params.items():
        if key not in fields:
            raise timed.unknown_param(key)
        name, field = fields[key]
        timed.set_one(p, client, bodies[name], field, value)


def simulate(args: argparse.Namespace) -> dict:
    fps = int(args.fps)
    frame_end = max(2, int(round(float(args.duration_sec) * fps)))
    substeps = int(args.substeps)
    dt = 1.0 / float(fps * substeps)
    radius = float(args.ball_radius)
    ball_initial_location = tuple(float(value) for value in args.ball_initial_location)
    ball_initial_velocity = tuple(float(value) for value in args.ball_initial_velocity)

    client = p.connect(p.DIRECT)
    try:
        p.resetSimulation(physicsClientId=client)
        p.setGravity(0.0, 0.0, float(args.gravity_z), physicsClientId=client)
        p.setTimeStep(dt, physicsClientId=client)
        p.setPhysicsEngineParameter(
            fixedTimeStep=dt,
            numSolverIterations=180,
            contactBreakingThreshold=0.002,
            deterministicOverlappingPairs=1,
            physicsClientId=client,
        )

        # Static floor as a large box (more stable than GEOM_PLANE for small objects).
        floor_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=(FLOOR_SIZE / 2, FLOOR_SIZE / 2, FLOOR_THICKNESS / 2),
            physicsClientId=client,
        )
        floor_id = p.createMultiBody(
            0.0,
            floor_shape,
            -1,
            (0.0, 0.0, FLOOR_Z - FLOOR_THICKNESS / 2),
            physicsClientId=client,
        )
        # Floor restitution is set to 1.0 so the ball's own restitution fully
        # controls the bounce. PyBullet multiplies the two restitution values.
        p.changeDynamics(
            floor_id,
            -1,
            lateralFriction=float(args.floor_friction),
            restitution=1.0,
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
            spinningFriction=0.02,
            rollingFriction=0.0015,
            restitution=float(args.ball_restitution),
            linearDamping=0.005,
            angularDamping=0.005,
            physicsClientId=client,
        )

        frames = []
        min_ball_floor_gap = float("inf")
        timed_edits = timed.load_timed_edits(args.timed_edits_json)
        timed.check_horizon(timed_edits, frame_end)
        for frame_index in range(1, frame_end + 1):
            if frame_index > 1:
                for _ in range(substeps):
                    p.stepSimulation(physicsClientId=client)

            # The edit lands at the top of its frame: this frame is the
            # first one that shows it, and every frame before it is the
            # source video.
            if frame_index in timed_edits:
                apply_timed_params(client, timed_edits[frame_index], bodies=dict(ball=ball_id))

            ball_pos, ball_quat = p.getBasePositionAndOrientation(ball_id, physicsClientId=client)
            ball_lin, ball_ang = p.getBaseVelocity(ball_id, physicsClientId=client)

            ball_floor_gap = ball_pos[2] - radius
            min_ball_floor_gap = min(min_ball_floor_gap, ball_floor_gap)

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
                "floor": {
                    "friction": float(args.floor_friction),
                },
            },
            "physics": {
                "gravity": [0.0, 0.0, float(args.gravity_z)],
            },
            "timed_edits": [
                {"frame": frame, "params": params}
                for frame, params in sorted(timed_edits.items())
            ],
            "quality": {
                "min_ball_floor_gap": min_ball_floor_gap,
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
