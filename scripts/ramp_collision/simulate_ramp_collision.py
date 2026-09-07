from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pybullet as p


# Scene geometry matches render_ramp_collision.py so the PyBullet trajectory
# can be applied directly as Blender keyframes.
TABLE_HEIGHT = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-sec", type=float, default=3.0)
    parser.add_argument("--substeps", type=int, default=12)
    parser.add_argument("--ball-radius", type=float, default=0.012)
    parser.add_argument("--ramp-angle-deg", type=float, default=12.0)
    parser.add_argument("--ramp-length", type=float, default=0.22)
    parser.add_argument("--ramp-thickness", type=float, default=0.025)
    parser.add_argument("--ramp-width", type=float, default=0.16)
    parser.add_argument("--ball-mass", type=float, default=0.05)
    parser.add_argument("--marble-mass", type=float, default=0.05)
    parser.add_argument("--marble-mass-0", type=float, default=None)
    parser.add_argument("--marble-mass-1", type=float, default=None)
    parser.add_argument("--marble-radius", type=float, default=0.012)
    parser.add_argument("--floor-friction", type=float, default=0.4)
    parser.add_argument("--ramp-friction", type=float, default=0.7)
    parser.add_argument("--ball-friction", type=float, default=0.45)
    parser.add_argument("--ball-restitution", type=float, default=0.6)
    parser.add_argument("--ball-rolling-friction", type=float, default=0.002)
    parser.add_argument("--marble-friction", type=float, default=0.15)
    parser.add_argument("--marble-restitution", type=float, default=0.3)
    # Per-marble on/off (two marbles: 0 = blue, 1 = yellow). 1 = present, 0 = removed.
    parser.add_argument("--marble-active", nargs=2, type=int, default=(1, 1))
    parser.add_argument("--marble-initial-velocity-0", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument("--marble-initial-velocity-1", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    # Edits that land partway through the clip, as
    # [{"frame": n, "params": {physics_key: new_value, ...}}]. Everything runs
    # on the CLI values until frame n, where params are written into the live
    # simulation and the run carries on from the state it had reached.
    parser.add_argument("--timed-edits-json", type=Path, default=None)
    return parser.parse_args()


def load_timed_edits(path: Path | None) -> dict[int, dict]:
    """Read the schedule into {frame: merged params}."""
    if path is None:
        return {}
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    schedule: dict[int, dict] = {}
    for entry in entries:
        frame = int(entry["frame"])
        if frame < 2:
            raise ValueError(
                f"Timed edit at frame {frame}: frame 1 is the initial state, "
                "which is set through the ordinary CLI parameters."
            )
        schedule.setdefault(frame, {}).update(dict(entry["params"]))
    return schedule


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


def apply_timed_params(
    client: int,
    params: dict,
    *,
    ball_id: int,
    marble_ids: list,
    floor_id: int,
    ramp_id: int,
) -> None:
    """Write one frame's worth of edited physics into the live simulation.

    Only the parameters this scene's edit vocabulary can actually produce are
    handled; an unrecognised key raises rather than being ignored, because a
    silently dropped edit renders as a video that looks like the baseline and
    would be indistinguishable from a correct one in the benchmark.

    Removing a body mid-run is a real ``removeBody``: the marble stops
    colliding with anything from this frame on, which is the whole point of a
    delete that lands just before the impact.
    """
    for key, value in params.items():
        if key == "marble_active":
            for idx, active in enumerate(value):
                if int(active) or marble_ids[idx] is None:
                    continue
                p.removeBody(marble_ids[idx], physicsClientId=client)
                marble_ids[idx] = None
        elif key == "ball_mass":
            p.changeDynamics(ball_id, -1, mass=float(value), physicsClientId=client)
        elif key == "marble_masses":
            for idx, mass in enumerate(value):
                if marble_ids[idx] is None:
                    continue
                p.changeDynamics(marble_ids[idx], -1, mass=float(mass),
                                 physicsClientId=client)
        elif key in ("ball_friction", "ball_rolling_friction", "ball_restitution"):
            field = {"ball_friction": "lateralFriction",
                     "ball_rolling_friction": "rollingFriction",
                     "ball_restitution": "restitution"}[key]
            p.changeDynamics(ball_id, -1, **{field: float(value)},
                             physicsClientId=client)
        elif key in ("marble_friction", "marble_restitution"):
            field = {"marble_friction": "lateralFriction",
                     "marble_restitution": "restitution"}[key]
            for marble_id in marble_ids:
                if marble_id is None:
                    continue
                p.changeDynamics(marble_id, -1, **{field: float(value)},
                                 physicsClientId=client)
        elif key == "floor_friction":
            p.changeDynamics(floor_id, -1, lateralFriction=float(value),
                             physicsClientId=client)
        elif key == "ramp_friction":
            p.changeDynamics(ramp_id, -1, lateralFriction=float(value),
                             physicsClientId=client)
        else:
            raise ValueError(
                f"Timed edit sets {key!r}, which this scene cannot change "
                "mid-run. Add it to apply_timed_params or write the edit as a "
                "whole-clip edit."
            )


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
    timed_edits = load_timed_edits(args.timed_edits_json)
    latest_edit_frame = max(timed_edits, default=0)
    if latest_edit_frame > frame_end:
        raise ValueError(
            f"Timed edit at frame {latest_edit_frame} is past the end of a "
            f"{frame_end}-frame clip"
        )

    cos_a = math.cos(ramp_angle)
    sin_a = math.sin(ramp_angle)

    floor_z = TABLE_HEIGHT

    # Ramp center height places the bottom of the box on the table top.
    ramp_center_z = floor_z + ramp_length / 2 * sin_a + ramp_thickness / 2 * cos_a

    # Ball at the HIGH end of the ramp (local -X, resting on top surface)
    ball_local_x = -(ramp_length / 2 - radius - 0.002)
    ball_local_z = ramp_thickness / 2 + radius + 0.0002
    ball_initial_x = ball_local_x * cos_a + ball_local_z * sin_a
    ball_initial_z = -ball_local_x * sin_a + ball_local_z * cos_a + ramp_center_z

    # Rightmost point of the ramp at the low end (top corner, local +X,+Z)
    ramp_low_top_x = ramp_length / 2 * cos_a + ramp_thickness / 2 * sin_a

    ball_initial_location = (ball_initial_x, 0.0, ball_initial_z)

    # Marbles on the table at the low end, matching render_ramp_collision.py.
    marble_base_x = ramp_low_top_x + marble_radius + 0.04
    stationary_positions = [
        (marble_base_x, 0.0, floor_z + marble_radius),
        (marble_base_x + 0.03, -0.15, floor_z + marble_radius),
    ]
    marble_locations = [
        (x, y, z)
        for x, y, z in stationary_positions
    ]

    client = p.connect(p.DIRECT)
    try:
        p.resetSimulation(physicsClientId=client)
        p.setGravity(0.0, 0.0, -9.8, physicsClientId=client)
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
            (0.0, 0.0, floor_z - 0.1),
            physicsClientId=client,
        )
        p.changeDynamics(
            floor_id,
            -1,
            lateralFriction=float(args.floor_friction),
            restitution=0.2,
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

        marble_active = tuple(bool(int(v)) for v in args.marble_active)
        marble_initial_velocities = (
            tuple(float(v) for v in args.marble_initial_velocity_0),
            tuple(float(v) for v in args.marble_initial_velocity_1),
        )
        marble_masses = (
            float(args.marble_mass_0) if args.marble_mass_0 is not None else float(args.marble_mass),
            float(args.marble_mass_1) if args.marble_mass_1 is not None else float(args.marble_mass),
        )
        marble_ids: list[int | None] = []
        for idx, ml in enumerate(marble_locations):
            if not marble_active[idx]:
                marble_ids.append(None)
                continue
            marble_id = p.createMultiBody(
                baseMass=marble_masses[idx],
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
                rollingFriction=0.0012,
                restitution=float(args.marble_restitution),
                linearDamping=0.1,
                angularDamping=0.1,
                collisionMargin=0.001,
                physicsClientId=client,
            )
            v_init = marble_initial_velocities[idx]
            if any(abs(component) > 1e-9 for component in v_init):
                p.resetBaseVelocity(
                    marble_id,
                    linearVelocity=v_init,
                    angularVelocity=(0.0, 0.0, 0.0),
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
            rollingFriction=float(args.ball_rolling_friction),
            restitution=float(args.ball_restitution),
            linearDamping=0.05,
            angularDamping=0.05,
            collisionMargin=0.001,
            physicsClientId=client,
        )

        frames = []
        min_ball_marble_gap = float("inf")
        # Where each marble was last seen. A marble that is deleted partway
        # through has a real pose to freeze at; one that was never built keeps
        # its initial placement, which is what the ground truth recorded before
        # timed edits existed.
        marble_last_pose = [
            {"location": list(ml), "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]}
            for ml in marble_locations
        ]

        for frame_index in range(1, frame_end + 1):
            if frame_index > 1:
                for _ in range(substeps):
                    p.stepSimulation(physicsClientId=client)

            # The edit lands at the top of its frame: this frame is the first
            # one that shows it, and every frame before it is the source video.
            if frame_index in timed_edits:
                apply_timed_params(
                    client,
                    timed_edits[frame_index],
                    ball_id=ball_id,
                    marble_ids=marble_ids,
                    floor_id=floor_id,
                    ramp_id=ramp_id,
                )

            ball_pos, ball_quat = p.getBasePositionAndOrientation(ball_id, physicsClientId=client)
            ball_lin, ball_ang = p.getBaseVelocity(ball_id, physicsClientId=client)
            ball_floor_gap = ball_pos[2] - radius

            marble_data = []
            for idx, ml_id in enumerate(marble_ids):
                if ml_id is None:
                    marble_data.append({
                        "active": False,
                        "location": list(marble_last_pose[idx]["location"]),
                        "quaternion_xyzw": list(marble_last_pose[idx]["quaternion_xyzw"]),
                        "linear_velocity": [0.0, 0.0, 0.0],
                        "angular_velocity": [0.0, 0.0, 0.0],
                        "gap_to_ball": None,
                    })
                    continue
                mpos, mquat = p.getBasePositionAndOrientation(ml_id, physicsClientId=client)
                mlin, mang = p.getBaseVelocity(ml_id, physicsClientId=client)
                dx = ball_pos[0] - mpos[0]
                dy = ball_pos[1] - mpos[1]
                dz = ball_pos[2] - mpos[2]
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                gap = dist - radius - marble_radius
                min_ball_marble_gap = min(min_ball_marble_gap, gap)
                marble_last_pose[idx] = {"location": list(mpos),
                                         "quaternion_xyzw": list(mquat)}
                marble_data.append({
                    "active": True,
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
                    "masses": list(marble_masses),
                    "initial_locations": [list(ml) for ml in marble_locations],
                    "active": [bool(v) for v in marble_active],
                    "initial_velocities": [list(v) for v in marble_initial_velocities],
                    "friction": float(args.marble_friction),
                    "restitution": float(args.marble_restitution),
                },
                "floor": {
                    "friction": float(args.floor_friction),
                },
            },
            "timed_edits": [
                {"frame": frame, "params": params}
                for frame, params in sorted(timed_edits.items())
            ],
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
