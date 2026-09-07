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


# Geometry matches render_car_ramp_climb.py. A toy car is given a fixed
# running-start push up an inclined ramp; how far it climbs (and whether it
# holds its ground once it stops, or slides back down) depends entirely on
# the ramp surface's friction -- everything else (launch speed, mass, ramp
# angle) stays identical across renders. This is a direct synthetic
# instance of the "same push, different surface -> different outcome"
# friction-comparison scenario (four toy-car-up-a-ramp panels with
# different surface materials side by side).
#
# The ramp is tilted about world Y by `ramp_angle_deg`, following the same
# convention as ramp_collision's book ramp: local -X is the high end, local
# +X is the low end (base, resting on the floor). The car starts at the low
# end and is launched toward local -X (up-slope). Box half-extents match
# toy_car_ball's real-world car dimensions.
CAR_HALF_WIDTH = 0.0504
CAR_HALF_LENGTH = 0.11685
CAR_HALF_HEIGHT = 0.03255

# -90 degree yaw about Z, matching toy_car_ball: the car's local -Y (nose)
# ends up pointing toward local -X once combined with the ramp's own tilt
# below, i.e. facing up-slope.
CAR_YAW_QUAT_XYZW = (0.0, 0.0, -0.70710678, 0.70710678)

FLOOR_Z = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-sec", type=float, default=3.0)
    parser.add_argument("--substeps", type=int, default=60)
    parser.add_argument("--car-mass", type=float, default=0.35)
    parser.add_argument("--car-restitution", type=float, default=0.05)
    parser.add_argument("--ramp-angle-deg", type=float, default=20.0)
    parser.add_argument("--ramp-length", type=float, default=0.9)
    parser.add_argument("--ramp-width", type=float, default=0.35)
    parser.add_argument("--ramp-thickness", type=float, default=0.03)
    parser.add_argument(
        "--ramp-friction",
        type=float,
        default=0.25,
        help="The single parameter this scene varies across renders. The "
        "default (0.25, matching the 'asphalt_grey' surface) is low enough "
        "that the car actually launches off the top of the ramp with the "
        "default launch_speed, rather than just climbing partway.",
    )
    parser.add_argument(
        "--floor-friction",
        type=float,
        default=0.9,
        help="Fixed friction of the ground beyond the ramp's base -- unlike "
        "the ramp surface, this doesn't vary between renders, so a car that "
        "slides back down (or lands after launching off the top) "
        "decelerates and stops instead of coasting indefinitely.",
    )
    parser.add_argument(
        "--launch-speed",
        type=float,
        default=2.7,
        help="Initial up-slope push speed (m/s). At the default ramp_friction "
        "(0.25) the car launches off the ramp top; 2.7 is fast enough that it "
        "completes its forward rotation in the air and lands flat on its "
        "underside (bottom-down) close to the ramp, whereas at 2.6 it "
        "under-rotates and settles inverted on its roof. The four-surface "
        "PCVE suite pins its own lower launch speed (2.6; see "
        "build_pcve_car_ramp_climb.py).",
    )
    parser.add_argument("--gravity-z", type=float, default=-9.8)
    # Edits that land partway through the clip, as
    # [{"frame": n, "params": {physics_key: new_value, ...}}]. Everything runs
    # on the CLI values until frame n, where params are written into the live
    # simulation and the run carries on from the state it had reached.
    parser.add_argument("--timed-edits-json", type=Path, default=None)
    return parser.parse_args()


def apply_timed_params(client: int, params: dict, *, bodies: dict) -> None:
    """Write one frame's worth of edited physics into the live simulation.

    ``bodies`` carries the car and the ramp it climbs. The scene's `car.friction` knob is the ramp surface's coefficient -- Bullet multiplies the pair, so the grip under the wheels is the same thing either way. Anything this scene's edit vocabulary cannot
    produce raises: an edit that is silently dropped renders as a video that
    looks like the baseline and nothing downstream would catch it.
    """
    fields = {"ramp_friction": ("ramp", "lateralFriction"),
              "car_mass": ("car", "mass"),
              "car_friction": ("car", "lateralFriction"),
              "car_restitution": ("car", "restitution")}
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

    ramp_angle = math.radians(float(args.ramp_angle_deg))
    ramp_length = float(args.ramp_length)
    ramp_width = float(args.ramp_width)
    ramp_thickness = float(args.ramp_thickness)
    cos_a = math.cos(ramp_angle)
    sin_a = math.sin(ramp_angle)

    # Ramp center height places the low-end bottom corner on the floor.
    ramp_center_z = FLOOR_Z + ramp_length / 2 * sin_a + ramp_thickness / 2 * cos_a

    # Car starts at the low end (local +X), resting on the ramp surface.
    car_local_x = ramp_length / 2 - CAR_HALF_LENGTH - 0.01
    car_local_z = ramp_thickness / 2 + CAR_HALF_HEIGHT + 0.0005
    car_start_x = car_local_x * cos_a + car_local_z * sin_a
    car_start_z = -car_local_x * sin_a + car_local_z * cos_a + ramp_center_z

    # Climbing (up-slope) direction unit vector in world space.
    climb_dir = (-cos_a, 0.0, sin_a)
    launch_speed = float(args.launch_speed)
    car_velocity = tuple(v * launch_speed for v in climb_dir)

    ramp_orientation = p.getQuaternionFromEuler((0.0, ramp_angle, 0.0))

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
            p.GEOM_BOX, halfExtents=(8.0, 8.0, 0.1), physicsClientId=client,
        )
        floor_id = p.createMultiBody(
            0.0, floor_shape, -1, (0.0, 0.0, FLOOR_Z - 0.1), physicsClientId=client,
        )
        p.changeDynamics(
            floor_id, -1, lateralFriction=float(args.floor_friction),
            restitution=0.1, physicsClientId=client,
        )

        ramp_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=(ramp_length / 2.0, ramp_width / 2.0, ramp_thickness / 2.0),
            physicsClientId=client,
        )
        ramp_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=ramp_shape,
            baseVisualShapeIndex=-1,
            basePosition=(0.0, 0.0, ramp_center_z),
            baseOrientation=ramp_orientation,
            physicsClientId=client,
        )
        p.changeDynamics(
            ramp_id, -1, lateralFriction=float(args.ramp_friction),
            restitution=0.05, collisionMargin=0.001, physicsClientId=client,
        )

        car_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=(CAR_HALF_WIDTH, CAR_HALF_LENGTH, CAR_HALF_HEIGHT),
            physicsClientId=client,
        )
        _, car_orientation = p.multiplyTransforms(
            (0.0, 0.0, 0.0), ramp_orientation, (0.0, 0.0, 0.0), CAR_YAW_QUAT_XYZW,
        )
        car_id = p.createMultiBody(
            baseMass=float(args.car_mass),
            baseCollisionShapeIndex=car_shape,
            baseVisualShapeIndex=-1,
            basePosition=(car_start_x, 0.0, car_start_z),
            baseOrientation=car_orientation,
            physicsClientId=client,
        )
        p.resetBaseVelocity(
            car_id, linearVelocity=car_velocity, angularVelocity=(0.0, 0.0, 0.0),
            physicsClientId=client,
        )
        p.changeDynamics(
            car_id, -1,
            lateralFriction=float(args.ramp_friction),
            spinningFriction=0.01,
            rollingFriction=0.002,
            restitution=float(args.car_restitution),
            linearDamping=0.0,
            angularDamping=0.05,
            collisionMargin=0.0005,
            physicsClientId=client,
        )

        frames = []
        car_min_local_x = car_local_x

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
                apply_timed_params(client, timed_edits[frame_index], bodies=dict(car=car_id, ramp=ramp_id))

            car_pos, car_quat = p.getBasePositionAndOrientation(car_id, physicsClientId=client)
            car_lin, car_ang = p.getBaseVelocity(car_id, physicsClientId=client)

            world_x = car_pos[0]
            world_z_rel = car_pos[2] - ramp_center_z
            local_x = world_x * cos_a - world_z_rel * sin_a
            car_min_local_x = min(car_min_local_x, local_x)

            frames.append({
                "frame_index": frame_index,
                "time_sec": (frame_index - 1) / float(fps),
                "car": {
                    "location": list(car_pos),
                    "quaternion_xyzw": list(car_quat),
                    "linear_velocity": list(car_lin),
                    "angular_velocity": list(car_ang),
                    "ramp_local_x": local_x,
                },
            })

        final_local_x = frames[-1]["car"]["ramp_local_x"]
        final_speed = math.sqrt(sum(v * v for v in frames[-1]["car"]["linear_velocity"]))
        top_edge_local_x = -(ramp_length / 2.0 - CAR_HALF_LENGTH)
        reached_top = car_min_local_x <= top_edge_local_x
        slid_back_down = (final_local_x - car_min_local_x) > 0.02

        return {
            "schema_version": 1,
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
                "center_z": ramp_center_z,
            },
            "objects": {
                "car": {
                    "half_extents": [CAR_HALF_WIDTH, CAR_HALF_LENGTH, CAR_HALF_HEIGHT],
                    "mass": float(args.car_mass),
                    "launch_speed": launch_speed,
                    "start_location": [car_start_x, 0.0, car_start_z],
                },
            },
            "timed_edits": [
                {"frame": frame, "params": params}
                for frame, params in sorted(timed_edits.items())
            ],
            "quality": {
                "car_min_local_x_reached": car_min_local_x,
                "car_final_local_x": final_local_x,
                "car_final_speed": final_speed,
                "reached_top": reached_top,
                "slid_back_down": slid_back_down,
                "climb_distance": car_local_x - car_min_local_x,
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
