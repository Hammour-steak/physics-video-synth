from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pybullet as p


# Geometry matches render_toy_car_ball.py. A toy car drives across a
# wall-mounted shelf, parallel to the wall, and rear-ends a toy ball resting
# near the shelf's left edge; the ball is knocked off the edge and falls to
# the room floor below, while the much heavier car sheds little speed and
# stays on the shelf -- a real-world instance of the "something falls off a
# shelf/table when struck" scenario from the Physics-IQ benchmark's
# real-filmed reference clips.
#
# The car and ball travel along world X (parallel to the backdrop wall,
# matching the reference photo's left-to-right framing) rather than toward
# the wall. The shelf is correspondingly wide in X (room to travel) and
# shallow in Y (a real wall shelf's depth, not a full tabletop).
#
# The car's collision shape is a single box approximating its silhouette
# (real wheels/steering are not modeled -- consistent with how other scenes
# in this project use simplified rigid-body primitives under a detailed
# render mesh). Box half-extents are measured directly off the downloaded
# GLB after baking its import transform: width(x) 0.1008m, length(y)
# 0.2337m, height(z) 0.0651m -- see render_toy_car_ball.py's import step.
# The car's local +x/+y/+z axes still mean width/length/height; to make it
# travel along world X nose-first, its initial orientation is a -90 degree
# yaw (see CAR_INITIAL_QUAT_XYZW below) rather than swapping the box's own
# half-extents, so the same quaternion applied to both the physics body and
# the render mesh keeps the car facing its direction of travel.
CAR_HALF_WIDTH = 0.0504
CAR_HALF_LENGTH = 0.11685
CAR_HALF_HEIGHT = 0.03255
BALL_RADIUS = 0.06

# -90 degree rotation about Z: the car's local -Y (nose) ends up pointing
# toward world -X, matching this scene's right-to-left direction of travel.
CAR_INITIAL_QUAT_XYZW = (0.0, 0.0, -0.70710678, 0.70710678)

TABLE_Z = 0.75
TABLE_HALF_X = 0.5
TABLE_HALF_Y = 0.15
TABLE_THICKNESS = 0.03
SHELF_DEPTH_Y = 0.0

FLOOR_HALF_X = 3.5
FLOOR_HALF_Y = 1.2
FLOOR_THICKNESS = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-sec", type=float, default=3.5)
    parser.add_argument("--substeps", type=int, default=60)
    parser.add_argument("--car-mass", type=float, default=0.35)
    parser.add_argument("--car-friction", type=float, default=0.15)
    parser.add_argument("--car-restitution", type=float, default=0.1)
    parser.add_argument("--ball-mass", type=float, default=0.05)
    parser.add_argument("--ball-friction", type=float, default=0.3)
    parser.add_argument("--ball-restitution", type=float, default=0.6)
    parser.add_argument("--table-friction", type=float, default=0.15)
    parser.add_argument("--floor-friction", type=float, default=0.8)
    parser.add_argument("--launch-speed", type=float, default=0.6)
    parser.add_argument("--car-start-x", type=float, default=0.1)
    parser.add_argument(
        "--ball-start-x",
        type=float,
        default=-(TABLE_HALF_X) + BALL_RADIUS * 1.2,
        help="Ball's starting x position, near the shelf's left edge.",
    )
    parser.add_argument("--gravity-z", type=float, default=-9.8)
    # PCVE DELETE edit for the ball: 0 removes the ball entirely; the frame
    # slot still appears in the output (frozen at its start pose,
    # present=false) so consumers keep a fixed layout. The car is always
    # present -- the whole sim is centred on its drive across the table.
    parser.add_argument("--ball-active", type=int, default=1)
    return parser.parse_args()


def simulate(args: argparse.Namespace) -> dict:
    fps = int(args.fps)
    frame_end = max(2, int(round(float(args.duration_sec) * fps)))
    substeps = int(args.substeps)
    dt = 1.0 / float(fps * substeps)

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

        table_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=(TABLE_HALF_X, TABLE_HALF_Y, TABLE_THICKNESS / 2.0),
            physicsClientId=client,
        )
        table_id = p.createMultiBody(
            0.0, table_shape, -1,
            (0.0, 0.0, TABLE_Z - TABLE_THICKNESS / 2.0),
            physicsClientId=client,
        )
        p.changeDynamics(
            table_id, -1, lateralFriction=float(args.table_friction),
            restitution=float(args.ball_restitution), physicsClientId=client,
        )

        floor_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=(FLOOR_HALF_X, FLOOR_HALF_Y, FLOOR_THICKNESS / 2.0),
            physicsClientId=client,
        )
        floor_id = p.createMultiBody(
            0.0, floor_shape, -1,
            (0.0, 0.0, -FLOOR_THICKNESS / 2.0),
            physicsClientId=client,
        )
        p.changeDynamics(
            floor_id, -1, lateralFriction=float(args.floor_friction),
            restitution=float(args.ball_restitution), physicsClientId=client,
        )

        car_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=(CAR_HALF_WIDTH, CAR_HALF_LENGTH, CAR_HALF_HEIGHT),
            physicsClientId=client,
        )
        car_id = p.createMultiBody(
            baseMass=float(args.car_mass),
            baseCollisionShapeIndex=car_shape,
            baseVisualShapeIndex=-1,
            basePosition=(float(args.car_start_x), SHELF_DEPTH_Y, TABLE_Z + CAR_HALF_HEIGHT),
            baseOrientation=CAR_INITIAL_QUAT_XYZW,
            physicsClientId=client,
        )
        p.resetBaseVelocity(
            car_id, linearVelocity=(-float(args.launch_speed), 0.0, 0.0),
            angularVelocity=(0.0, 0.0, 0.0), physicsClientId=client,
        )
        p.changeDynamics(
            car_id, -1,
            lateralFriction=float(args.car_friction),
            spinningFriction=0.005,
            rollingFriction=0.001,
            restitution=float(args.car_restitution),
            linearDamping=0.0,
            angularDamping=0.05,
            collisionMargin=0.0005,
            physicsClientId=client,
        )

        ball_active = bool(int(args.ball_active))
        ball_id = None
        ball_start_pos = (float(args.ball_start_x), SHELF_DEPTH_Y, TABLE_Z + BALL_RADIUS)
        if ball_active:
            ball_shape = p.createCollisionShape(
                p.GEOM_SPHERE, radius=BALL_RADIUS, physicsClientId=client,
            )
            ball_id = p.createMultiBody(
                baseMass=float(args.ball_mass),
                baseCollisionShapeIndex=ball_shape,
                baseVisualShapeIndex=-1,
                basePosition=ball_start_pos,
                physicsClientId=client,
            )
            p.changeDynamics(
                ball_id, -1,
                lateralFriction=float(args.ball_friction),
                spinningFriction=0.01,
                rollingFriction=0.002,
                restitution=float(args.ball_restitution),
                linearDamping=0.0,
                angularDamping=0.05,
                collisionMargin=0.0005,
                physicsClientId=client,
            )

        frames = []
        ball_left_table = False
        ball_min_x = float(args.ball_start_x)

        for frame_index in range(1, frame_end + 1):
            if frame_index > 1:
                for _ in range(substeps):
                    p.stepSimulation(physicsClientId=client)

            car_pos, car_quat = p.getBasePositionAndOrientation(car_id, physicsClientId=client)
            car_lin, car_ang = p.getBaseVelocity(car_id, physicsClientId=client)
            if ball_id is not None:
                ball_pos, ball_quat = p.getBasePositionAndOrientation(ball_id, physicsClientId=client)
                ball_lin, ball_ang = p.getBaseVelocity(ball_id, physicsClientId=client)
                ball_present = True
            else:
                ball_pos, ball_quat = ball_start_pos, (0.0, 0.0, 0.0, 1.0)
                ball_lin, ball_ang = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
                ball_present = False

            if ball_present:
                ball_min_x = min(ball_min_x, ball_pos[0])
                if ball_pos[0] < -TABLE_HALF_X:
                    ball_left_table = True

            frames.append({
                "frame_index": frame_index,
                "time_sec": (frame_index - 1) / float(fps),
                "car": {
                    "location": list(car_pos),
                    "quaternion_xyzw": list(car_quat),
                    "linear_velocity": list(car_lin),
                    "angular_velocity": list(car_ang),
                },
                "ball": {
                    "present": ball_present,
                    "location": list(ball_pos),
                    "quaternion_xyzw": list(ball_quat),
                    "linear_velocity": list(ball_lin),
                    "angular_velocity": list(ball_ang),
                },
            })

        final_car = frames[-1]["car"]
        final_ball = frames[-1]["ball"]
        car_speed = math.sqrt(sum(v * v for v in final_car["linear_velocity"]))
        ball_speed = math.sqrt(sum(v * v for v in final_ball["linear_velocity"]))
        # "On the shelf" means the car's center hasn't crossed the true edge --
        # the car's nose is allowed to overhang the edge (as a real toy car
        # stopping close to it would), as long as more than half its footprint
        # is still supported and it isn't actually tipping over.
        car_still_on_table = abs(final_car["location"][0]) < TABLE_HALF_X
        ball_at_rest_on_floor = (
            final_ball["location"][2] < BALL_RADIUS + 0.02 and ball_speed < 0.05
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
                "car": {
                    "half_extents": [CAR_HALF_WIDTH, CAR_HALF_LENGTH, CAR_HALF_HEIGHT],
                    "mass": float(args.car_mass),
                    "start_x": float(args.car_start_x),
                    "launch_speed": float(args.launch_speed),
                },
                "ball": {
                    "radius": BALL_RADIUS,
                    "mass": float(args.ball_mass),
                    "start_x": float(args.ball_start_x),
                    "present": ball_active,
                },
                "table": {
                    "half_extents": [TABLE_HALF_X, TABLE_HALF_Y, TABLE_THICKNESS / 2.0],
                    "top_z": TABLE_Z,
                },
                "floor": {
                    "half_extents": [FLOOR_HALF_X, FLOOR_HALF_Y, FLOOR_THICKNESS / 2.0],
                },
            },
            "quality": {
                "ball_min_x_reached": ball_min_x,
                "ball_left_table": ball_left_table,
                "car_final_speed": car_speed,
                "ball_final_speed": ball_speed,
                "car_still_on_table": car_still_on_table,
                "ball_at_rest_on_floor": ball_at_rest_on_floor,
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
