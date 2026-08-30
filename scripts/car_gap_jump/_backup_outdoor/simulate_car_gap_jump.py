from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pybullet as p


# Geometry matches render_car_gap_jump.py. A sports car speeds along an
# elevated approach road, rides up a take-off ramp at the broken edge of the
# bridge, launches into the air, and either clears the gap and lands on the
# far deck or falls short and plummets into the chasm below. Nothing is
# scripted mid-air: the whole arc is emergent from the car's one initial
# approach speed plus the ramp geometry and gravity. The single thing the
# PCVE suite varies is that launch speed (and the gap width) -- everything
# else stays fixed -- so the outcome flips cleanly between "clears" and
# "falls short."
#
# The car travels along world +X. Box half-extents are a real Nissan GT-R
# footprint (~4.70 m long, 1.90 m wide, 1.34 m tall); the detailed GLB is
# rescaled to match in the render.
CAR_HALF_LENGTH = 2.35   # along travel (+X)
CAR_HALF_WIDTH = 0.95    # across (Y)
CAR_HALF_HEIGHT = 0.67   # vertical (Z)

# Deck (road) surfaces: tops at z = 0, a thick slab hanging below.
DECK_TOP_Z = 0.0
DECK_THICKNESS = 0.6
DECK_WIDTH = 5.0

# Take-off ramp: rises from the approach deck (base at z=0) up to the launch
# lip at x = 0. Local +X is the high (launch) end.
RAMP_ANGLE_DEG = 12.0
RAMP_LENGTH = 13.0       # along the slope; well over 2x the car so it rides up
                         # and pitches cleanly instead of see-sawing on the lip

APPROACH_RUN = 12.0      # flat road length behind the ramp base
FAR_LEN = 45.0           # landing deck length (long, so a fast jump lands on it
                         # and rolls to a stop rather than overshooting the far end)
CHASM_DEPTH = 8.0        # how far below the decks the chasm floor sits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-sec", type=float, default=4.5)
    parser.add_argument("--substeps", type=int, default=48)
    parser.add_argument("--car-mass", type=float, default=1500.0)
    parser.add_argument("--car-restitution", type=float, default=0.1)
    parser.add_argument(
        "--launch-speed",
        type=float,
        default=12.0,
        help="Initial approach speed in m/s along +X. The scene's main knob: "
        "fast enough and the car clears the gap and lands on the far deck; too "
        "slow and it falls short into the chasm.",
    )
    parser.add_argument(
        "--gap-width",
        type=float,
        default=8.0,
        help="Horizontal width of the broken-bridge gap in metres, measured "
        "from the launch lip (x=0) to the near edge of the far landing deck.",
    )
    parser.add_argument(
        "--deck-friction",
        type=float,
        default=0.06,
        help="Lateral friction of the road decks. Low, so the box car proxy "
        "coasts like a car rolling on wheels rather than sliding to a stop.",
    )
    parser.add_argument("--ramp-friction", type=float, default=0.06)
    parser.add_argument("--gravity-z", type=float, default=-9.8)
    return parser.parse_args()


def ramp_geometry(gap_width: float) -> dict:
    angle = math.radians(RAMP_ANGLE_DEG)
    run = RAMP_LENGTH * math.cos(angle)
    rise = RAMP_LENGTH * math.sin(angle)
    # Top (launch lip) at x=0, z=rise; base at x=-run, z=0.
    mid_top = (-run / 2.0, 0.0, rise / 2.0)
    normal = (-math.sin(angle), 0.0, math.cos(angle))  # up out of the slope
    center = tuple(mid_top[i] - (DECK_THICKNESS / 2.0) * normal[i] for i in range(3))
    orientation = p.getQuaternionFromEuler((0.0, -angle, 0.0))
    return {
        "angle": angle,
        "run": run,
        "rise": rise,
        "center": center,
        "orientation": orientation,
        "lip": (0.0, 0.0, rise),
    }


def make_static_box(client, half_extents, position, friction, restitution=0.1, orientation=(0.0, 0.0, 0.0, 1.0)):
    shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=client)
    body = p.createMultiBody(
        0.0, shape, -1, position, baseOrientation=orientation, physicsClientId=client,
    )
    p.changeDynamics(
        body, -1, lateralFriction=friction, restitution=restitution,
        collisionMargin=0.001, physicsClientId=client,
    )
    return body


def simulate(args: argparse.Namespace) -> dict:
    fps = int(args.fps)
    frame_end = max(2, int(round(float(args.duration_sec) * fps)))
    substeps = int(args.substeps)
    dt = 1.0 / float(fps * substeps)
    gap_width = float(args.gap_width)

    ramp = ramp_geometry(gap_width)
    run = ramp["run"]

    # Car starts on the flat approach deck, a few metres behind the ramp base.
    car_start_x = -run - 6.0
    car_start = (car_start_x, 0.0, CAR_HALF_HEIGHT + 0.002)

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

        # Chasm floor far below.
        make_static_box(
            client, (80.0, 80.0, 1.0), (0.0, 0.0, -CHASM_DEPTH - 1.0),
            friction=0.9, restitution=0.0,
        )

        # Approach deck: top at z=0, from x=-(run+APPROACH_RUN) to x=-run.
        approach_center_x = -(run + APPROACH_RUN / 2.0)
        make_static_box(
            client, (APPROACH_RUN / 2.0, DECK_WIDTH / 2.0, DECK_THICKNESS / 2.0),
            (approach_center_x, 0.0, -DECK_THICKNESS / 2.0), friction=float(args.deck_friction),
        )

        # Take-off ramp wedge (tilted about Y so +X is the high launch end).
        make_static_box(
            client, (RAMP_LENGTH / 2.0, DECK_WIDTH / 2.0, DECK_THICKNESS / 2.0),
            ramp["center"], friction=float(args.ramp_friction),
            orientation=ramp["orientation"],
        )

        # Far landing deck: top at z=0, from x=gap_width to x=gap_width+FAR_LEN.
        far_center_x = gap_width + FAR_LEN / 2.0
        make_static_box(
            client, (FAR_LEN / 2.0, DECK_WIDTH / 2.0, DECK_THICKNESS / 2.0),
            (far_center_x, 0.0, -DECK_THICKNESS / 2.0), friction=float(args.deck_friction),
        )

        # Car (dynamic box), flat, nose toward +X.
        car_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=(CAR_HALF_LENGTH, CAR_HALF_WIDTH, CAR_HALF_HEIGHT),
            physicsClientId=client,
        )
        car_id = p.createMultiBody(
            baseMass=float(args.car_mass),
            baseCollisionShapeIndex=car_shape,
            baseVisualShapeIndex=-1,
            basePosition=car_start,
            baseOrientation=(0.0, 0.0, 0.0, 1.0),
            physicsClientId=client,
        )
        p.resetBaseVelocity(
            car_id, linearVelocity=(float(args.launch_speed), 0.0, 0.0),
            angularVelocity=(0.0, 0.0, 0.0), physicsClientId=client,
        )
        p.changeDynamics(
            car_id, -1,
            lateralFriction=float(args.deck_friction),
            spinningFriction=0.005,
            rollingFriction=0.002,
            restitution=float(args.car_restitution),
            linearDamping=0.0,
            angularDamping=0.1,
            collisionMargin=0.0005,
            physicsClientId=client,
        )

        frames = []
        max_height = car_start[2]
        min_height = car_start[2]
        for frame_index in range(1, frame_end + 1):
            if frame_index > 1:
                for _ in range(substeps):
                    p.stepSimulation(physicsClientId=client)
            car_pos, car_quat = p.getBasePositionAndOrientation(car_id, physicsClientId=client)
            car_lin, car_ang = p.getBaseVelocity(car_id, physicsClientId=client)
            max_height = max(max_height, car_pos[2])
            min_height = min(min_height, car_pos[2])
            frames.append({
                "frame_index": frame_index,
                "time_sec": (frame_index - 1) / float(fps),
                "car": {
                    "location": list(car_pos),
                    "quaternion_xyzw": list(car_quat),
                    "linear_velocity": list(car_lin),
                    "angular_velocity": list(car_ang),
                },
            })

        final = frames[-1]["car"]
        final_x = final["location"][0]
        final_z = final["location"][2]
        # World up vector rotated by the car's final orientation; z-component
        # near 1 means it's still upright (wheels down), near/below 0 means
        # it landed on its side or roof.
        fq = final["quaternion_xyzw"]
        up_z = 1.0 - 2.0 * (fq[0] * fq[0] + fq[1] * fq[1])
        far_deck_end = gap_width + FAR_LEN
        # Cleared: rest on the far deck span, at deck level, still upright.
        cleared = (
            gap_width < final_x < far_deck_end
            and final_z > -1.0
            and up_z > 0.7
        )
        fell_into_chasm = final_z < -CHASM_DEPTH + 3.0

        return {
            "schema_version": 1,
            "simulator": "pybullet",
            "fps": fps,
            "frame_start": 1,
            "frame_end": frame_end,
            "duration_sec": float(args.duration_sec),
            "substeps_per_frame": substeps,
            "physics_dt": dt,
            "geometry": {
                "gap_width": gap_width,
                "ramp_angle_deg": RAMP_ANGLE_DEG,
                "ramp_length": RAMP_LENGTH,
                "ramp_run": run,
                "ramp_rise": ramp["rise"],
                "approach_run": APPROACH_RUN,
                "far_len": FAR_LEN,
                "deck_width": DECK_WIDTH,
                "deck_thickness": DECK_THICKNESS,
                "chasm_depth": CHASM_DEPTH,
            },
            "objects": {
                "car": {
                    "half_extents": [CAR_HALF_LENGTH, CAR_HALF_WIDTH, CAR_HALF_HEIGHT],
                    "mass": float(args.car_mass),
                    "launch_speed": float(args.launch_speed),
                    "start_location": list(car_start),
                },
            },
            "quality": {
                "final_x": final_x,
                "final_z": final_z,
                "final_up_z": up_z,
                "max_height": max_height,
                "min_height": min_height,
                "cleared_gap": bool(cleared),
                "fell_into_chasm": bool(fell_into_chasm),
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
