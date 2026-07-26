from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import pybullet as p


@dataclass(frozen=True)
class DominoChainConfig:
    fps: int = 24
    duration_sec: float = 8.0
    substeps: int = 24
    dimensions: tuple[float, float, float] = (0.18, 0.58, 1.18)
    spacing: float = 0.68
    first_tilt_deg: float = 12.0
    mass: float = 0.36
    floor_friction: float = 0.90
    domino_friction: float = 0.90
    restitution: float = 0.04
    linear_damping: float = 0.03
    angular_damping: float = 0.025


def supported_center_height(
    *,
    dimensions: tuple[float, float, float],
    pitch_rad: float,
) -> float:
    """Height of a pitched box center when its lowest edge touches z=0."""
    thickness, _width, height = (float(value) for value in dimensions)
    return 0.5 * (
        height * abs(math.cos(float(pitch_rad)))
        + thickness * abs(math.sin(float(pitch_rad)))
    )


def _tilt_degrees(quaternion_xyzw: tuple[float, float, float, float]) -> float:
    up = p.rotateVector(quaternion_xyzw, (0.0, 0.0, 1.0))
    cosine = max(-1.0, min(1.0, float(up[2])))
    return float(math.degrees(math.acos(cosine)))


def _event_record(*, elapsed_steps: int, config: DominoChainConfig) -> dict[str, float]:
    time_sec = float(elapsed_steps) / float(config.fps * config.substeps)
    return {
        "frame_index": 1.0 + time_sec * float(config.fps),
        "time_sec": time_sec,
    }


def simulate(config: DominoChainConfig) -> dict[str, Any]:
    fps = max(1, int(config.fps))
    substeps = max(1, int(config.substeps))
    frame_end = max(2, int(round(float(config.duration_sec) * fps)))
    fixed_dt = 1.0 / float(fps * substeps)
    dimensions = tuple(float(value) for value in config.dimensions)
    half_extents = tuple(0.5 * value for value in dimensions)
    positions_x = (-float(config.spacing), 0.0, float(config.spacing))

    client = p.connect(p.DIRECT)
    try:
        p.resetSimulation(physicsClientId=client)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=client)
        p.setTimeStep(fixed_dt, physicsClientId=client)
        p.setPhysicsEngineParameter(
            fixedTimeStep=fixed_dt,
            numSolverIterations=200,
            contactBreakingThreshold=0.0015,
            deterministicOverlappingPairs=1,
            physicsClientId=client,
        )

        floor_shape = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=client)
        floor_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=floor_shape,
            baseVisualShapeIndex=-1,
            basePosition=(0.0, 0.0, 0.0),
            physicsClientId=client,
        )
        p.changeDynamics(
            floor_id,
            -1,
            lateralFriction=float(config.floor_friction),
            restitution=0.0,
            physicsClientId=client,
        )

        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=client,
        )
        domino_ids: list[int] = []
        for index, x_position in enumerate(positions_x):
            pitch = math.radians(float(config.first_tilt_deg)) if index == 0 else 0.0
            orientation = p.getQuaternionFromEuler((0.0, pitch, 0.0))
            center_height = supported_center_height(
                dimensions=dimensions,
                pitch_rad=pitch,
            )
            domino_id = p.createMultiBody(
                baseMass=float(config.mass),
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=-1,
                basePosition=(x_position, 0.0, center_height + 0.001),
                baseOrientation=orientation,
                physicsClientId=client,
            )
            p.resetBaseVelocity(
                domino_id,
                linearVelocity=(0.0, 0.0, 0.0),
                angularVelocity=(0.0, 0.0, 0.0),
                physicsClientId=client,
            )
            p.changeDynamics(
                domino_id,
                -1,
                lateralFriction=float(config.domino_friction),
                spinningFriction=0.03,
                rollingFriction=0.01,
                restitution=float(config.restitution),
                linearDamping=float(config.linear_damping),
                angularDamping=float(config.angular_damping),
                physicsClientId=client,
            )
            domino_ids.append(domino_id)

        contact_events: dict[str, dict[str, float] | None] = {
            "domino_1_to_domino_2": None,
            "domino_2_to_domino_3": None,
        }
        max_tilt_deg = [0.0, 0.0, 0.0]
        frames: list[dict[str, Any]] = []
        elapsed_steps = 0

        for frame_index in range(1, frame_end + 1):
            if frame_index > 1:
                for _ in range(substeps):
                    p.stepSimulation(physicsClientId=client)
                    elapsed_steps += 1
                    for pair_index, event_name in enumerate(contact_events):
                        if contact_events[event_name] is not None:
                            continue
                        if p.getContactPoints(
                            domino_ids[pair_index],
                            domino_ids[pair_index + 1],
                            physicsClientId=client,
                        ):
                            contact_events[event_name] = _event_record(
                                elapsed_steps=elapsed_steps,
                                config=config,
                            )

            frame_record: dict[str, Any] = {
                "frame_index": int(frame_index),
                "time_sec": float(frame_index - 1) / float(fps),
            }
            for index, domino_id in enumerate(domino_ids, start=1):
                location, quaternion = p.getBasePositionAndOrientation(
                    domino_id,
                    physicsClientId=client,
                )
                linear_velocity, angular_velocity = p.getBaseVelocity(
                    domino_id,
                    physicsClientId=client,
                )
                aabb_min, _aabb_max = p.getAABB(
                    domino_id,
                    physicsClientId=client,
                )
                tilt = _tilt_degrees(quaternion)
                max_tilt_deg[index - 1] = max(max_tilt_deg[index - 1], tilt)
                prefix = f"domino_{index}"
                frame_record.update(
                    {
                        f"{prefix}_location": list(location),
                        f"{prefix}_quaternion_xyzw": list(quaternion),
                        f"{prefix}_linear_velocity": list(linear_velocity),
                        f"{prefix}_angular_velocity": list(angular_velocity),
                        f"{prefix}_floor_gap": float(aabb_min[2]),
                        f"{prefix}_tilt_deg": float(tilt),
                    }
                )
            frame_record["domino_1_domino_2_contact"] = bool(
                p.getContactPoints(
                    domino_ids[0], domino_ids[1], physicsClientId=client
                )
            )
            frame_record["domino_2_domino_3_contact"] = bool(
                p.getContactPoints(
                    domino_ids[1], domino_ids[2], physicsClientId=client
                )
            )
            frames.append(frame_record)

        first_contact = contact_events["domino_1_to_domino_2"]
        second_contact = contact_events["domino_2_to_domino_3"]
        ordered_chain_complete = bool(
            first_contact is not None
            and second_contact is not None
            and float(first_contact["time_sec"]) < float(second_contact["time_sec"])
            and all(value > 65.0 for value in max_tilt_deg)
        )
        return {
            "schema_version": 1,
            "simulation": "three_domino_gravity_chain",
            "config": asdict(config),
            "frame_start": 1,
            "frame_end": int(frame_end),
            "fps": int(fps),
            "initial_state": {
                "first_domino_tilt_deg": float(config.first_tilt_deg),
                "linear_velocity_is_zero": True,
                "angular_velocity_is_zero": True,
                "external_impulse_applied": False,
            },
            "contact_events": contact_events,
            "summary": {
                "ordered_chain_complete": bool(ordered_chain_complete),
                "max_tilt_deg": [float(value) for value in max_tilt_deg],
            },
            "frames": frames,
        }
    finally:
        p.disconnect(physicsClientId=client)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-sec", type=float, default=8.0)
    parser.add_argument("--substeps", type=int, default=24)
    parser.add_argument("--spacing", type=float, default=0.68)
    parser.add_argument("--first-tilt-deg", type=float, default=12.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = simulate(
        DominoChainConfig(
            fps=int(args.fps),
            duration_sec=float(args.duration_sec),
            substeps=int(args.substeps),
            spacing=float(args.spacing),
            first_tilt_deg=float(args.first_tilt_deg),
        )
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
