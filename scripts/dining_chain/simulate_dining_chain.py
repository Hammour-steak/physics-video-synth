from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pybullet as p


# A three-object sliding chain on a dining tabletop: a cola can is given a
# horizontal push, slides into a soda cup, and the cup in turn slides into a
# milk carton. Everything is emergent from the can's single initial velocity
# (no per-object impulse). The objects are low, flat-based drink containers
# so they *slide* rather than roll or topple; the tabletop friction is low
# enough that each hit passes momentum to the next but high enough that
# nobody slides off the far edge. Coordinates are the real modern dining-room
# world frame (dining_room__kichen_baked.glb): tabletop surface at z=0.778,
# the chain laid out along +Y at x=1.15.
TABLE_TOP_Z = 0.778
CHAIN_X = 1.15

# Collision proxies (metres). Base masses default to these; a PCVE edit can
# override any one via --object-masses.
CAN = {"kind": "cyl", "r": 0.034, "h": 0.122, "mass": 0.36, "start_y": -1.20}
CUP = {"kind": "cyl", "r": 0.044, "h": 0.160, "mass": 0.30, "start_y": -0.70}
MILK = {"kind": "box", "hx": 0.027, "hy": 0.028, "hz": 0.0625, "mass": 0.35, "start_y": -0.20}

ORDER = ("can", "cup", "milk")
SPECS = {"can": CAN, "cup": CUP, "milk": MILK}
DEFAULT_MASSES = (CAN["mass"], CUP["mass"], MILK["mass"])
DEFAULT_FRICTION = 0.30
DEFAULT_RESTITUTION = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-sec", type=float, default=3.0)
    parser.add_argument("--substeps", type=int, default=40)

    # ---- Scene-wide knobs (kept for back-compat with old callers) ----------
    parser.add_argument(
        "--launch-speed", type=float, default=3.3,
        help="Initial +Y push given to the can (m/s). Overridden by --can-initial-speed.",
    )
    parser.add_argument(
        "--table-friction", type=float, default=0.30,
        help="Tabletop lateral friction. PyBullet multiplies it with each object's friction.",
    )
    parser.add_argument("--object-friction", type=float, default=DEFAULT_FRICTION,
                        help="Global object lateral friction (fallback when per-object list not given).")
    parser.add_argument("--restitution", type=float, default=DEFAULT_RESTITUTION,
                        help="Global object AND table restitution (fallback).")
    parser.add_argument("--gravity-z", type=float, default=-9.8)

    # ---- Per-object overrides (the PCVE edit surface) ----------------------
    # Order is always (can, cup, milk).
    parser.add_argument(
        "--object-masses", nargs=3, type=float, default=None,
        help="Per-object mass in kg, in chain order (can, cup, milk). "
        "Overrides the built-in defaults when given.",
    )
    parser.add_argument(
        "--object-frictions", nargs=3, type=float, default=None,
        help="Per-object lateral friction, in chain order. Overrides --object-friction.",
    )
    parser.add_argument(
        "--object-restitutions", nargs=3, type=float, default=None,
        help="Per-object restitution, in chain order. Overrides --restitution "
        "for the objects (the table's own restitution stays at --restitution).",
    )
    parser.add_argument(
        "--object-active", nargs=3, type=int, default=(1, 1, 1),
        help="Which chain objects exist, in chain order. A 0 removes that object "
        "from the simulation entirely; its frames still appear in the output "
        "(frozen at its start position, active=false) so consumers keep a "
        "fixed three-slot layout.",
    )
    parser.add_argument(
        "--can-initial-speed", type=float, default=None,
        help="Initial +Y speed of the can in m/s. Overrides --launch-speed when given. "
        "Only the can has a non-zero baseline velocity.",
    )
    return parser.parse_args()


def up_axis_z(quat_xyzw) -> float:
    x, y, z, w = quat_xyzw
    return 1.0 - 2.0 * (x * x + y * y)


def _resolve_lists(args: argparse.Namespace):
    if args.object_masses is not None:
        masses = tuple(float(v) for v in args.object_masses)
    else:
        masses = DEFAULT_MASSES
    if args.object_frictions is not None:
        frictions = tuple(float(v) for v in args.object_frictions)
    else:
        frictions = (float(args.object_friction),) * 3
    if args.object_restitutions is not None:
        restitutions = tuple(float(v) for v in args.object_restitutions)
    else:
        restitutions = (float(args.restitution),) * 3
    active = tuple(bool(int(v)) for v in args.object_active)
    can_push = float(args.can_initial_speed) if args.can_initial_speed is not None else float(args.launch_speed)
    return masses, frictions, restitutions, active, can_push


def simulate(args: argparse.Namespace) -> dict:
    fps = int(args.fps)
    frame_end = max(2, int(round(float(args.duration_sec) * fps)))
    substeps = int(args.substeps)
    dt = 1.0 / float(fps * substeps)

    masses, frictions, restitutions, active, can_push = _resolve_lists(args)

    client = p.connect(p.DIRECT)
    try:
        p.resetSimulation(physicsClientId=client)
        p.setGravity(0.0, 0.0, float(args.gravity_z), physicsClientId=client)
        p.setTimeStep(dt, physicsClientId=client)
        p.setPhysicsEngineParameter(
            fixedTimeStep=dt, numSolverIterations=200,
            deterministicOverlappingPairs=1, enableConeFriction=1,
            physicsClientId=client,
        )

        table_shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=(4.0, 4.0, 0.4), physicsClientId=client,
        )
        table_id = p.createMultiBody(
            0.0, table_shape, -1, (0.0, 0.0, TABLE_TOP_Z - 0.4), physicsClientId=client,
        )
        p.changeDynamics(
            table_id, -1, lateralFriction=float(args.table_friction),
            restitution=float(args.restitution), physicsClientId=client,
        )

        def make_cyl(spec, mass):
            shape = p.createCollisionShape(
                p.GEOM_CYLINDER, radius=spec["r"], height=spec["h"], physicsClientId=client,
            )
            z = TABLE_TOP_Z + spec["h"] / 2.0 + 0.001
            return p.createMultiBody(
                mass, shape, -1, (CHAIN_X, spec["start_y"], z), physicsClientId=client,
            ), z

        def make_box(spec, mass):
            shape = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=(spec["hx"], spec["hy"], spec["hz"]),
                physicsClientId=client,
            )
            z = TABLE_TOP_Z + spec["hz"] + 0.001
            return p.createMultiBody(
                mass, shape, -1, (CHAIN_X, spec["start_y"], z), physicsClientId=client,
            ), z

        bodies: dict[str, int | None] = {}
        start_z: dict[str, float] = {}
        for index, name in enumerate(ORDER):
            spec = SPECS[name]
            if not active[index]:
                bodies[name] = None
                start_z[name] = TABLE_TOP_Z + (spec["h"] / 2.0 if spec["kind"] == "cyl" else spec["hz"]) + 0.001
                continue
            if spec["kind"] == "cyl":
                body, z = make_cyl(spec, masses[index])
            else:
                body, z = make_box(spec, masses[index])
            bodies[name] = body
            start_z[name] = z
            p.changeDynamics(
                body, -1, lateralFriction=frictions[index],
                spinningFriction=0.02, rollingFriction=0.002,
                restitution=restitutions[index], physicsClientId=client,
            )

        if bodies["can"] is not None:
            p.resetBaseVelocity(
                bodies["can"], linearVelocity=(0.0, can_push, 0.0),
                angularVelocity=(0.0, 0.0, 0.0), physicsClientId=client,
            )

        start_y = {name: SPECS[name]["start_y"] for name in ORDER}
        frames = []
        for frame_index in range(1, frame_end + 1):
            if frame_index > 1:
                for _ in range(substeps):
                    p.stepSimulation(physicsClientId=client)
            entry = {"frame_index": frame_index, "time_sec": (frame_index - 1) / float(fps), "objects": {}}
            for name in ORDER:
                body = bodies[name]
                if body is None:
                    z = start_z[name]
                    entry["objects"][name] = {
                        "active": False,
                        "location": [CHAIN_X, start_y[name], z],
                        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                        "linear_velocity": [0.0, 0.0, 0.0],
                        "angular_velocity": [0.0, 0.0, 0.0],
                    }
                    continue
                pos, quat = p.getBasePositionAndOrientation(body, physicsClientId=client)
                lin, ang = p.getBaseVelocity(body, physicsClientId=client)
                entry["objects"][name] = {
                    "active": True,
                    "location": list(pos),
                    "quaternion_xyzw": list(quat),
                    "linear_velocity": list(lin),
                    "angular_velocity": list(ang),
                }
            frames.append(entry)

        quality = {}
        for name in ORDER:
            last = frames[-1]["objects"][name]
            quality[name] = {
                "displacement_y": last["location"][1] - start_y[name],
                "final_y": last["location"][1],
                "final_up_z": up_axis_z(last["quaternion_xyzw"]),
                "final_speed": math.hypot(last["linear_velocity"][0], last["linear_velocity"][1]),
            }
        moving_active = [name for i, name in enumerate(ORDER) if active[i] and name != "can"]
        chain_ok = all(quality[name]["displacement_y"] > 0.03 for name in moving_active)

        return {
            "schema_version": 1,
            "simulator": "pybullet",
            "fps": fps,
            "frame_start": 1,
            "frame_end": frame_end,
            "duration_sec": float(args.duration_sec),
            "substeps_per_frame": substeps,
            "physics_dt": dt,
            "table_top_z": TABLE_TOP_Z,
            "chain_x": CHAIN_X,
            "objects_geometry": {"can": CAN, "cup": CUP, "milk": MILK},
            "object_order": list(ORDER),
            "params": {
                "launch_speed": can_push,
                "can_initial_speed": can_push,
                "table_friction": float(args.table_friction),
                "restitution": float(args.restitution),
                "object_masses": list(masses),
                "object_frictions": list(frictions),
                "object_restitutions": list(restitutions),
                "object_active": [int(a) for a in active],
            },
            "quality": {**quality, "chain_ok": chain_ok, "start_z": start_z, "start_y": start_y},
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
    print(f"[sim] chain_ok={q['chain_ok']}  "
          f"can dY={q['can']['displacement_y']:+.2f}  "
          f"cup dY={q['cup']['displacement_y']:+.2f}  "
          f"milk dY={q['milk']['displacement_y']:+.2f} (up_z {q['milk']['final_up_z']:+.2f})")


if __name__ == "__main__":
    main()
