from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pybullet as p


# Geometry matches render_air_hockey_chain.py. Three identical air-hockey
# mallets sit in a line down the middle of an arcade air-hockey table. The one
# at the far end is given a single push along the table; it strikes the second,
# the second strikes the third, and the third slides away toward the near end.
#
# The physical crux is the equal-mass, near-elastic head-on collision. For two
# bodies of equal mass in one dimension, v1' = v1(1-e)/2 and v2' = v1(1+e)/2:
# at e = 1 the striker stops dead and hands its entire speed to the target. Air
# hockey is about as close to that ideal as everyday objects get -- the mallets
# are identical by design, the air cushion makes the table nearly frictionless,
# so almost nothing is lost between collisions, and the moulded plastic faces
# bounce hard. Every one of those three properties is load-bearing: break any
# of them and the chain stops looking like a relay (see the PCVE suite, which
# breaks them one at a time).
#
# Sim frame: x runs down the table's length, y across its width (0 = centre
# line), z up from the playing surface. The render maps this onto the table
# inside the arcade GLB, and with the camera the render sets up, x=0 is the end
# of the table nearest the lens and x=TABLE_LEN the end away from it.

# Mallet dimensions measured off the table model's own mallets, at the scale
# the render places the table in the room (a standard 0.78 m table height):
# a 124 mm disc, 66 mm tall. Physics sees it as a plain cylinder.
MALLET_RADIUS = 0.0620
MALLET_HEIGHT = 0.0656
MALLET_MASS = 0.120        # an air-hockey striker this size, ~120 g

# Playing field, measured off the same model: 1.16 m across by 2.39 m long,
# ringed by a low rail. The rails matter only as a backstop -- the chain runs
# straight down the centre line and never reaches the sides.
TABLE_LEN = 2.391
TABLE_WIDTH = 1.164
RAIL_HEIGHT = 0.06

# Where the three mallets start along the table. Spacing is deliberate: the
# mallets are NOT touching. A sequential-impulse solver resolves a row of
# resting contacts (a Newton's-cradle stack) unreliably -- the impulse can jump
# the whole row in one step, or stall. Leaving a clear gap makes every impact a
# separate two-body event, which is both trustworthy and what the eye reads as
# "one hit the next".
# The three sit on the quarter, half and three-quarter points of the table's
# length, which spreads them as far apart as a three-mallet relay can be while
# still leaving the last one room to run out. The wide gaps are what make each
# handoff legible: the eye gets a clear stretch of coasting between hits rather
# than three impacts on top of each other.
# The relay runs from the end away from the camera back towards it, so the
# chain steps down -x: the struck disc always comes at the viewer rather than
# receding, and each handoff happens nearer the lens and larger in frame than
# the one before it. The push and the spacing carry the same sign.
RELAY_DIRECTION = -1.0
START_X = 3.0 * TABLE_LEN / 4.0
SPACING = RELAY_DIRECTION * TABLE_LEN / 4.0
# The relay runs down the table's centre line. This model's playing surface is
# flat and unobstructed -- the raycast height map over the whole field found
# nothing standing on it but the two mallets -- so nothing has to be dodged.
RELAY_Y = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-sec", type=float, default=1.5)
    parser.add_argument(
        "--substeps",
        type=int,
        default=160,
        help="Physics substeps per rendered frame. This is not a quality dial "
        "to be turned down: below about 120 the impacts land several substeps "
        "deep, Bullet applies restitution against an already partly corrected "
        "velocity, and the collision quietly turns inelastic -- the striker "
        "keeps a fifth of its speed instead of a few percent, at some push "
        "speeds and not others. See the README.",
    )
    parser.add_argument(
        "--settle-sec",
        type=float,
        default=0.25,
        help="Time the mallets are left to settle onto the surface before the "
        "push is applied and recording starts. Without it they are still "
        "micro-bouncing off their initial drop when the first impact lands, so "
        "the contact normal is tilted and the collision is oblique: the striker "
        "keeps a fifth of its speed instead of a few percent, and how bad it "
        "gets depends on where the impact happens to fall between bounces.",
    )
    parser.add_argument(
        "--push-speed",
        type=float,
        default=1.0,
        help="Speed in m/s given to the first mallet along the relay direction "
        "at frame 1. The "
        "scene's main knob: it sets how fast the whole relay runs and how far "
        "the last mallet gets.",
    )
    parser.add_argument(
        "--mallet-restitution",
        type=float,
        default=0.95,
        help="Restitution of each mallet. PyBullet multiplies the two bodies' "
        "values, so mallet-on-mallet is this squared (~0.90 by default), which "
        "leaves the striker with (1-e)/2 = 5%% of its speed -- it visibly stops "
        "dead. Drop this and the pair starts moving off together instead.",
    )
    parser.add_argument(
        "--table-restitution",
        type=float,
        default=0.10,
        help="Restitution of the playing surface. Kept low on purpose: because "
        "PyBullet multiplies restitutions, a low value here damps the mallets' "
        "vertical settling without touching the mallet-on-mallet bounce.",
    )
    parser.add_argument(
        "--surface-friction",
        type=float,
        default=0.06,
        help="Lateral friction of the playing surface. Multiplied by the "
        "mallet's own friction, this is the air cushion: the effective "
        "coefficient is a few thousandths, so a mallet coasts the length of "
        "the table with almost no loss.",
    )
    parser.add_argument("--mallet-friction", type=float, default=0.06)
    parser.add_argument(
        "--middle-mass-scale",
        type=float,
        default=1.0,
        help="Multiplies the middle mallet's mass. At 1.0 the masses are equal "
        "and the collision is a clean velocity swap; heavier and the striker "
        "rebounds backwards instead of stopping.",
    )
    # --- Per-mallet overrides (the PCVE edit surface) ------------------------
    # The three globals above set every mallet at once, which is what the
    # batch sweeps want. A PCVE edit changes exactly one object, so each
    # property also has a per-mallet list form that wins when given. Left at
    # None the list is filled from the corresponding global, so the defaults
    # and every existing caller are unchanged.
    parser.add_argument(
        "--mallet-masses",
        nargs=3,
        type=float,
        default=None,
        help="Per-mallet mass in kg, in relay order (blue, red, white). "
        "Overrides MALLET_MASS and --middle-mass-scale when given.",
    )
    parser.add_argument(
        "--mallet-restitutions",
        nargs=3,
        type=float,
        default=None,
        help="Per-mallet restitution, in relay order. Overrides "
        "--mallet-restitution when given. Remember PyBullet multiplies the "
        "pair, so changing one mallet only affects the impacts it is in.",
    )
    parser.add_argument(
        "--mallet-frictions",
        nargs=3,
        type=float,
        default=None,
        help="Per-mallet lateral friction, in relay order. Overrides "
        "--mallet-friction when given.",
    )
    parser.add_argument(
        "--mallet-active",
        nargs=3,
        type=int,
        default=(1, 1, 1),
        help="Which mallets exist, in relay order. A 0 removes that mallet "
        "from the simulation entirely; its frames still appear in the output "
        "(frozen at its start position, active=false) so consumers keep a "
        "fixed three-slot layout.",
    )
    parser.add_argument("--gravity-z", type=float, default=-9.8)
    return parser.parse_args()


def make_static_box(client, half_extents, position, friction, restitution):
    shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=client)
    body = p.createMultiBody(0.0, shape, -1, position, physicsClientId=client)
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
    push = float(args.push_speed)

    # Resolve the per-mallet properties. The list forms win; where they are
    # absent every mallet takes the global value (and the middle one its mass
    # scale), which reproduces the pre-PCVE behaviour exactly.
    active = tuple(bool(int(v)) for v in args.mallet_active)
    if args.mallet_masses is not None:
        masses = tuple(float(v) for v in args.mallet_masses)
    else:
        masses = (
            MALLET_MASS,
            MALLET_MASS * float(args.middle_mass_scale),
            MALLET_MASS,
        )
    if args.mallet_restitutions is not None:
        restitutions = tuple(float(v) for v in args.mallet_restitutions)
    else:
        restitutions = (float(args.mallet_restitution),) * 3
    if args.mallet_frictions is not None:
        frictions = tuple(float(v) for v in args.mallet_frictions)
    else:
        frictions = (float(args.mallet_friction),) * 3

    start_positions = [
        (START_X + index * SPACING, RELAY_Y, MALLET_HEIGHT / 2.0) for index in range(3)
    ]

    client = p.connect(p.DIRECT)
    try:
        p.resetSimulation(physicsClientId=client)
        p.setGravity(0.0, 0.0, float(args.gravity_z), physicsClientId=client)
        p.setTimeStep(dt, physicsClientId=client)
        p.setPhysicsEngineParameter(
            fixedTimeStep=dt,
            numSolverIterations=200,
            contactBreakingThreshold=0.0005,
            deterministicOverlappingPairs=1,
            enableConeFriction=1,
            # Below this impact speed Bullet ignores restitution entirely and
            # the contact goes inelastic. The default is high enough to kill
            # the later, slower hits in the chain -- the relay would visibly
            # degrade into shoving. Set it under the slowest impact we expect.
            restitutionVelocityThreshold=0.005,
            # Separate the penetration-recovery push from the restitution
            # impulse. Without it, recovering from an overlap feeds velocity
            # back into the bounce and the relay picks up or loses energy
            # depending on how deep the discs happened to overlap.
            useSplitImpulse=1,
            splitImpulsePenetrationThreshold=-0.002,
            physicsClientId=client,
        )

        # Playing surface: a slab whose top face is z=0.
        make_static_box(
            client, (TABLE_LEN / 2.0, TABLE_WIDTH / 2.0, 0.05),
            (TABLE_LEN / 2.0, 0.0, -0.05),
            friction=float(args.surface_friction), restitution=float(args.table_restitution),
        )
        # Rails around the field.
        rail = RAIL_HEIGHT / 2.0
        for sy in (-1.0, 1.0):
            make_static_box(
                client, (TABLE_LEN / 2.0, 0.02, rail),
                (TABLE_LEN / 2.0, sy * (TABLE_WIDTH / 2.0 + 0.02), rail),
                friction=0.2, restitution=float(args.table_restitution),
            )
        for ex in (-0.02, TABLE_LEN + 0.02):
            make_static_box(
                client, (0.02, TABLE_WIDTH / 2.0, rail), (ex, 0.0, rail),
                friction=0.2, restitution=float(args.table_restitution),
            )

        mallet_shape = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=MALLET_RADIUS, height=MALLET_HEIGHT,
            physicsClientId=client,
        )
        # A removed mallet gets no body at all, and its slot holds None. The
        # output keeps all three slots either way, so the renderer and the
        # ground truth always see the same three-mallet layout and only have to
        # read the active flag.
        mallets: list[int | None] = []
        for index in range(3):
            if not active[index]:
                mallets.append(None)
                continue
            body = p.createMultiBody(
                baseMass=masses[index],
                baseCollisionShapeIndex=mallet_shape,
                baseVisualShapeIndex=-1,
                basePosition=start_positions[index],
                baseOrientation=(0.0, 0.0, 0.0, 1.0),
                physicsClientId=client,
            )
            p.changeDynamics(
                body, -1,
                lateralFriction=frictions[index],
                spinningFriction=0.0005,
                rollingFriction=0.0,
                restitution=restitutions[index],
                linearDamping=0.0,
                angularDamping=0.2,
                collisionMargin=0.001,
                physicsClientId=client,
            )
            mallets.append(body)

        # Let everything come to rest before anything moves, then freeze out the
        # residual jitter so all three start from an identical, exactly level
        # state -- the head-on collision this scene is about only stays head-on
        # if the discs are flat on the surface when they meet.
        for _ in range(int(round(float(args.settle_sec) / dt))):
            p.stepSimulation(physicsClientId=client)
        for index, body in enumerate(mallets):
            if body is None:
                continue
            pos, _quat = p.getBasePositionAndOrientation(body, physicsClientId=client)
            start = start_positions[index]
            p.resetBasePositionAndOrientation(
                body, (start[0], start[1], pos[2]), (0.0, 0.0, 0.0, 1.0),
                physicsClientId=client,
            )
            p.resetBaseVelocity(
                body, linearVelocity=(0.0, 0.0, 0.0), angularVelocity=(0.0, 0.0, 0.0),
                physicsClientId=client,
            )

        if mallets[0] is not None:
            p.resetBaseVelocity(
                mallets[0], linearVelocity=(RELAY_DIRECTION * push, 0.0, 0.0),
                angularVelocity=(0.0, 0.0, 0.0),
                physicsClientId=client,
            )

        frames = []
        # Peak speed each mallet ever reaches, and the speed each one is left
        # with once the chain has passed it -- together these are what "the
        # striker stopped dead and handed everything over" actually means.
        peak_speed = [0.0, 0.0, 0.0]
        started = [False, False, False]
        for frame_index in range(1, frame_end + 1):
            if frame_index > 1:
                for _ in range(substeps):
                    p.stepSimulation(physicsClientId=client)
            record = {"frame_index": frame_index, "time_sec": (frame_index - 1) / float(fps)}
            for index, body in enumerate(mallets):
                if body is None:
                    record[f"mallet_{index}"] = {
                        "active": False,
                        "location": list(start_positions[index]),
                        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                        "linear_velocity": [0.0, 0.0, 0.0],
                        "angular_velocity": [0.0, 0.0, 0.0],
                    }
                    continue
                pos, quat = p.getBasePositionAndOrientation(body, physicsClientId=client)
                lin, ang = p.getBaseVelocity(body, physicsClientId=client)
                speed = math.hypot(lin[0], lin[1])
                peak_speed[index] = max(peak_speed[index], speed)
                if speed > 0.05:
                    started[index] = True
                record[f"mallet_{index}"] = {
                    "active": True,
                    "location": list(pos),
                    "quaternion_xyzw": list(quat),
                    "linear_velocity": list(lin),
                    "angular_velocity": list(ang),
                }
            frames.append(record)

        # Relay quality is measured at the moment of each handoff, not on the
        # last frame. The table is nearly frictionless, so the last mallet
        # eventually reaches the end rail and rebounds back into the others; by
        # the final frame the positions say nothing about the original chain.
        # What matters is what each striker was left with the instant the next
        # one took off.
        def first_moving_frame(index):
            for record in frames:
                lin = record[f"mallet_{index}"]["linear_velocity"]
                if math.hypot(lin[0], lin[1]) > 0.10:
                    return record["frame_index"]
            return None

        handoff_frame = [first_moving_frame(i) for i in range(3)]
        speed_after_handoff = []
        retained_fraction = []
        for index in range(2):
            frame_index = handoff_frame[index + 1]
            if frame_index is None:
                speed_after_handoff.append(None)
                retained_fraction.append(None)
                continue
            lin = frames[frame_index - 1][f"mallet_{index}"]["linear_velocity"]
            speed = math.hypot(lin[0], lin[1])
            speed_after_handoff.append(speed)
            retained_fraction.append(speed / peak_speed[index] if peak_speed[index] else None)

        final = frames[-1]
        final_speed = [
            math.hypot(*final[f"mallet_{i}"]["linear_velocity"][:2]) for i in range(3)
        ]
        final_x = [final[f"mallet_{i}"]["location"][0] for i in range(3)]
        # A clean relay: every mallet still in the scene moved, each striker was
        # left with only a few percent of its speed, and the last one carried
        # most of the push. A removed mallet is not a failure to relay, so it is
        # excluded rather than counted as a mallet that never started.
        relay_completed = all(s for s, a in zip(started, active) if a)
        exchange_efficiency = peak_speed[2] / push if push else 0.0
        strikers_stopped = all(
            r is not None and r < 0.20 for r in retained_fraction
        )
        # A striker that rebounds backwards has ended up moving the wrong way:
        # the signature of hitting something heavier than itself.
        rebounded = any(
            frames[handoff_frame[i + 1] - 1][f"mallet_{i}"]["linear_velocity"][0]
            * RELAY_DIRECTION < -0.05
            for i in range(2)
            if handoff_frame[i + 1] is not None
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
            "geometry": {
                "table_len": TABLE_LEN,
                "table_width": TABLE_WIDTH,
                "mallet_radius": MALLET_RADIUS,
                "mallet_height": MALLET_HEIGHT,
                "start_x": START_X,
                "spacing": SPACING,
                "relay_y": RELAY_Y,
            },
            "objects": {
                "mallet_masses": list(masses),
                "mallet_restitutions": list(restitutions),
                "mallet_frictions": list(frictions),
                "mallet_active": [int(a) for a in active],
                "push_speed": push,
                "surface_friction": float(args.surface_friction),
                "table_restitution": float(args.table_restitution),
                # PyBullet multiplies the two bodies' restitutions, so the
                # number that governs each impact is the product of the pair,
                # not either mallet's own value.
                "pair_restitution": [
                    restitutions[0] * restitutions[1],
                    restitutions[1] * restitutions[2],
                ],
            },
            "quality": {
                "peak_speed": peak_speed,
                "handoff_frame": handoff_frame,
                "speed_after_handoff": speed_after_handoff,
                "retained_fraction": retained_fraction,
                "final_speed": final_speed,
                "final_x": final_x,
                "relay_completed": bool(relay_completed),
                "exchange_efficiency": exchange_efficiency,
                "strikers_stopped": bool(strikers_stopped),
                "striker_rebounded": bool(rebounded),
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
