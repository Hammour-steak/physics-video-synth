from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pybullet as p


# Geometry matches render_car_gap_jump.py. An indoor tabletop stunt: a 1:24
# die-cast toy sports car rolls along the top of a stack of hardback books
# sitting at the edge of a dining table, shoots off the front of the stack,
# and either clears the gap to a second table of the same height and skids to
# a stop on it, or falls short and drops to the room floor. Nothing is scripted
# mid-air: the whole arc is emergent from the car's one initial push speed
# plus the launch height and gravity. The single thing the PCVE suite varies
# is that push speed (and the gap width) -- everything else stays fixed -- so
# the outcome flips cleanly between "clears" and "falls short."
#
# The car travels along world +X. Box half-extents are a 1:24 Mini Cooper S
# (real car ~3.85 x 1.73 x 1.41 m -> ~16.0 x 7.2 x 6.0 cm); the detailed GLB
# is rescaled to match in the render.
CAR_HALF_LENGTH = 0.080    # along travel (+X)
CAR_HALF_WIDTH = 0.036     # across (Y)
CAR_HALF_HEIGHT = 0.030    # vertical (Z)

# Launch table: top at z = 0, a tabletop slab hanging below. The render adds
# legs down to the room floor; only the slabs matter to physics.
DECK_TOP_Z = 0.0
TABLE_THICKNESS = 0.04
DECK_WIDTH = 0.60
APPROACH_LEN = 1.05      # launch tabletop length; its front edge is at x=0

# The launch pad: a real stack of hardback books
# (assets/models/harry_potter_books_stack.glb) sitting at the table's front
# edge, the front edge of its top book flush with the table edge (x=0). The
# car rolls along that top cover and launches off the front of the stack, so
# the stack's height is what buys the jump its airtime. Physics sees one
# static box whose top face is that cover.
#
# The scanned stack is a shallow wedge -- its base and its top cover are about
# 2 degrees out of parallel -- so the render rests it on its measured base
# plane and then takes the remaining wedge out of the mesh, leaving the cover
# level. That is why this box is a plain axis-aligned one: the surface the car
# rolls on really is horizontal. BOOK_STACK_H is the stack's own thickness, so
# resting the cover there puts its base on the tabletop.
BOOK_STACK_LEN = 0.241     # top book, along the runway
BOOK_STACK_WIDTH = 0.162   # top book, across
BOOK_STACK_H = 0.1490      # cover height above the tabletop
# The stack is set back from the table edge rather than flush with it. Flush
# looks contrived -- the book's front edge and the table's front edge collapse
# into one line, and you cannot read the books as resting *on* the table -- and
# nobody stacks books perfectly level with an edge anyway. The car therefore
# launches off the books, sails over this last strip of tabletop, and only then
# crosses the gap. Even the weakest push in range clears the strip comfortably.
BOOK_STACK_SETBACK = 0.045

# Landing table: a second table of the SAME height as the launch table, across
# the gap. Launching horizontally, the car must drop the book-stack height
# (0.150 m) while crossing the gap -- the stack is the entire reason the jump
# has any range at all.
FAR_LEN = 1.50           # landing tabletop length (long enough that even the
                         # fastest sampled push lands and skids out on it)

CHASM_DEPTH = 0.74       # launch tabletop height above the room floor -- the
                         # "chasm" a short jump drops into


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-sec", type=float, default=1.6)
    parser.add_argument("--substeps", type=int, default=48)
    parser.add_argument("--car-mass", type=float, default=0.5)
    parser.add_argument(
        "--car-restitution",
        type=float,
        default=0.3,
        help="Die-cast body on wood: combined with the table/floor restitution "
        "it gives the small landing bounce a real toy shows.",
    )
    parser.add_argument(
        "--launch-speed",
        type=float,
        default=1.9,
        help="Initial push speed in m/s given to the car on top of the book "
        "stack, along +X. The scene's main knob: fast enough and the car "
        "clears the gap and lands on the bench; too slow and it drops off the "
        "stack into the gap and hits the floor. The clear/fall threshold for "
        "the default 0.40 m gap is around 1.4 m/s.",
    )
    parser.add_argument(
        "--gap-width",
        type=float,
        default=0.28,
        help="Horizontal width of the gap between the two tables in metres, "
        "measured from the launch table's edge (x=0) to the near edge of the "
        "landing table.",
    )
    parser.add_argument(
        "--deck-friction",
        type=float,
        default=0.10,
        help="Lateral friction of the book stack's glossy top cover. Low "
        "(combined with the car's own friction) so the toy coasts on its "
        "free-rolling wheels rather than scrubbing to a stop before the edge.",
    )
    parser.add_argument(
        "--far-deck-friction",
        type=float,
        default=0.5,
        help="Lateral friction of the landing tabletop. Higher than the book "
        "cover: after the jump the car lands slightly askew and skids on its "
        "body/sideways wheels instead of rolling freely, so it stops on the "
        "table.",
    )
    parser.add_argument(
        "--car-friction",
        type=float,
        default=0.45,
        help="The car body's own lateral friction (PyBullet combines it with "
        "each surface's coefficient).",
    )
    parser.add_argument("--gravity-z", type=float, default=-9.8)
    return parser.parse_args()


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

    # Car starts at the back of the top book, flat on the cover, nose toward +X,
    # and is given its whole speed up front (a flick of the hand).
    car_start_x = -BOOK_STACK_SETBACK - BOOK_STACK_LEN + CAR_HALF_LENGTH + 0.015
    car_start = (car_start_x, 0.0, BOOK_STACK_H + CAR_HALF_HEIGHT + 0.002)
    push = float(args.launch_speed)

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
            physicsClientId=client,
        )

        # Room floor far below the tabletops (wood floor: some grip, a little
        # bounce for the dropped-toy thud).
        make_static_box(
            client, (4.0, 4.0, 0.05), (0.3, 0.0, -CHASM_DEPTH - 0.05),
            friction=0.6, restitution=0.3,
        )

        # Launch tabletop: top at z=0, front edge at x=0.
        make_static_box(
            client, (APPROACH_LEN / 2.0, DECK_WIDTH / 2.0, TABLE_THICKNESS / 2.0),
            (-APPROACH_LEN / 2.0, 0.0, -TABLE_THICKNESS / 2.0),
            friction=float(args.deck_friction), restitution=0.1,
        )

        # Book stack: top face at z=BOOK_STACK_H, front face at the setback.
        make_static_box(
            client, (BOOK_STACK_LEN / 2.0, BOOK_STACK_WIDTH / 2.0, BOOK_STACK_H / 2.0),
            (-BOOK_STACK_SETBACK - BOOK_STACK_LEN / 2.0, 0.0, BOOK_STACK_H / 2.0),
            friction=float(args.deck_friction), restitution=0.1,
        )

        # Landing table: same height as the launch table (top at z=0), from
        # x=gap_width to x=gap_width+FAR_LEN.
        far_center_x = gap_width + FAR_LEN / 2.0
        make_static_box(
            client, (FAR_LEN / 2.0, DECK_WIDTH / 2.0, TABLE_THICKNESS / 2.0),
            (far_center_x, 0.0, -TABLE_THICKNESS / 2.0),
            friction=float(args.far_deck_friction), restitution=0.3,
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
            car_id, linearVelocity=(push, 0.0, 0.0),
            angularVelocity=(0.0, 0.0, 0.0), physicsClientId=client,
        )
        p.changeDynamics(
            car_id, -1,
            lateralFriction=float(args.car_friction),
            spinningFriction=0.005,
            rollingFriction=0.002,
            restitution=float(args.car_restitution),
            linearDamping=0.0,
            angularDamping=0.1,
            collisionMargin=0.0003,
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
        # Cleared: rest on the landing-table span, at tabletop level (well
        # above the floor), upright.
        cleared = (
            gap_width < final_x < far_deck_end
            and final_z > -0.2
            and up_z > 0.7
        )
        fell_into_chasm = final_z < -0.4 and final_x < far_deck_end
        # A push far past what the scene is designed for skids all the way
        # across the landing table and off its far end -- a different failure
        # from falling short, so it is reported separately rather than being
        # lumped in with the cars that never made it across.
        overshot_far_end = final_x > far_deck_end

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
                "book_stack_len": BOOK_STACK_LEN,
                "book_stack_width": BOOK_STACK_WIDTH,
                "book_stack_height": BOOK_STACK_H,
                "book_stack_setback": BOOK_STACK_SETBACK,
                "approach_len": APPROACH_LEN,
                "far_len": FAR_LEN,
                "deck_width": DECK_WIDTH,
                "table_thickness": TABLE_THICKNESS,
                "chasm_depth": CHASM_DEPTH,
            },
            "objects": {
                "car": {
                    "half_extents": [CAR_HALF_LENGTH, CAR_HALF_WIDTH, CAR_HALF_HEIGHT],
                    "mass": float(args.car_mass),
                    "launch_speed": push,
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
                "overshot_far_end": bool(overshot_far_end),
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
