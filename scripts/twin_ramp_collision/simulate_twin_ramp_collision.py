"""PyBullet simulation for the twin-ramp head-on collision.

Two glass marbles are held at the crests of two facing wooden ramps,
released at the same instant, roll down, and meet head-on on the short flat
valley between the ramp toes. Geometry comes from twin_ramp_geometry so the
Blender scene and the physics world describe the same apparatus.

Output is a trajectory JSON that render_twin_ramp_collision.py replays as
keyframes, plus a `quality` block recording where and when the two balls
actually met -- the one thing about this scene that can silently go wrong.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pybullet as p

sys.path.insert(0, str(Path(__file__).resolve().parent))
from twin_ramp_geometry import build_track  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-sec", type=float, default=3.0)
    parser.add_argument("--substeps", type=int, default=16)
    parser.add_argument(
        "--hold-sec", type=float, default=0.25,
        help="Seconds the balls stay parked at the crests before release.",
    )

    # Two independent constraints fix this ramp, and they pull in opposite
    # directions along the angle.
    #
    # The RISE is capped by the frame rate. Arrival speed at the valley is
    # sqrt(2*g*rise/1.4) for a rolling solid sphere and the two balls close on
    # each other at twice that, so a 14 deg / 0.45 m first cut gave 1.12 m/s --
    # 93 mm of closing per rendered frame while the balls are only 64 mm across.
    # The impact fell entirely between two frames, 72 mm apart in one and
    # already rebounding in the next, so the collision was never on screen.
    # A 78 mm rise gives 0.92 m/s and leaves the last approach frame with the
    # balls 44 mm apart -- still well inside a ball diameter, so the near-touch
    # always lands. Much past 9 deg here and that margin is gone again.
    #
    # The LENGTH is set by the shot. Roll-down time works out to
    # sqrt(2.8*rise/g)/sin(angle): with the rise pinned, time is inversely
    # proportional to sin(angle), and the only way to buy more of it is to go
    # shallower and longer. At 10 deg over 0.33 m the balls met 1.04 s in and
    # everything had settled by 2.3 s, leaving the last third of a 3 s video
    # dead. Lengthening the run to 0.554 m stretches the descent to ~1.0 s and
    # puts the impact at 1.42 s, with the rebound and the second, gentler
    # meeting filling the rest of the clip.
    parser.add_argument("--ramp-angle-deg", type=float, default=8.0)
    parser.add_argument("--ramp-run", type=float, default=0.554)
    parser.add_argument("--ramp-width", type=float, default=0.24)
    parser.add_argument("--ramp-body-thickness", type=float, default=0.10)
    parser.add_argument("--valley-half", type=float, default=0.09)
    parser.add_argument("--plank-thickness", type=float, default=0.018)
    parser.add_argument("--plank-length", type=float, default=1.42)
    parser.add_argument("--plank-width", type=float, default=0.28)

    # A 64 mm glass marble: soda-lime glass at 2500 kg/m^3 over 1.373e-4 m^3.
    # Neither the mass nor the lateral friction actually moves this scenario --
    # rolling down a slope is mass-independent, an equal-mass head-on exchange is
    # too, and the balls roll rather than slide throughout (8 deg needs only
    # mu = 0.04) -- but the ground truth should describe the object on screen.
    parser.add_argument("--ball-radius", type=float, default=0.032)
    parser.add_argument("--ball-mass", type=float, default=0.343)
    parser.add_argument("--ball-friction", type=float, default=0.30)
    parser.add_argument("--ball-restitution", type=float, default=0.87)
    parser.add_argument("--ball-rolling-friction", type=float, default=0.0012)
    parser.add_argument("--ball-spinning-friction", type=float, default=0.004)
    parser.add_argument("--ball-linear-damping", type=float, default=0.0)
    parser.add_argument("--ball-angular-damping", type=float, default=0.0)

    parser.add_argument("--track-friction", type=float, default=0.55)
    parser.add_argument("--track-restitution", type=float, default=0.14)

    parser.add_argument(
        "--release-inset", type=float, default=0.030,
        help="Distance down the sloped face from the crest to the ball's contact point.",
    )
    parser.add_argument(
        "--release-inset-bias", type=float, default=0.004,
        help="Extra inset applied to the -x ball only, so the setup is not perfectly symmetric.",
    )
    parser.add_argument("--gravity-z", type=float, default=-9.8)

    # ---- Per-ball overrides (the PCVE edit surface) -----------------------
    # ball_a is the +X side (rendered purple), ball_b is the -X side (yellow).
    # Left at None each falls back to the corresponding global (--ball-*)
    # value, which reproduces the pre-PCVE behaviour exactly.
    for prefix in ("ball-a", "ball-b"):
        parser.add_argument(f"--{prefix}-mass", type=float, default=None)
        parser.add_argument(f"--{prefix}-friction", type=float, default=None)
        parser.add_argument(f"--{prefix}-restitution", type=float, default=None)
        parser.add_argument(f"--{prefix}-rolling-friction", type=float, default=None)
    # 0 removes that ball from the sim entirely; its frame slot still
    # appears in the output (frozen at its start pose, present=false).
    parser.add_argument("--ball-a-active", type=int, default=1)
    parser.add_argument("--ball-b-active", type=int, default=1)
    return parser.parse_args()


def add_static_box(client, half_extents, position, orientation, friction, restitution):
    shape = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=half_extents, physicsClientId=client,
    )
    body = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=shape,
        baseVisualShapeIndex=-1,
        basePosition=position,
        baseOrientation=orientation,
        physicsClientId=client,
    )
    p.changeDynamics(
        body, -1,
        lateralFriction=float(friction),
        restitution=float(restitution),
        collisionMargin=0.0005,
        physicsClientId=client,
    )
    return body


def simulate(args: argparse.Namespace) -> dict:
    fps = int(args.fps)
    frame_end = max(2, int(round(float(args.duration_sec) * fps)))
    hold_frames = max(0, int(round(float(args.hold_sec) * fps)))
    substeps = int(args.substeps)
    dt = 1.0 / float(fps * substeps)
    radius = float(args.ball_radius)

    track = build_track(
        ramp_angle_deg=args.ramp_angle_deg,
        ramp_run=args.ramp_run,
        valley_half=args.valley_half,
        ramp_width=args.ramp_width,
        ramp_body_thickness=args.ramp_body_thickness,
        plank_thickness=args.plank_thickness,
        plank_length=args.plank_length,
        plank_width=args.plank_width,
    )

    # Side +1 is ball A, side -1 is ball B. B is released a few millimetres
    # further down its ramp: a real pair of ramps is never matched to the
    # micron, and a dead-symmetric release makes the whole shot look like an
    # animation rather than a measurement. 4 mm out of a 557 mm slope shifts the
    # meeting point by only a couple of millimetres, so the balls still meet on
    # the flat valley where the camera is pointed.
    releases = {
        1: track.release_center(1, radius, float(args.release_inset), clearance=0.0002),
        -1: track.release_center(
            -1, radius,
            float(args.release_inset) + float(args.release_inset_bias),
            clearance=0.0002,
        ),
    }

    client = p.connect(p.DIRECT)
    try:
        p.resetSimulation(physicsClientId=client)
        p.setGravity(0.0, 0.0, float(args.gravity_z), physicsClientId=client)
        p.setTimeStep(dt, physicsClientId=client)
        p.setPhysicsEngineParameter(
            fixedTimeStep=dt,
            numSolverIterations=400,
            contactBreakingThreshold=0.0002,
            deterministicOverlappingPairs=1,
            enableConeFriction=1,
            physicsClientId=client,
        )

        # Floor, purely as a catcher: nothing in a correct run ever touches it.
        add_static_box(
            client,
            half_extents=(8.0, 8.0, 0.1),
            position=(0.0, 0.0, -0.1),
            orientation=(0.0, 0.0, 0.0, 1.0),
            friction=0.7,
            restitution=0.1,
        )

        add_static_box(
            client,
            half_extents=(
                track.plank_length / 2.0,
                track.plank_width / 2.0,
                track.plank_thickness / 2.0,
            ),
            position=(0.0, 0.0, track.plank_thickness / 2.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            friction=float(args.track_friction),
            restitution=float(args.track_restitution),
        )

        for side in (1, -1):
            center, pitch = track.ramp_pose(side)
            add_static_box(
                client,
                half_extents=track.ramp_half_extents(),
                position=center,
                orientation=p.getQuaternionFromEuler((0.0, pitch, 0.0)),
                friction=float(args.track_friction),
                restitution=float(args.track_restitution),
            )

        # Resolve per-ball values: side 1 = ball_a, side -1 = ball_b.
        def resolve(prefix: str, key: str) -> float:
            v = getattr(args, f"{prefix}_{key}", None)
            return float(v) if v is not None else float(getattr(args, f"ball_{key}"))

        side_prefix = {1: "ball_a", -1: "ball_b"}
        side_active = {
            1: bool(int(args.ball_a_active)),
            -1: bool(int(args.ball_b_active)),
        }

        ball_shape = p.createCollisionShape(
            p.GEOM_SPHERE, radius=radius, physicsClientId=client,
        )
        ball_ids: dict[int, int | None] = {}
        for side in (1, -1):
            if not side_active[side]:
                ball_ids[side] = None
                continue
            pfx = side_prefix[side]
            body = p.createMultiBody(
                baseMass=resolve(pfx, "mass"),
                baseCollisionShapeIndex=ball_shape,
                baseVisualShapeIndex=-1,
                basePosition=releases[side],
                baseOrientation=(0.0, 0.0, 0.0, 1.0),
                physicsClientId=client,
            )
            p.changeDynamics(
                body, -1,
                lateralFriction=resolve(pfx, "friction"),
                rollingFriction=resolve(pfx, "rolling_friction"),
                spinningFriction=float(args.ball_spinning_friction),
                restitution=resolve(pfx, "restitution"),
                linearDamping=float(args.ball_linear_damping),
                angularDamping=float(args.ball_angular_damping),
                collisionMargin=0.0005,
                physicsClientId=client,
            )
            ball_ids[side] = body

        both_present = ball_ids[1] is not None and ball_ids[-1] is not None

        def ball_gap() -> tuple[float, tuple, tuple]:
            if not both_present:
                # No pair to measure; return a sentinel so the min-gap logic
                # simply never triggers a contact.
                return (float("inf"), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
            pos_a, _ = p.getBasePositionAndOrientation(ball_ids[1], physicsClientId=client)
            pos_b, _ = p.getBasePositionAndOrientation(ball_ids[-1], physicsClientId=client)
            return math.dist(pos_a, pos_b) - 2.0 * radius, pos_a, pos_b

        frames = []
        min_gap = float("inf")
        contact_frame = None
        contact_x = None
        approach_speeds = None
        separation_speeds = None
        left_track = False
        contact_frames = []

        for frame_index in range(1, frame_end + 1):
            touched_this_frame = False
            if frame_index > 1 and frame_index > hold_frames:
                # The impact itself is far shorter than a rendered frame -- at
                # 0.9 m/s the two balls are in contact for well under a
                # millisecond -- so it has to be looked for inside the substep
                # loop. Sampling only at frame boundaries reports the balls as
                # never having touched even in runs where they visibly rebound.
                for _ in range(substeps):
                    p.stepSimulation(physicsClientId=client)
                    if both_present:
                        gap_now, pos_a_now, pos_b_now = ball_gap()
                        if gap_now < min_gap:
                            min_gap = gap_now
                        if p.getContactPoints(
                            bodyA=ball_ids[1], bodyB=ball_ids[-1], physicsClientId=client,
                        ):
                            touched_this_frame = True
                            if contact_frame is None:
                                contact_frame = frame_index
                                contact_x = 0.5 * (pos_a_now[0] + pos_b_now[0])

            record = {
                "frame_index": frame_index,
                "time_sec": (frame_index - 1) / float(fps),
                "released": frame_index > hold_frames,
                "balls": {},
            }
            state: dict[str, tuple] = {}
            for name, side in (("a", 1), ("b", -1)):
                if ball_ids[side] is None:
                    # DELETE edit: emit a frozen slot with present=false.
                    frozen_pos = releases[side]
                    record["balls"][name] = {
                        "side": side,
                        "present": False,
                        "location": list(frozen_pos),
                        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                        "linear_velocity": [0.0, 0.0, 0.0],
                        "angular_velocity": [0.0, 0.0, 0.0],
                        "speed": 0.0,
                        "height_above_track": frozen_pos[2] - track.track_z - radius,
                    }
                    continue
                pos, quat = p.getBasePositionAndOrientation(
                    ball_ids[side], physicsClientId=client,
                )
                lin, ang = p.getBaseVelocity(ball_ids[side], physicsClientId=client)
                speed = math.sqrt(sum(v * v for v in lin))
                state[name] = (pos, lin, speed)
                record["balls"][name] = {
                    "side": side,
                    "present": True,
                    "location": list(pos),
                    "quaternion_xyzw": list(quat),
                    "linear_velocity": list(lin),
                    "angular_velocity": list(ang),
                    "speed": speed,
                    "height_above_track": pos[2] - track.track_z - radius,
                }
                if abs(pos[1]) > track.plank_width / 2.0 or pos[2] < track.track_z - radius:
                    left_track = True

            if "a" in state and "b" in state:
                (pos_a, _, speed_a) = state["a"]
                (pos_b, _, speed_b) = state["b"]
                gap = math.dist(pos_a, pos_b) - 2.0 * radius
            else:
                # One ball is deleted; leave collision-metric fields as
                # sentinels so the quality block downstream reports nothing
                # rather than lying.
                pos_a = pos_b = None
                speed_a = speed_b = 0.0
                gap = float("inf")
            record["gap_between_balls"] = gap
            record["in_contact"] = touched_this_frame
            if gap < min_gap:
                min_gap = gap
            if touched_this_frame:
                contact_frames.append(frame_index)
            if contact_frame == frame_index and frames:
                # Speeds as of the last frame before the impact, which is the
                # last sample where the balls are still purely rolling.
                previous = frames[-1]["balls"]
                approach_speeds = [previous["a"]["speed"], previous["b"]["speed"]]
            if (
                contact_frame is not None
                and separation_speeds is None
                and frame_index >= contact_frame + 4
            ):
                # Four frames on, friction has finished reversing the spin the
                # balls carried through the impact, so this is the speed they
                # actually leave with (see the note on `quality` below).
                separation_speeds = [speed_a, speed_b]

            frames.append(record)

        return {
            "schema_version": 1,
            "simulator": "pybullet",
            "scene": "twin_ramp_collision",
            "fps": fps,
            "frame_start": 1,
            "frame_end": frame_end,
            "hold_frames": hold_frames,
            "duration_sec": float(args.duration_sec),
            "substeps_per_frame": substeps,
            "physics_dt": dt,
            "track": {
                "ramp_angle_deg": float(args.ramp_angle_deg),
                "ramp_run": track.run,
                "ramp_rise": track.rise,
                "slope_length": track.slope_length,
                "ramp_width": track.ramp_width,
                "valley_half": track.valley_half,
                "plank_thickness": track.plank_thickness,
                "plank_length": track.plank_length,
                "plank_width": track.plank_width,
                "track_z": track.track_z,
                "friction": float(args.track_friction),
                "restitution": float(args.track_restitution),
            },
            "objects": {
                "ball_a": {
                    "side": 1,
                    "radius": radius,
                    "mass": resolve("ball_a", "mass"),
                    "friction": resolve("ball_a", "friction"),
                    "restitution": resolve("ball_a", "restitution"),
                    "rolling_friction": resolve("ball_a", "rolling_friction"),
                    "initial_location": list(releases[1]),
                    "present": side_active[1],
                },
                "ball_b": {
                    "side": -1,
                    "radius": radius,
                    "mass": resolve("ball_b", "mass"),
                    "friction": resolve("ball_b", "friction"),
                    "restitution": resolve("ball_b", "restitution"),
                    "rolling_friction": resolve("ball_b", "rolling_friction"),
                    "initial_location": list(releases[-1]),
                    "present": side_active[-1],
                },
                "ball_material": {
                    "friction": float(args.ball_friction),
                    "restitution": float(args.ball_restitution),
                    "rolling_friction": float(args.ball_rolling_friction),
                    "spinning_friction": float(args.ball_spinning_friction),
                },
            },
            # `separation_speeds` comes out far below restitution x approach,
            # and that is correct rather than a solver artefact. Each ball
            # arrives rolling, so it carries angular momentum the head-on
            # impulse does not touch: it rebounds translating backwards while
            # still spinning forwards, and track friction then has to reverse
            # that spin, eating most of the rebound. For a solid sphere the
            # algebra gives a settled rebound of (2.5e - 1)/3.5 of the approach
            # speed, where e is the *effective* ball-on-ball restitution.
            #
            # Note that e is not --ball-restitution: Bullet combines the two
            # bodies' restitutions by multiplying them, so two balls at 0.87
            # collide at e = 0.87^2 = 0.76. That predicts a settled rebound of
            # 0.255x, and the default scenario measures 0.192 against an
            # approach of 0.769 -- 0.250x. Below e = 0.4 the formula goes
            # negative, meaning the pair has no rebound left to give and simply
            # stops dead where it meets; --ball-restitution 0.40 (e = 0.16) is
            # well past that point and does exactly that.
            "quality": {
                "min_gap_between_balls": min_gap,
                "contact_frame": contact_frame,
                "contact_frames": contact_frames,
                "contact_time_sec": (
                    None if contact_frame is None else (contact_frame - 1) / float(fps)
                ),
                "contact_x": contact_x,
                "contact_inside_valley": (
                    None if contact_x is None else abs(contact_x) <= track.valley_half
                ),
                "approach_speeds": approach_speeds,
                "separation_speeds": separation_speeds,
                "left_track": left_track,
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

    q = records["quality"]
    if q["contact_frame"] is None:
        print(f"[WARN] The balls never touched; closest approach {q['min_gap_between_balls']:.4f} m.")
    else:
        print(
            f"[INFO] Contact at frame {q['contact_frame']} "
            f"(t={q['contact_time_sec']:.3f}s), x={q['contact_x']:.4f} m, "
            f"approach speeds {q['approach_speeds'][0]:.3f} / {q['approach_speeds'][1]:.3f} m/s."
        )
        if not q["contact_inside_valley"]:
            print("[WARN] The balls met on a ramp rather than on the flat valley.")
    if q["left_track"]:
        print("[WARN] A ball left the plank.")


if __name__ == "__main__":
    main()
