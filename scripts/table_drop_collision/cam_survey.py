"""Render the drop from a set of candidate cameras, for choosing one.

Each candidate is rendered at the same four moments -- the ball sitting on the
table, the frame it leaves the lip, the contact, and both balls at rest -- so the
options can be compared on the three things that decide this shot: whether the
roll across the table reads at all, whether the fall and the impact happen in
clear space rather than under the table's overhang, and whether both balls are
still in frame once they have stopped.

    python3 scripts/table_drop_collision/cam_survey.py \\
        --out-root renders/tdc_camera_survey --round angle
    python3 scripts/table_drop_collision/cam_survey.py \\
        --out-root renders/tdc_distance_survey --round distance

The output is disposable; delete the directory once a framing is chosen.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


WORKSPACE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_BLENDER = WORKSPACE_DIR / "tools" / "blender-3.6.23-linux-x64" / "blender"
RENDER_SCRIPT = Path(__file__).with_name("render_table_drop_collision.py")

# --- Round one: the angle ------------------------------------------------------
#
# Every candidate stands south of the table, because that is the only side there
# is. The sofa's front face is 0.356 m off the table's north rim, so there is no
# room for a camera behind it and nothing to see from there anyway.
#
# The shot is laid out to run *across* the camera rather than at it: the ball
# rolls west, leaves by the west lip, and everything after it happens further
# west still. A first pass had it rolling south and falling toward the lens, and
# 67 mm of ball coming straight at a camera two metres away reads as nothing at
# all -- no fall, no impact, just a dot that gets slightly bigger.
#
# Three hard walls bound the rest. The room shell's south face is at y = -2.458,
# so nothing can stand further back than about 2.1 m from the action without
# shooting through it. The floor plane stops at x = +1.111 and the east wall at
# +1.454, leaving about a metre of standing room on that side. West it is open to
# the glazed wall at x = -1.88.
#
# The action runs from the launch at (0.315, -0.290, 0.516) to the struck ball's
# rest at (-1.130, -0.128): 1.445 m across the frame, with the lip at x = -0.342
# and the contact at x = -0.722 in the middle of it. Every candidate has to hold
# all of that, and the last frame is the demanding one -- the two balls come to
# rest 0.81 m apart and their separation is the content.
ACTION_TARGET = (-0.35, -0.32, 0.20)
STANDARD_FRAMES = (1, 15, 22, 67)

ANGLE_CANDIDATES = [
    ("S1_square_low", (-0.35, -2.20, 0.78), ACTION_TARGET, 40.0, None, None,
     "Square to the action, knee height. Table top edge-on as a horizon."),
    ("S2_square_mid", (-0.35, -2.20, 1.15), ACTION_TARGET, 40.0, None, None,
     "Square, seated height. More table surface, so more of the roll."),
    ("S3_square_tele", (-0.35, -2.36, 0.95), ACTION_TARGET, 55.0, None, None,
     "Square at 55 mm: tighter crop, flatter perspective, camera not moved in."),
    ("S4_quarter_east_low", (0.34, -2.12, 0.78), ACTION_TARGET, 40.0, None, None,
     "Three-quarter from the east, low. Camera behind the launch, action opening out."),
    ("S5_quarter_east_mid", (0.34, -2.12, 1.18), ACTION_TARGET, 40.0, None, None,
     "Three-quarter from the east, seated height."),
    ("S6_quarter_west_low", (-1.05, -2.10, 0.78), ACTION_TARGET, 40.0, None, None,
     "Three-quarter from the west, low. Camera ahead of the action, table receding."),
    ("S7_quarter_west_mid", (-1.05, -2.10, 1.18), ACTION_TARGET, 40.0, None, None,
     "Three-quarter from the west, seated height."),
    ("S8_east_corner", (0.92, -1.88, 0.98), ACTION_TARGET, 40.0, None, None,
     "From the room's south-east corner, sofa and window across the back."),
    ("S9_west_corner", (-1.58, -1.78, 0.98), ACTION_TARGET, 40.0, None, None,
     "From the south-west, with the glazed wall close on the left."),
    ("S10_close_wide", (-0.36, -1.52, 0.70), (-0.34, -0.28, 0.22), 28.0, None, None,
     "Three-quarters of the standoff at 28 mm. Balls larger, room stretched."),
    ("S11_ball_height", (-0.35, -1.95, 0.42), (-0.36, -0.30, 0.16), 35.0, None, None,
     "Six ball diameters off the floor. The floor as a horizon, the table above it."),
    ("S12_standing", (-0.35, -1.95, 1.38), ACTION_TARGET, 35.0, None, None,
     "What somebody standing over it sees. Most table surface, flattest balls."),
]

# --- Round two: the distance ---------------------------------------------------
#
# S1 was chosen, and then, as always, asked to come closer. There is a hard limit
# on how close it can get: at 40 mm on a 36 mm sensor the frame covers 0.9 times
# the subject distance, S1 stands 1.97 m off and the action is 1.512 m wide with
# the balls counted, so S1 is already holding it with 13 per cent to spare. Twelve
# per cent closer and both ends are touching the frame edge.
#
# So the ladder below moves *two* levers, not one. The camera comes in, and the
# run-up is shortened to give it room -- because the run-up is the one part of
# this shot that can be shortened for nothing at all. The lip is fixed by the
# table, so pulling the launch west and raising the push to compensate reproduces
# the lip speed to a thousandth and everything downstream with it: at a 0.502 m
# run-up instead of 0.657 the landing point is identical, the contact normal
# moves 0.1 deg, and the two balls come to rest within 18 mm of where they did.
# Moving the subject is nearly always the cheaper fix; the camera position is
# load-bearing and the run-up is not.
#
# The frame numbers move with it, which is why they are per candidate: a shorter
# run-up means the ball reaches the lip sooner, and the survey has to sample the
# lip and the contact, not fixed times.
DISTANCE_CANDIDATES = [
    ("D0_chosen", (-0.35, -2.20, 0.78), ACTION_TARGET, 40.0, None, (1, 15, 22, 67),
     "S1 exactly as chosen. The baseline everything else is judged against."),
    ("D1_in_12", (-0.35, -1.98, 0.77), ACTION_TARGET, 40.0, None, (1, 15, 22, 67),
     "12 per cent closer, run-up untouched. Both ends of the action at the frame edge."),
    ("D2_in_20_short", (-0.35, -1.83, 0.76), ACTION_TARGET, 40.0,
     {"launch_x": 0.160, "launch_speed": 1.230}, (1, 12, 19, 67),
     "20 per cent closer with the run-up cut to 0.50 m. Same lip speed, same landing."),
    ("D3_in_32_short", (-0.35, -1.62, 0.75), ACTION_TARGET, 40.0,
     {"launch_x": 0.050, "launch_speed": 1.194}, (1, 10, 17, 67),
     "32 per cent closer, run-up cut to 0.39 m. The push starts near the table's centre."),
    ("D4_in_32_wide", (-0.35, -1.62, 0.75), ACTION_TARGET, 35.0,
     {"launch_x": 0.160, "launch_speed": 1.230}, (1, 12, 19, 67),
     "Same standoff as D3 at 35 mm, keeping the longer run-up. Wider, more room distortion."),
    ("D5_long_50", (-0.35, -2.20, 0.78), ACTION_TARGET, 50.0,
     {"launch_x": 0.050, "launch_speed": 1.194}, (1, 10, 17, 67),
     "S1's own position at 50 mm with the short run-up. Tighter crop, camera not moved."),
]

ROUNDS = {"angle": ANGLE_CANDIDATES, "distance": DISTANCE_CANDIDATES}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", choices=tuple(ROUNDS), default="angle")
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument(
        "--frames", nargs="+", type=int, default=None,
        help="Override the per-candidate frames. By default each candidate is "
        "sampled on the table, leaving the lip, at the contact, and at rest, and "
        "the middle two move when a candidate shortens the run-up.",
    )
    parser.add_argument("--resolution", nargs=2, type=int, default=(720, 405))
    parser.add_argument("--samples", type=int, default=56)
    parser.add_argument("--duration-sec", type=float, default=2.8)
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--blender", type=Path, default=None)
    parser.add_argument("--only", nargs="+", default=None,
                        help="Render only these candidate names.")
    return parser.parse_args()


def resolve_blender(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    env = os.environ.get("BLENDER_BIN")
    if env:
        return Path(env)
    if DEFAULT_BLENDER.exists():
        return DEFAULT_BLENDER
    found = shutil.which("blender")
    if not found:
        raise RuntimeError("Cannot find Blender; pass --blender or set BLENDER_BIN.")
    return Path(found)


def main() -> None:
    args = parse_args()
    blender = resolve_blender(args.blender)
    out_root = args.out_root or (
        WORKSPACE_DIR / "renders" / f"tdc_{args.round}_survey")
    out_root.mkdir(parents=True, exist_ok=True)
    wanted = set(args.only) if args.only else None

    manifest = {"round": args.round, "candidates": []}
    for name, location, target, lens, physics, frames, note in ROUNDS[args.round]:
        if wanted is not None and name not in wanted:
            continue
        frames = tuple(args.frames or frames or STANDARD_FRAMES)
        work_dir = out_root / name
        work_dir.mkdir(parents=True, exist_ok=True)
        overrides: dict = {"camera": {
            "location": list(location), "target": list(target), "lens_mm": lens,
        }}
        if physics:
            overrides["physics"] = dict(physics)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(overrides, handle)
            overrides_path = Path(handle.name)
        try:
            command = [
                str(blender), "-b", "--python", str(RENDER_SCRIPT), "--",
                "--mode", "preview",
                "--out-dir", str(work_dir),
                "--resolution", str(args.resolution[0]), str(args.resolution[1]),
                "--samples", str(args.samples),
                "--duration-sec", str(args.duration_sec),
                "--device", args.device,
                "--preview-frames", *[str(f) for f in frames],
                "--scenario-overrides-json", str(overrides_path),
            ]
            print(f"[SURVEY] {name}: {note}", flush=True)
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                print(result.stdout[-2000:])
                print(result.stderr[-2000:])
                raise RuntimeError(f"{name} failed with exit code {result.returncode}")
        finally:
            overrides_path.unlink(missing_ok=True)

        for frame in frames:
            src = work_dir / f"preview_f{frame:03d}.png"
            if src.exists():
                shutil.copy2(src, out_root / f"{name}_f{frame:03d}.png")

        # The render script saves a .blend beside every output, and this model's
        # textures are packed into it. A dozen of those is a couple of gigabytes
        # of survey for a few megabytes of the stills anyone actually looks at,
        # and nothing here needs a scene file: the framing is three numbers in the
        # manifest, and re-rendering a candidate is one --only away.
        for blend in work_dir.glob("*.blend"):
            blend.unlink()
        manifest["candidates"].append({
            "name": name, "location": list(location), "target": list(target),
            "lens_mm": lens, "physics": physics, "frames": list(frames),
            "note": note,
        })

    (out_root / "survey_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    print(f"[INFO] Survey written to {out_root}")


if __name__ == "__main__":
    main()
