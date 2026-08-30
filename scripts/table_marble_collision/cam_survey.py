"""Render the marble collision from a set of candidate cameras, for choosing one.

Each candidate is rendered at the same four moments -- the launch, the contact,
two frames after it, and both marbles at rest -- so the options can be compared
on the things that matter: whether the whole 1.58 m of action fits, whether the
impact reads, and whether the two balls are still both in frame at the end.

    python3 scripts/table_marble_collision/cam_survey.py \
        --out-root renders/tmc_camera_survey

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
RENDER_SCRIPT = Path(__file__).with_name("render_table_marble_collision.py")

# Every candidate stands north of the table, because that is the only side there
# is: the table is a peninsula against a wall on its south edge, its east end is
# 67 mm from a wardrobe, and its west end runs into the kitchen worktop.
#
# They also all stand back at least 1.2 m. The action runs from x = -0.83 at the
# launch to x = +0.75 where the struck marble stops, which is 1.58 m, and a
# 32 mm lens on a 36 mm sensor only covers that from about 1.5 m. A first pass
# shot from 0.96 m lost the struck marble off the near edge of frame before it
# stopped.
#
# And they are all low. The subject is two balls 100 and 50 mm across on a
# horizontal surface; from standing height they flatten into dots on a plan, and
# the thing that makes them read as objects rolling on a table is being close to
# the height of the table itself.
CANDIDATES = [
    ("N1_square_low", (-0.02, 1.85, 0.24), (-0.02, 0.00, 0.02), 32.0,
     "Square to the table, low. The whole run left to right, nothing foreshortened."),
    ("N2_quarter_east", (0.75, 1.80, 0.26), (-0.05, 0.00, 0.02), 32.0,
     "Three-quarter from the east. Approach comes toward camera, impact off centre."),
    ("N3_quarter_west", (-0.80, 1.80, 0.26), (0.05, 0.00, 0.02), 32.0,
     "Three-quarter from the west, behind the launch. Balls run away from camera."),
    ("N4_marble_height", (0.10, 1.35, 0.10), (-0.05, 0.00, 0.03), 28.0,
     "Two marble diameters off the table. Table top as a horizon line."),
    ("N5_standing", (0.20, 2.05, 0.55), (-0.05, 0.00, 0.00), 30.0,
     "What somebody standing at the bar sees. Most of the table surface visible."),
    ("N6_room_wide", (1.60, 2.60, 0.75), (-0.25, 0.05, -0.05), 24.0,
     "The whole flat: table, stools, kitchen, floor. Marbles small in frame."),
    ("N7_impact_tight", (0.55, 1.20, 0.20), (0.05, 0.00, 0.03), 50.0,
     "Tight on the impact point. Biggest collision, launch happens off frame."),
    ("N8_along_from_east", (1.55, 1.05, 0.22), (-0.55, 0.02, 0.03), 40.0,
     "Down the table's length from the east end, balls coming at camera."),
    ("N9_along_from_west", (-1.60, 1.00, 0.22), (0.55, 0.00, 0.03), 40.0,
     "Down the length from the kitchen end, balls running away."),
    ("N10_quarter_east_high", (0.85, 1.55, 0.42), (-0.10, 0.00, 0.00), 35.0,
     "Three-quarter from the east, chest height. More table surface, smaller balls."),
    # Second round. N8 had the best depth of the ten -- the table running away
    # from the camera with the kitchen and the window behind it -- but the struck
    # marble rolls toward the camera and left frame before it stopped. Both of
    # these hold the same look and stand further back so that the last frame
    # still has both balls in it.
    ("N8b_along_from_east_back", (1.98, 1.42, 0.26), (-0.45, 0.02, 0.03), 35.0,
     "N8 pulled back east and north. Same depth, struck marble still in frame at rest."),
    ("N2b_quarter_east_back", (0.90, 2.10, 0.27), (-0.10, 0.00, 0.02), 35.0,
     "N2 pulled back. Table across the frame with the kitchen run behind it."),
    # Third round: N2b was chosen, then asked to come in closer. There is a hard
    # limit on how close the framing can get and still hold the whole shot -- the
    # action runs 1.58 m -- so the ladder below trades the launch away
    # progressively rather than pretending it can be had for nothing.
    #
    # The pull-in moves the camera in x and y but *holds* the height near 0.24 m
    # rather than scaling it down the sight line. Scaling it would keep the
    # grazing angle identical, which sounds right and is not: at 0.14 m the
    # camera is a ball and a half off the table, and the two marbles start
    # overlapping each other instead of reading as two objects.
    ("T1_in_mild", (0.70, 1.68, 0.25), (0.00, 0.00, 0.03), 35.0,
     "20 per cent closer. Still holds the launch, less dead wall above."),
    ("T2_in_medium", (0.58, 1.40, 0.24), (0.08, 0.00, 0.03), 35.0,
     "35 per cent closer, recentred on the impact. Launch sits at the frame edge."),
    ("T3_in_tight", (0.45, 1.10, 0.22), (0.22, 0.00, 0.03), 35.0,
     "Half the standoff, framed on contact-to-rest. The run-up starts off frame."),
    ("T4_long_lens", (0.90, 2.10, 0.27), (0.05, 0.00, 0.03), 50.0,
     "N2b's own position at 50 mm: tighter crop, flatter perspective, no move."),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path,
                        default=WORKSPACE_DIR / "renders" / "tmc_camera_survey")
    parser.add_argument(
        "--frames", nargs="+", type=int, default=(2, 18, 24, 44),
        help="Launch, contact, two frames after, and both marbles at rest.",
    )
    parser.add_argument("--resolution", nargs=2, type=int, default=(720, 405))
    parser.add_argument("--samples", type=int, default=56)
    parser.add_argument("--duration-sec", type=float, default=2.4)
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
    args.out_root.mkdir(parents=True, exist_ok=True)
    wanted = set(args.only) if args.only else None

    manifest = {"frames": list(args.frames), "candidates": []}
    for name, location, target, lens, note in CANDIDATES:
        if wanted is not None and name not in wanted:
            continue
        work_dir = args.out_root / name
        work_dir.mkdir(parents=True, exist_ok=True)
        overrides = {"camera": {
            "location": list(location), "target": list(target), "lens_mm": lens,
        }}
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
                "--preview-frames", *[str(f) for f in args.frames],
                "--scenario-overrides-json", str(overrides_path),
            ]
            print(f"[SURVEY] {name}: {note}")
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                print(result.stdout[-2000:])
                print(result.stderr[-2000:])
                raise RuntimeError(f"{name} failed with exit code {result.returncode}")
        finally:
            overrides_path.unlink(missing_ok=True)

        for frame in args.frames:
            src = work_dir / f"preview_f{frame:03d}.png"
            if src.exists():
                shutil.copy2(src, args.out_root / f"{name}_f{frame:03d}.png")

        # The render script saves a .blend beside every output, and this one is
        # 164 MB because the flat's textures are packed into it. Twelve of them is
        # 1.9 GB of survey for 12 MB of the stills anyone actually looks at, and
        # nothing here needs a scene file: the framing is three numbers in the
        # manifest, and re-rendering a candidate is one --only away.
        for blend in work_dir.glob("*.blend"):
            blend.unlink()
        manifest["candidates"].append({
            "name": name, "location": list(location), "target": list(target),
            "lens_mm": lens, "note": note,
        })

    (args.out_root / "survey_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    print(f"[INFO] Survey written to {args.out_root}")


if __name__ == "__main__":
    main()
