"""Render the twin-ramp scene from a set of candidate cameras, for choosing one.

Each candidate is rendered at the same three moments -- the balls parked at the
crests, the last frame before they touch, and the rebound -- so the options can
be compared on the thing that matters, which is whether the impact reads.

    python3 scripts/twin_ramp_collision/survey_cameras.py \
        --out-root renders/trc_camera_survey

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
RENDER_SCRIPT = Path(__file__).with_name("render_twin_ramp_collision.py")

# All candidates sit low. The apparatus is only 76 mm tall and its whole story
# happens in the 180 mm of flat valley in the middle, so a high angle flattens
# the ramps into stripes and turns the collision into two dots meeting.
#
# They also all stand at least a metre back. A first round shot from 0.5-0.9 m
# and every one of them cropped the apparatus: it is 0.84 m from crest to crest,
# which a 50 mm lens on a 36 mm sensor only covers from about 1.2 m, and the
# closest candidates framed nothing but two balls with no track around them.
CANDIDATES = [
    ("R1_corner_mid",   (-0.78, -1.42, 0.36), (0.00, 0.0, 0.050), 40.0,
     "Over the table's front-left corner. Rig, table edge, window and floor all in shot."),
    ("R2_side_mid",     (0.00, -1.45, 0.34), (0.00, 0.0, 0.045), 45.0,
     "Square to the track. The cleanest read of the two slopes and the impact point."),
    ("R3_quarter_low",  (-0.70, -1.30, 0.22), (0.00, 0.0, 0.048), 45.0,
     "Three-quarter, low to the table. Balls near eye level."),
    ("R4_quarter_high", (-0.85, -1.55, 0.52), (0.00, 0.0, 0.030), 40.0,
     "Three-quarter, standing height. Most of the table and the room behind."),
    ("R5_room_wide",    (-1.15, -2.05, 0.62), (0.00, 0.0, -0.02), 32.0,
     "The whole room: table, legs, floor, window. Rig is small in frame."),
    ("R6_end_on",       (-1.45, -0.70, 0.28), (0.05, 0.0, 0.050), 50.0,
     "Along the track from one end."),
    ("R7_valley_tight", (-0.30, -0.95, 0.20), (0.00, 0.0, 0.045), 65.0,
     "Tight on the valley. Biggest impact, crests out of frame."),
    ("R8_side_low",     (0.00, -1.30, 0.15), (0.00, 0.0, 0.048), 50.0,
     "Square to the track and almost at table level."),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path,
                        default=WORKSPACE_DIR / "renders" / "trc_camera_survey")
    parser.add_argument("--frames", nargs="+", type=int, default=(4, 25, 30))
    parser.add_argument("--resolution", nargs=2, type=int, default=(720, 405))
    parser.add_argument("--samples", type=int, default=72)
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--blender", type=Path, default=None)
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

    manifest = {"frames": list(args.frames), "candidates": []}
    for name, location, target, lens, note in CANDIDATES:
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
