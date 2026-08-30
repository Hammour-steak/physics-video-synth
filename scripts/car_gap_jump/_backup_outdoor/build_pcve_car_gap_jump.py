from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

WORKSPACE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_BLENDER = WORKSPACE_DIR / "tools" / "blender-3.6.23-linux-x64" / "blender"
RENDER_SCRIPT = WORKSPACE_DIR / "scripts" / "car_gap_jump" / "render_car_gap_jump.py"


@dataclass(frozen=True)
class RenderCase:
    case_id: str
    description: str
    seed: int
    launch_speed: float
    gap_width: float


# The suite isolates whether the car clears the broken-bridge gap, gated by the
# two knobs that control it: the approach/launch speed and the gap width.
# Everything else -- the ramp, the car, the decks, the chasm -- is identical
# between cases. Every outcome below is verified against
# simulate_car_gap_jump.py (a projectile launched off the ramp lip either
# reaches the far deck and lands upright, or falls short into the chasm).
CASES = (
    RenderCase(
        case_id="car_gap_jump_clears",
        description="The full jump: at 12 m/s the car rides up the ramp, launches off the lip, "
        "arcs across the 8 m gap, and lands upright on the far deck (~19 m past the lip) before "
        "rolling to a stop.",
        seed=31,
        launch_speed=12.0,
        gap_width=8.0,
    ),
    RenderCase(
        case_id="car_gap_jump_barely_clears",
        description="A slower 11 m/s approach over the same 8 m gap: the car still just reaches "
        "the far deck and lands upright, but only ~15 m past the lip -- a narrow success.",
        seed=32,
        launch_speed=11.0,
        gap_width=8.0,
    ),
    RenderCase(
        case_id="car_gap_jump_too_slow",
        description="Too little speed (9 m/s) for the same 8 m gap: the car noses off the lip but "
        "its arc falls short of the far deck, and it plummets into the chasm below.",
        seed=33,
        launch_speed=9.0,
        gap_width=8.0,
    ),
    RenderCase(
        case_id="car_gap_jump_wide_gap",
        description="The same firm 12 m/s launch as the baseline, but a wider 12 m gap: the car "
        "launches cleanly yet can't reach the far side -- it clears the lip, arcs, and drops into "
        "the chasm short of the landing deck.",
        seed=34,
        launch_speed=12.0,
        gap_width=12.0,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the car gap-jump PCVE suite.")
    parser.add_argument(
        "--out-root", type=Path, default=WORKSPACE_DIR / "renders" / "pcve_car_gap_jump_suite",
    )
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--resolution", nargs=2, type=int, default=(1280, 720))
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-sec", type=float, default=4.5)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def render_command(args: argparse.Namespace, case: RenderCase, case_dir: Path) -> list[str]:
    return [
        str(args.blender.expanduser().resolve()),
        "-b", "--python", str(RENDER_SCRIPT.resolve()), "--",
        "--mode", "animation",
        "--out-dir", str(case_dir.resolve()),
        "--resolution", str(int(args.resolution[0])), str(int(args.resolution[1])),
        "--fps", str(int(args.fps)),
        "--duration-sec", str(float(args.duration_sec)),
        "--samples", str(int(args.samples)),
        "--device", str(args.device),
        "--seed", str(int(case.seed)),
        "--launch-speed", str(float(case.launch_speed)),
        "--gap-width", str(float(case.gap_width)),
    ]


def standardize_outputs(case_dir: Path) -> dict[str, str]:
    src = case_dir / "car_gap_jump.mp4"
    if not src.exists():
        matches = sorted(case_dir.glob("*.mp4"))
        if not matches:
            raise FileNotFoundError(f"No mp4 found in {case_dir}")
        src = matches[0]
    dst = case_dir / "video.mp4"
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    outputs = {
        "video": dst,
        "ground_truth": case_dir / "ground_truth_transforms.json",
        "scenario_metadata": case_dir / "scenario_metadata.json",
    }
    return {key: str(path.resolve()) for key, path in outputs.items() if path.exists()}


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_root / "pcve_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "suite": "car_gap_jump",
        "description": "A sports car jumping the gap of a broken bridge, varying the launch speed "
        "and the gap width so the outcome flips between clearing the gap (landing upright on the "
        "far deck) and falling short into the chasm.",
        "resolution": [int(args.resolution[0]), int(args.resolution[1])],
        "fps": int(args.fps),
        "duration_sec": float(args.duration_sec),
        "samples": int(args.samples),
        "cases": [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for case in CASES:
        case_dir = args.out_root / "cases" / case.case_id
        command = render_command(args, case, case_dir)
        record: dict[str, Any] = {
            "case_id": case.case_id,
            "description": case.description,
            "seed": case.seed,
            "launch_speed": case.launch_speed,
            "gap_width": case.gap_width,
            "case_dir": str(case_dir.resolve()),
            "command": command,
            "status": "pending",
        }
        manifest["cases"].append(record)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        expected = case_dir / "video.mp4"
        if args.skip_existing and expected.exists():
            record["status"] = "skipped_existing"
            record["outputs"] = standardize_outputs(case_dir)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"[suite] skip existing {case.case_id}")
            continue

        if args.dry_run:
            record["status"] = "dry_run"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(" ".join(command))
            continue

        case_dir.mkdir(parents=True, exist_ok=True)
        print(f"[suite] render {case.case_id} (speed={case.launch_speed}, gap={case.gap_width})")
        started = time.time()
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError:
            record["status"] = "failed"
            record["elapsed_sec"] = round(time.time() - started, 1)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            raise
        record["elapsed_sec"] = round(time.time() - started, 1)
        record["outputs"] = standardize_outputs(case_dir)
        record["status"] = "completed"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[suite] completed {case.case_id} in {record['elapsed_sec']:.1f}s")

    print(f"[suite] manifest={manifest_path.resolve()}")


if __name__ == "__main__":
    main()
