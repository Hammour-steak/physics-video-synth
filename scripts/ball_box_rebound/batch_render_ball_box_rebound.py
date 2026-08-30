from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


WORKSPACE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_BLENDER = WORKSPACE_DIR / "tools" / "blender-3.6.23-linux-x64" / "blender"
RENDER_SCRIPT = WORKSPACE_DIR / "scripts" / "ball_box_rebound" / "render_ball_box_rebound.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch render ball-off-toy-chest videos with Blender.",
    )
    parser.add_argument("--out-root", type=Path,
                        default=WORKSPACE_DIR / "renders" / "batch_ball_box_rebound")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed-base", type=int, default=11000)
    parser.add_argument("--mode", choices=("preview", "animation", "frames"), default="preview")
    parser.add_argument("--resolution", nargs=2, type=int, default=(960, 540))
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-sec", type=float, default=2.6)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--preview-frame", type=int, default=23)
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--blender", type=Path, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
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
        raise RuntimeError(
            "Cannot find Blender. Pass --blender, set BLENDER_BIN, or install "
            f"it at {DEFAULT_BLENDER}."
        )
    return Path(found)


def sample_output_name(mode: str) -> str:
    if mode == "preview":
        return "preview.png"
    if mode == "frames":
        return "frame_0001.png"
    return "ball_box_rebound.mp4"


def render_command(args: argparse.Namespace, *, out_dir: Path, seed: int) -> list[str]:
    return [
        str(args.blender), "-b", "--python", str(RENDER_SCRIPT), "--",
        "--mode", str(args.mode),
        "--out-dir", str(out_dir),
        "--resolution", str(int(args.resolution[0])), str(int(args.resolution[1])),
        "--fps", str(int(args.fps)),
        "--duration-sec", str(float(args.duration_sec)),
        "--samples", str(int(args.samples)),
        "--preview-frame", str(int(args.preview_frame)),
        "--device", str(args.device),
        "--seed", str(int(seed)),
    ]


def main() -> None:
    args = parse_args()
    args.blender = resolve_blender(args.blender)
    args.out_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "schema_version": 1,
        "scene": "ball_box_rebound",
        "blender": str(args.blender),
        "mode": args.mode,
        "count": int(args.count),
        "samples": [],
    }

    for index in range(int(args.start_index), int(args.start_index) + int(args.count)):
        seed = int(args.seed_base) + index
        out_dir = args.out_root / f"sample_{index:04d}"
        expected = out_dir / sample_output_name(args.mode)
        command = render_command(args, out_dir=out_dir, seed=seed)

        entry = {
            "index": index,
            "seed": seed,
            "out_dir": str(out_dir),
            "expected_output": str(expected),
            "command": command,
        }
        manifest["samples"].append(entry)

        if args.dry_run:
            print("[DRY-RUN]", " ".join(command))
            continue
        if args.skip_existing and expected.exists():
            print(f"[SKIP] {expected} already exists.")
            entry["status"] = "skipped"
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[RENDER] sample {index} (seed {seed}) -> {out_dir}")
        result = subprocess.run(command)
        entry["status"] = "ok" if result.returncode == 0 else f"failed({result.returncode})"
        if result.returncode != 0:
            print(f"[ERROR] sample {index} failed with exit code {result.returncode}.")

    (args.out_root / "batch_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    print(f"[INFO] Batch manifest written to {args.out_root / 'batch_manifest.json'}")


if __name__ == "__main__":
    main()
