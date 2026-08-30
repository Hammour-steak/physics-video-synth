from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


WORKSPACE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_BLENDER = WORKSPACE_DIR / "tools" / "blender-3.6.23-linux-x64" / "blender"
RENDER_SCRIPT = WORKSPACE_DIR / "scripts" / "domino_chain" / "render_domino_chain.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch render randomized domino chain videos with Blender."
    )
    parser.add_argument("--out-root", type=Path, default=WORKSPACE_DIR / "renders" / "batch_ramp")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed-base", type=int, default=6000)
    parser.add_argument("--mode", choices=("preview", "animation", "frames"), default="preview")
    parser.add_argument("--resolution", nargs=2, type=int, default=(960, 540))
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-sec", type=float, default=3.0)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--preview-frame", type=int, default=30)
    parser.add_argument("--device", choices=("auto", "cpu"), default="cpu")
    parser.add_argument("--blender", type=Path, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sample_output_name(args: argparse.Namespace) -> str:
    if args.mode == "preview":
        return "preview.png"
    if args.mode == "frames":
        return "frame_0001.png"
    return "domino_chain.mp4"


def render_command(args: argparse.Namespace, *, out_dir: Path, seed: int) -> list[str]:
    cmd = [
        str(args.blender),
        "-b",
        "--python",
        str(RENDER_SCRIPT),
        "--",
        "--mode",
        str(args.mode),
        "--out-dir",
        str(out_dir),
        "--resolution",
        str(int(args.resolution[0])),
        str(int(args.resolution[1])),
        "--fps",
        str(int(args.fps)),
        "--duration-sec",
        str(float(args.duration_sec)),
        "--samples",
        str(int(args.samples)),
        "--preview-frame",
        str(int(args.preview_frame)),
        "--device",
        str(args.device),
        "--seed",
        str(int(seed)),
    ]
    return cmd


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def executable_path(path: Path) -> Path | None:
    expanded_path = path.expanduser()
    if expanded_path.exists():
        return expanded_path.resolve()
    resolved_path = shutil.which(str(path))
    if resolved_path is not None:
        return Path(resolved_path).resolve()
    return None


def resolve_blender_path(configured_path: Path | None) -> Path:
    candidates: list[Path] = []
    if configured_path is not None:
        candidates.append(configured_path)
    blender_env = os.environ.get("BLENDER_BIN")
    if blender_env:
        candidates.append(Path(blender_env))
    candidates.append(DEFAULT_BLENDER)
    candidates.append(Path("blender"))

    for candidate in candidates:
        blender_path = executable_path(candidate)
        if blender_path is not None:
            return blender_path

    raise FileNotFoundError(
        "Blender executable not found. Pass --blender, set BLENDER_BIN, "
        f"install Blender on PATH, or place it at {DEFAULT_BLENDER}."
    )


def main() -> None:
    args = parse_args()
    if int(args.count) <= 0:
        raise ValueError("--count must be positive.")
    args.blender = resolve_blender_path(args.blender)

    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_root / "batch_manifest.json"
    manifest: dict[str, object] = {
        "schema_version": 1,
        "mode": args.mode,
        "count": int(args.count),
        "start_index": int(args.start_index),
        "seed_base": int(args.seed_base),
        "samples": [],
    }
    write_manifest(manifest_path, manifest)

    for offset in range(int(args.count)):
        sample_index = int(args.start_index) + offset
        seed = int(args.seed_base) + sample_index
        sample_dir = args.out_root / f"sample_{sample_index:04d}"
        expected_output = sample_dir / sample_output_name(args)
        command = render_command(args, out_dir=sample_dir, seed=seed)
        sample_record = {
            "sample_index": sample_index,
            "seed": seed,
            "out_dir": str(sample_dir.resolve()),
            "expected_output": str(expected_output.resolve()),
            "command": command,
            "status": "pending",
        }
        manifest["samples"].append(sample_record)
        write_manifest(manifest_path, manifest)

        if args.skip_existing and expected_output.exists():
            sample_record["status"] = "skipped_existing"
            write_manifest(manifest_path, manifest)
            print(f"[batch] skip sample_{sample_index:04d}: {expected_output}")
            continue

        print(f"[batch] render sample_{sample_index:04d} seed={seed}")
        if args.dry_run:
            sample_record["status"] = "dry_run"
            write_manifest(manifest_path, manifest)
            print(" ".join(command))
            continue

        sample_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, check=True)
        sample_record["status"] = "completed"
        write_manifest(manifest_path, manifest)

    print(f"[batch] manifest={manifest_path.resolve()}")


if __name__ == "__main__":
    main()
