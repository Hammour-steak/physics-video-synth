from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_BLENDER = WORKSPACE_DIR / "tools" / "blender-3.6.23-linux-x64" / "blender"
RENDER_SCRIPT = WORKSPACE_DIR / "scripts" / "dining_chain" / "render_dining_chain.py"

# The scene is otherwise deterministic, so each sample is varied by jittering
# the two physics knobs within ranges that keep the full can->cup->milk chain
# (grippy finished-wood friction, firm push) but change the timing/spread.
LAUNCH_SPEED_RANGE = (3.0, 3.6)
TABLE_FRICTION_RANGE = (0.28, 0.32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch render randomized dining-table sliding-chain videos with Blender."
    )
    parser.add_argument("--out-root", type=Path, default=WORKSPACE_DIR / "renders" / "batch_dining_chain")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed-base", type=int, default=27000)
    parser.add_argument("--mode", choices=("preview", "animation"), default="preview")
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


def sample_output_name(mode: str) -> str:
    return "preview.png" if mode == "preview" else "dining_chain.mp4"


def sample_params(seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    return {
        "launch_speed": round(rng.uniform(*LAUNCH_SPEED_RANGE), 3),
        "table_friction": round(rng.uniform(*TABLE_FRICTION_RANGE), 3),
    }


def render_command(args: argparse.Namespace, out_dir: Path, params: dict[str, float]) -> list[str]:
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
        "--launch-speed", str(params["launch_speed"]),
        "--table-friction", str(params["table_friction"]),
    ]


def executable_path(path: Path) -> Path | None:
    expanded = path.expanduser()
    if expanded.exists():
        return expanded.resolve()
    resolved = shutil.which(str(path))
    return Path(resolved).resolve() if resolved else None


def resolve_blender_path(configured: Path | None) -> Path:
    candidates: list[Path] = []
    if configured is not None:
        candidates.append(configured)
    if os.environ.get("BLENDER_BIN"):
        candidates.append(Path(os.environ["BLENDER_BIN"]))
    candidates += [DEFAULT_BLENDER, Path("blender")]
    for c in candidates:
        p = executable_path(c)
        if p is not None:
            return p
    raise FileNotFoundError(
        f"Blender executable not found. Pass --blender, set BLENDER_BIN, or place it at {DEFAULT_BLENDER}."
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
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for offset in range(int(args.count)):
        sample_index = int(args.start_index) + offset
        seed = int(args.seed_base) + sample_index
        params = sample_params(seed)
        sample_dir = args.out_root / f"sample_{sample_index:04d}"
        expected = sample_dir / sample_output_name(args.mode)
        command = render_command(args, sample_dir, params)
        record = {
            "sample_index": sample_index,
            "seed": seed,
            "params": params,
            "out_dir": str(sample_dir.resolve()),
            "expected_output": str(expected.resolve()),
            "command": command,
            "status": "pending",
        }
        manifest["samples"].append(record)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        if args.skip_existing and expected.exists():
            record["status"] = "skipped_existing"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"[batch] skip sample_{sample_index:04d}: {expected}")
            continue

        print(f"[batch] render sample_{sample_index:04d} seed={seed} params={params}")
        if args.dry_run:
            record["status"] = "dry_run"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(" ".join(command))
            continue

        sample_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, check=True)
        record["status"] = "completed"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[batch] manifest={manifest_path.resolve()}")


if __name__ == "__main__":
    main()
