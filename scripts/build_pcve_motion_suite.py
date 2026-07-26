from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


WORKSPACE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BLENDER = WORKSPACE_DIR / "tools" / "blender-3.6.23-linux-x64" / "blender"
RENDER_SCRIPT = WORKSPACE_DIR / "scripts" / "render_ball_block_impact.py"
CANONICAL_BLOCK_TEXTURE = "wood_table"


@dataclass(frozen=True)
class RenderCase:
    case_id: str
    description: str
    seed: int
    motion: str
    block_texture_asset: str
    overrides: dict[str, Any]


NEW_CASES = (
    RenderCase(
        case_id="existing_side_impact_wood_table",
        description="Ground rolling ball, moderate side impact, static camera.",
        seed=1000,
        motion="side_impact",
        block_texture_asset=CANONICAL_BLOCK_TEXTURE,
        overrides={
            "motion": "side_impact",
            "physics": {
                "motion": "side_impact",
                "ball_initial_location": [-3.2121548713, -0.1494058978, 0.341],
                "ball_initial_velocity": [4.20, 0.01162, 0.0],
                "block_location": [0.2899241957, 0.0104738339, 0.35],
                "block_yaw_deg": 1.0638405158,
                "ball_mass": 0.5374993180,
                "block_mass": 0.84,
                "floor_friction": 0.8431475864,
                "ball_friction": 1.10,
                "ball_restitution": 0.7570391780,
                "block_friction": 0.23,
                "block_restitution": 0.72,
            },
        },
    ),
    RenderCase(
        case_id="side_moderate_head_on",
        description="Ground rolling ball, moderate head-on contact, fully visible.",
        seed=3101,
        motion="side_impact",
        block_texture_asset=CANONICAL_BLOCK_TEXTURE,
        overrides={
            "motion": "side_impact",
            "physics": {
                "motion": "side_impact",
                "ball_initial_location": [-2.45, -0.10, 0.341],
                "ball_initial_velocity": [4.45, 0.02, 0.0],
                "block_location": [0.23, -0.02, 0.35],
                "block_yaw_deg": 0.0,
                "ball_mass": 0.58,
                "block_mass": 0.65,
                "floor_friction": 0.82,
                "ball_friction": 0.38,
                "ball_restitution": 0.76,
                "block_friction": 0.32,
                "block_restitution": 0.52,
            },
        },
    ),
    RenderCase(
        case_id="side_oblique_moderate",
        description="Ground rolling ball with mild oblique velocity and block yaw.",
        seed=3102,
        motion="side_impact",
        block_texture_asset=CANONICAL_BLOCK_TEXTURE,
        overrides={
            "motion": "side_impact",
            "physics": {
                "motion": "side_impact",
                "ball_initial_location": [-2.60, -0.22, 0.341],
                "ball_initial_velocity": [4.80, 0.34, 0.0],
                "block_location": [0.20, 0.02, 0.35],
                "block_yaw_deg": 4.0,
                "ball_mass": 0.58,
                "block_mass": 0.68,
                "floor_friction": 0.78,
                "ball_friction": 0.36,
                "ball_restitution": 0.74,
                "block_friction": 0.34,
                "block_restitution": 0.50,
            },
        },
    ),
    RenderCase(
        case_id="side_slow_graze",
        description="Lower-energy ground contact with a shallow lateral graze.",
        seed=3103,
        motion="side_impact",
        block_texture_asset=CANONICAL_BLOCK_TEXTURE,
        overrides={
            "motion": "side_impact",
            "physics": {
                "motion": "side_impact",
                "ball_initial_location": [-2.30, -0.34, 0.341],
                "ball_initial_velocity": [3.95, 0.28, 0.0],
                "block_location": [0.24, -0.04, 0.35],
                "block_yaw_deg": -3.0,
                "ball_mass": 0.60,
                "block_mass": 0.66,
                "floor_friction": 0.86,
                "ball_friction": 0.42,
                "ball_restitution": 0.70,
                "block_friction": 0.36,
                "block_restitution": 0.48,
            },
        },
    ),
    RenderCase(
        case_id="drop_centered_soft",
        description="Aerial free fall nearly centered above the wood block.",
        seed=3201,
        motion="drop_onto_block",
        block_texture_asset=CANONICAL_BLOCK_TEXTURE,
        overrides={
            "motion": "drop_onto_block",
            "physics": {
                "motion": "drop_onto_block",
                "ball_initial_location": [0.07, -0.02, 2.12],
                "ball_initial_velocity": [0.08, 0.00, -0.16],
                "block_location": [0.23, -0.02, 0.35],
                "block_yaw_deg": 0.0,
                "ball_mass": 0.58,
                "block_mass": 0.68,
                "floor_friction": 0.82,
                "ball_friction": 0.38,
                "ball_restitution": 0.76,
                "block_friction": 0.34,
                "block_restitution": 0.50,
            },
        },
    ),
    RenderCase(
        case_id="drop_lateral_mild",
        description="Aerial drop with mild lateral drift before contact.",
        seed=3202,
        motion="drop_onto_block",
        block_texture_asset=CANONICAL_BLOCK_TEXTURE,
        overrides={
            "motion": "drop_onto_block",
            "physics": {
                "motion": "drop_onto_block",
                "ball_initial_location": [0.02, -0.11, 2.20],
                "ball_initial_velocity": [0.28, 0.09, -0.18],
                "block_location": [0.25, -0.01, 0.35],
                "block_yaw_deg": 3.0,
                "ball_mass": 0.56,
                "block_mass": 0.70,
                "floor_friction": 0.80,
                "ball_friction": 0.36,
                "ball_restitution": 0.78,
                "block_friction": 0.35,
                "block_restitution": 0.52,
            },
        },
    ),
    RenderCase(
        case_id="incline_slide_falloff",
        description="Ball slides down a finite incline, leaves the edge, and lands on the floor.",
        seed=3301,
        motion="incline_slide_falloff",
        block_texture_asset=CANONICAL_BLOCK_TEXTURE,
        overrides={
            "motion": "incline_slide_falloff",
            "physics": {
                "motion": "incline_slide_falloff",
                "ball_initial_location": [-1.1059, -0.01, 1.4081],
                "ball_initial_velocity": [0.46, 0.0, -0.02],
                "block_location": [0.20, 1.05, 0.35],
                "block_yaw_deg": 0.0,
                "ball_mass": 0.58,
                "block_mass": 0.68,
                "floor_friction": 0.82,
                "ball_friction": 0.36,
                "ball_restitution": 0.48,
                "block_friction": 0.35,
                "block_restitution": 0.52,
                "ramp_enabled": True,
                "ramp_location": [-0.35, 0.0, 0.77],
                "ramp_dimensions": [2.5, 1.1, 0.08],
                "ramp_pitch_deg": 18.0,
                "ramp_friction": 0.58,
                "ramp_restitution": 0.05,
            },
        },
    ),
    RenderCase(
        case_id="wood_incline_slide_falloff",
        description="Wood block slides down a finite incline, leaves the edge, and lands on the floor.",
        seed=3302,
        motion="wood_incline_slide_falloff",
        block_texture_asset=CANONICAL_BLOCK_TEXTURE,
        overrides={
            "motion": "wood_incline_slide_falloff",
            "physics": {
                "motion": "wood_incline_slide_falloff",
                "ball_enabled": False,
                "block_enabled": True,
                "ball_initial_location": [0.0, -10.0, 0.341],
                "ball_initial_velocity": [0.0, 0.0, 0.0],
                "block_location": [-1.5132, 0.0, 1.5579],
                "block_yaw_deg": 0.0,
                "block_pitch_deg": 18.0,
                "block_initial_velocity": [0.1427, 0.0, -0.0464],
                "ball_mass": 0.58,
                "block_mass": 0.68,
                "floor_friction": 0.82,
                "block_friction": 0.35,
                "block_restitution": 0.18,
                "ramp_enabled": True,
                "ramp_location": [-0.35, 0.0, 0.77],
                "ramp_dimensions": [2.5, 1.1, 0.08],
                "ramp_pitch_deg": 18.0,
                "ramp_friction": 0.58,
                "ramp_restitution": 0.05,
            },
        },
    ),
    RenderCase(
        case_id="wood_incline_grounded_slide_falloff_v3",
        description=(
            "Wood block slides down a floor-connected solid incline, leaves the "
            "lower edge, and lands on the floor."
        ),
        seed=3302,
        motion="wood_incline_slide_falloff",
        block_texture_asset=CANONICAL_BLOCK_TEXTURE,
        overrides={
            "motion": "wood_incline_slide_falloff",
            "physics": {
                "motion": "wood_incline_slide_falloff",
                "ball_enabled": False,
                "block_enabled": True,
                "ball_initial_location": [0.0, -10.0, 0.341],
                "ball_initial_velocity": [0.0, 0.0, 0.0],
                "block_location": [-1.037881411, 0.0, 1.403576487],
                "block_yaw_deg": 0.0,
                "block_pitch_deg": 18.0,
                "block_initial_velocity": [0.1427, 0.0, -0.0464],
                "ball_mass": 0.58,
                "block_mass": 0.68,
                "floor_friction": 0.82,
                "block_friction": 0.35,
                "block_restitution": 0.18,
                "ramp_enabled": True,
                "ramp_location": [-0.35, 0.0, 0.77],
                "ramp_dimensions": [2.5, 1.1, 0.08],
                "ramp_pitch_deg": 18.0,
                "ramp_friction": 0.58,
                "ramp_restitution": 0.05,
                "ramp_profile": "grounded_wedge",
            },
            "camera": {
                "target": [0.12, -0.03248376797838282, 0.76],
                "lens_mm": 36.0,
            },
        },
    ),
    RenderCase(
        case_id="wall_bounce",
        description="Ground rolling ball collides with the back wall and rebounds.",
        seed=3401,
        motion="wall_bounce",
        block_texture_asset=CANONICAL_BLOCK_TEXTURE,
        overrides={
            "motion": "wall_bounce",
            "physics": {
                "motion": "wall_bounce",
                "ball_enabled": True,
                "block_enabled": False,
                "ball_initial_location": [-0.85, -1.32, 0.341],
                "ball_initial_velocity": [0.27070, 1.85, 0.0],
                "block_location": [0.20, 1.05, 0.35],
                "block_yaw_deg": 0.0,
                "ball_mass": 0.58,
                "floor_friction": 0.82,
                "ball_friction": 0.38,
                "ball_restitution": 0.82,
                "wall_enabled": True,
                "wall_location": [0.0, 3.05, 1.45],
                "wall_dimensions": [8.6, 0.08, 2.90],
                "wall_friction": 0.34,
                "wall_restitution": 0.82,
            },
        },
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a named PCVE synthetic motion benchmark suite."
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=WORKSPACE_DIR / "renders" / "pcve_general_motion_suite",
    )
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--resolution", nargs=2, type=int, default=(1280, 720))
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-sec", type=float, default=8.0)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--surface-marks", choices=("none", "subtle", "full"), default="none")
    parser.add_argument("--physics-jitter", type=float, default=0.0)
    parser.add_argument("--camera-jitter", type=float, default=0.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--case-id",
        action="append",
        choices=tuple(case.case_id for case in NEW_CASES),
        help="Render only the selected case; repeat to select multiple cases.",
    )
    parser.add_argument(
        "--keep-stale-cases",
        action="store_true",
        help="Keep old case directories that are no longer part of this suite.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--verbose-render",
        action="store_true",
        help="Stream the full Blender/ffmpeg render log instead of showing only suite progress.",
    )
    return parser.parse_args()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def block_texture_asset(metadata_path: Path) -> str | None:
    metadata = read_json(metadata_path)
    materials = metadata.get("materials")
    if not isinstance(materials, dict):
        return None
    value = materials.get("block_texture_asset")
    return str(value) if value is not None else None


def validate_block_texture(
    metadata_path: Path,
    *,
    expected: str = CANONICAL_BLOCK_TEXTURE,
) -> None:
    actual = block_texture_asset(metadata_path)
    if actual != expected:
        raise ValueError(
            f"Expected block_texture_asset={expected!r} in {metadata_path}, got {actual!r}."
        )


def clean_stale_case_dirs(out_root: Path, *, keep_case_ids: set[str]) -> None:
    cases_dir = out_root / "cases"
    if not cases_dir.exists():
        return
    for case_dir in sorted(cases_dir.iterdir()):
        if not case_dir.is_dir() or case_dir.name in keep_case_ids:
            continue
        print(f"[suite] remove stale case directory {case_dir}")
        shutil.rmtree(case_dir)


def case_outputs_match_texture(
    case_dir: Path,
    *,
    expected: str = CANONICAL_BLOCK_TEXTURE,
) -> bool:
    metadata_path = case_dir / "scenario_metadata.json"
    if not metadata_path.exists():
        return False
    try:
        validate_block_texture(metadata_path, expected=expected)
    except (json.JSONDecodeError, ValueError):
        return False
    return True


def preferred_video_path(case_dir: Path) -> Path:
    preferred = (
        case_dir / "ball_block_impact.mp4",
        case_dir / "ball_block_impact_cycles.mp4",
        case_dir / "ball_block_impact_cycles222111.mp4",
    )
    for path in preferred:
        if path.exists():
            return path
    matches = sorted(case_dir.glob("*.mp4"))
    if not matches:
        raise FileNotFoundError(f"No mp4 found in {case_dir}")
    return matches[0]


def require_file(path: Path, *, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def render_command(
    args: argparse.Namespace,
    case: RenderCase,
    *,
    case_dir: Path,
    overrides_path: Path,
) -> list[str]:
    return [
        str(args.blender.expanduser().resolve()),
        "-b",
        "--python",
        str(RENDER_SCRIPT.resolve()),
        "--",
        "--mode",
        "animation",
        "--out-dir",
        str(case_dir.resolve()),
        "--resolution",
        str(int(args.resolution[0])),
        str(int(args.resolution[1])),
        "--fps",
        str(int(args.fps)),
        "--duration-sec",
        str(float(args.duration_sec)),
        "--samples",
        str(int(args.samples)),
        "--device",
        str(args.device),
        "--seed",
        str(int(case.seed)),
        "--motion",
        str(case.motion),
        "--block-texture-asset",
        str(case.block_texture_asset),
        "--physics-jitter",
        str(float(args.physics_jitter)),
        "--camera-jitter",
        str(float(args.camera_jitter)),
        "--surface-marks",
        str(args.surface_marks),
        "--scenario-overrides-json",
        str(overrides_path.resolve()),
    ]


def standardize_render_outputs(case_dir: Path) -> dict[str, str]:
    video_source = preferred_video_path(case_dir)
    video_target = case_dir / "video.mp4"
    if video_source.resolve() != video_target.resolve():
        shutil.copy2(video_source, video_target)
    outputs = {
        "video": video_target,
        "ground_truth": case_dir / "ground_truth_transforms.json",
        "scenario_metadata": case_dir / "scenario_metadata.json",
        "scenario_overrides": case_dir / "scenario_overrides.json",
    }
    for key, path in outputs.items():
        require_file(path, description=f"rendered {key}")
    validate_block_texture(outputs["scenario_metadata"])
    return {key: str(path.resolve()) for key, path in outputs.items()}


def tail(text: str | None, *, max_lines: int = 80) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


def run_render(command: list[str], *, verbose: bool) -> None:
    if verbose:
        subprocess.run(command, check=True)
        return

    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode == 0:
        return

    print("[suite] render command failed; stdout tail:")
    print(tail(result.stdout))
    print("[suite] render command failed; stderr tail:")
    print(tail(result.stderr))
    raise subprocess.CalledProcessError(
        result.returncode,
        command,
        output=result.stdout,
        stderr=result.stderr,
    )


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    selected_case_ids = set(args.case_id or ())
    new_cases = tuple(
        case
        for case in NEW_CASES
        if not selected_case_ids or case.case_id in selected_case_ids
    )
    keep_case_ids = {
        *(case.case_id for case in NEW_CASES),
    }
    if not selected_case_ids and not args.keep_stale_cases and not args.dry_run:
        clean_stale_case_dirs(args.out_root, keep_case_ids=keep_case_ids)
    manifest_path = args.out_root / "suite_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "suite_name": "pcve_general_motion_suite",
        "description": (
            "Moderate ball/wood motion cases for checking whether PCVE free-motion "
            "and 4D fitting generalize beyond the two original examples."
        ),
        "resolution": [int(args.resolution[0]), int(args.resolution[1])],
        "fps": int(args.fps),
        "duration_sec": float(args.duration_sec),
        "samples": int(args.samples),
        "cases": [],
    }
    write_json(manifest_path, manifest)

    for case in new_cases:
        case_dir = args.out_root / "cases" / case.case_id
        overrides_path = case_dir / "scenario_overrides.json"
        write_json(overrides_path, case.overrides)
        command = render_command(
            args,
            case,
            case_dir=case_dir,
            overrides_path=overrides_path,
        )
        record = {
            "case_id": case.case_id,
            "kind": "rendered",
            "description": case.description,
            "seed": int(case.seed),
            "motion": case.motion,
            "case_dir": str(case_dir.resolve()),
            "scenario_overrides_json": str(overrides_path.resolve()),
            "command": command,
            "status": "pending",
        }
        manifest["cases"].append(record)
        write_json(manifest_path, manifest)

        expected_video = case_dir / "video.mp4"
        if (
            args.skip_existing
            and expected_video.exists()
            and case_outputs_match_texture(case_dir)
        ):
            record["status"] = "skipped_existing"
            record["outputs"] = standardize_render_outputs(case_dir)
            write_json(manifest_path, manifest)
            continue
        if args.skip_existing and expected_video.exists():
            print(
                f"[suite] existing {case.case_id} is stale or missing "
                f"{CANONICAL_BLOCK_TEXTURE}; rerender"
            )
        if args.dry_run:
            record["status"] = "dry_run"
            write_json(manifest_path, manifest)
            print(" ".join(command))
            continue

        case_dir.mkdir(parents=True, exist_ok=True)
        start_time = time.perf_counter()
        print(f"[suite] render {case.case_id}")
        try:
            run_render(command, verbose=bool(args.verbose_render))
        except subprocess.CalledProcessError:
            record["status"] = "failed"
            record["elapsed_sec"] = round(time.perf_counter() - start_time, 3)
            write_json(manifest_path, manifest)
            raise
        record["outputs"] = standardize_render_outputs(case_dir)
        record["elapsed_sec"] = round(time.perf_counter() - start_time, 3)
        record["status"] = "completed"
        write_json(manifest_path, manifest)
        print(f"[suite] completed {case.case_id} in {record['elapsed_sec']:.1f}s")

    print(f"[suite] manifest={manifest_path.resolve()}")


if __name__ == "__main__":
    main()
