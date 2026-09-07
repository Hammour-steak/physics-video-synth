"""Build the table_drop_collision PCVE suite.

Each edit is declared as a single ``edit_dsl`` string. Prompts (precise +
vague, zh + en), scenario overrides, and the physics diff for the manifest
are all derived from that one string.

The scene: a "new" tennis ball rolls off a coffee table, arcs to the floor,
and strikes an "old" tennis ball waiting on the rug in line with the lane.
At baseline the rolling ball leaves the lip at 1.05 m/s, lands 0.67 m out,
and the head-on hit sends the target 0.30 m west while the rolling ball is
kicked back 0.74 m east. Every value below was picked by sweeping
simulate_table_drop_collision.py directly; the numbers in each edit_summary
are that sweep's output at the suite's defaults.

Notes on the edit surface:
- Only the target ball can be DELETEd. The rolling ball is the trigger
  that drives the whole simulation (the sim is not built to run without
  it), so a DELETE on it is not exposed.
- The rolling ball is the only one with a non-zero baseline velocity, so
  `initial_velocity` is bound only on it (via launch_speed).
- 3+ different push-speed cases were surveyed; the sim's high-substep
  contact starts to miss above 1.5 m/s (fast rolling ball passes over the
  target), so the "hard break" analogue is not in the suite -- only the
  soft-push (miss) direction is.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pcve_edit_dsl as dsl  # noqa: E402
import edit_vocab             # noqa: E402


WORKSPACE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_BLENDER = WORKSPACE_DIR / "tools" / "blender-3.6.23-linux-x64" / "blender"
RENDER_SCRIPT = WORKSPACE_DIR / "scripts" / "table_drop_collision" / "render_table_drop_collision.py"


VOCAB = edit_vocab.VOCAB
BASELINE_PHYSICS = edit_vocab.BASELINE_PHYSICS


# --------------------------------------------------------------- edit cases


@dataclass(frozen=True)
class EditCase:
    case_id: str
    source_case_id: str
    seed: int
    dsl: str
    edit_summary: str
    # Optional per-case override of the suite's default duration; unused now
    # (was 3.4s for a few edits where the ball keeps rolling past the shot).
    duration_sec: float | None = None


SOURCE_CASE_ID = "table_drop_collision_baseline"


EDIT_CASES: tuple[EditCase, ...] = (
    EditCase(
        case_id="edit_heavy_rolling_ball",
        source_case_id=SOURCE_CASE_ID,
        seed=11101,
        dsl="SET rolling_ball.mass FROM 0.057 TO 0.285",
        edit_summary=(
            "Rolling ball made 5x heavier. The impact now carries much more "
            "momentum into the target: it rockets 1.64 m across the floor "
            "(5x the baseline's 0.30 m), and the rolling ball keeps driving "
            "on behind it, ending 0.98 m past the lip instead of 0.74 m."
        ),
    ),
    EditCase(
        case_id="edit_light_rolling_ball",
        source_case_id=SOURCE_CASE_ID,
        seed=11102,
        dsl="SET rolling_ball.mass FROM 0.057 TO 0.014",
        edit_summary=(
            "Rolling ball made 4x lighter (57 g -> 14 g). It still lands on "
            "top of the target, but bounces mostly straight up off it: the "
            "target barely moves (0.03 m vs 0.30 m at baseline) and the "
            "rolling ball comes to rest 0.19 m past the lip instead of 0.74 m."
        ),
    ),
    EditCase(
        case_id="edit_heavy_target_ball",
        source_case_id=SOURCE_CASE_ID,
        seed=11103,
        dsl="SET target_ball.mass FROM 0.057 TO 0.285",
        edit_summary=(
            "Target ball made 5x heavier. The rolling ball hits a wall of "
            "mass and rebounds backward -- ending up EAST of the lip "
            "instead of west of it -- and the target only creeps 0.02 m "
            "instead of the baseline's 0.30 m."
        ),
    ),
    EditCase(
        case_id="edit_soft_push",
        source_case_id=SOURCE_CASE_ID,
        seed=11104,
        dsl="SET rolling_ball.initial_velocity FROM 1.28 TO 0.8",
        edit_summary=(
            "Rolling ball pushed more gently across the table (1.28 -> 0.8 "
            "m/s). It reaches the lip at lower speed, is thrown 65 mm less "
            "far off the edge, and lands short of the target -- no contact "
            "at all. The target stays untouched at its starting position; "
            "the rolling ball rolls out west across the floor alone."
        ),
    ),
    EditCase(
        case_id="edit_dead_rolling_ball",
        source_case_id=SOURCE_CASE_ID,
        seed=11105,
        dsl="SET rolling_ball.restitution FROM 0.86 TO 0.1",
        edit_summary=(
            "Rolling ball's restitution killed (0.86 -> 0.1). Its floor "
            "bounce and its impact against the target both go inelastic: "
            "instead of transferring cleanly, the pair sticks and the "
            "rolling ball drives on behind the target. Target moves only "
            "0.12 m (vs 0.30 m at baseline); the rolling ball ends 0.97 m "
            "past the lip instead of 0.74 m."
        ),
    ),
    EditCase(
        case_id="edit_remove_target_ball",
        source_case_id=SOURCE_CASE_ID,
        seed=11106,
        dsl="DELETE target_ball",
        edit_summary=(
            "Target ball removed. The rolling ball leaves the table and "
            "lands on the rug exactly where the baseline predicts -- "
            "(-0.67, -0.29) -- and, with nothing to strike, rolls on "
            "unobstructed for 1.18 m before friction stops it."
        ),
    ),
)


# --------------------------------------------------------------------- CLI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the table_drop_collision PCVE suite (1 source + N edits)."
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=WORKSPACE_DIR / "renders" / "pcve_table_drop_collision_suite",
    )
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--resolution", nargs=2, type=int, default=(1280, 720))
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-sec", type=float, default=4.0,
                        help="Default clip length; edits that leave the ball "
                             "still running override this per-case.")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose-render", action="store_true")
    parser.add_argument(
        "--clean-stale-cases",
        action="store_true",
        help="Delete case directories that are no longer part of this suite.",
    )
    return parser.parse_args()


# --------------------------------------------------------------- render glue


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def render_command(
    args: argparse.Namespace,
    seed: int,
    *,
    case_dir: Path,
    overrides_path: Path | None,
    duration_sec: float,
) -> list[str]:
    cmd = [
        str(args.blender.expanduser().resolve()),
        "-b",
        "--python",
        str(RENDER_SCRIPT.resolve()),
        "--",
        "--mode", "animation",
        "--out-dir", str(case_dir.resolve()),
        "--resolution", str(int(args.resolution[0])), str(int(args.resolution[1])),
        "--fps", str(int(args.fps)),
        "--duration-sec", str(float(duration_sec)),
        "--samples", str(int(args.samples)),
        "--device", str(args.device),
        "--seed", str(int(seed)),
    ]
    if overrides_path is not None:
        cmd += ["--scenario-overrides-json", str(overrides_path.resolve())]
    return cmd


def standardize_render_outputs(case_dir: Path, *, has_overrides: bool) -> dict[str, str]:
    video_source = case_dir / "table_drop_collision.mp4"
    if not video_source.exists():
        candidates = sorted(case_dir.glob("*.mp4"))
        if not candidates:
            raise FileNotFoundError(f"No mp4 found in {case_dir}")
        video_source = candidates[0]
    video_target = case_dir / "video.mp4"
    if video_source.resolve() != video_target.resolve():
        shutil.copy2(video_source, video_target)

    outputs: dict[str, Path] = {
        "video": video_target,
        "ground_truth": case_dir / "ground_truth_transforms.json",
        "scenario_metadata": case_dir / "scenario_metadata.json",
    }
    if has_overrides:
        outputs["scenario_overrides"] = case_dir / "scenario_overrides.json"
    for key, path in outputs.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing rendered {key}: {path}")
    return {key: str(path.resolve()) for key, path in outputs.items()}


def render_case(
    args: argparse.Namespace,
    *,
    case_dir: Path,
    seed: int,
    overrides_path: Path | None,
    duration_sec: float,
) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    cmd = render_command(args, seed, case_dir=case_dir, overrides_path=overrides_path, duration_sec=duration_sec)
    if args.dry_run:
        print(" ".join(cmd))
        return
    if args.verbose_render:
        subprocess.run(cmd, check=True)
        return
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        tail = "\n".join((result.stderr or "").splitlines()[-40:])
        print(f"[suite] render failed; stderr tail:\n{tail}")
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)


def build_edit_record(case: EditCase) -> dict[str, Any]:
    parsed = dsl.parse(case.dsl, VOCAB)
    physics = dsl.to_physics_override(parsed, VOCAB)
    if isinstance(parsed, dsl.SetEdit):
        diff = {f"{parsed.property_name} ({parsed.object_id})":
                {"from": dsl.baseline_value_for(parsed, VOCAB), "to": parsed.to_value}}
    else:
        diff = {parsed.object_id: {"from": "present", "to": "removed"}}
    return {
        "edit_dsl": case.dsl,
        "edit_summary": case.edit_summary,
        "prompts": dsl.make_prompts(parsed, VOCAB),
        "physics_diff": diff,
        "physics_override": physics,
    }


def write_prompt_file(case_dir: Path, case: EditCase, edit_info: dict[str, Any]) -> Path:
    path = case_dir / "prompts.json"
    write_json(path, {
        "schema_version": 2,
        "case_id": case.case_id,
        "source_case_id": case.source_case_id,
        "edit_dsl": edit_info["edit_dsl"],
        "edit_summary": edit_info["edit_summary"],
        "physics_diff": edit_info["physics_diff"],
        "prompts": edit_info["prompts"],
    })
    return path


def clean_stale(out_root: Path, keep_ids: set[str]) -> None:
    cases_dir = out_root / "cases"
    if not cases_dir.exists():
        return
    for path in sorted(cases_dir.iterdir()):
        if not path.is_dir() or path.name in keep_ids:
            continue
        print(f"[suite] remove stale case directory {path}")
        shutil.rmtree(path)


# -------------------------------------------------------------------- main


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    keep_ids = {SOURCE_CASE_ID, *(c.case_id for c in EDIT_CASES)}
    if args.clean_stale_cases and not args.dry_run:
        clean_stale(args.out_root, keep_ids)

    manifest_path = args.out_root / "suite_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 3,
        "suite_name": "pcve_table_drop_collision_suite",
        "description": (
            "One source table-drop head-on-collision video plus N edited "
            "variants. A tennis ball rolls off a coffee table onto a second "
            "tennis ball on the floor. Each edit is declared with a single "
            "edit-DSL string; prompts (precise + vague, zh + en), scenario "
            "overrides, and the physics diff are all derived from that one "
            "string."
        ),
        "baseline_physics": BASELINE_PHYSICS,
        "resolution": [int(args.resolution[0]), int(args.resolution[1])],
        "fps": int(args.fps),
        "duration_sec": float(args.duration_sec),
        "samples": int(args.samples),
        "source": None,
        "edits": [],
    }
    write_json(manifest_path, manifest)

    # ------------------------------------------------------------- source
    source_dir = args.out_root / "cases" / SOURCE_CASE_ID
    source_record: dict[str, Any] = {
        "case_id": SOURCE_CASE_ID,
        "kind": "source",
        "description": (
            "Source video: default parameters. A new (bright yellow) tennis "
            "ball is pushed at 1.28 m/s along the table, leaves the west lip "
            "at 1.05 m/s, lands at (-0.67, -0.29) on the rug, and hits an "
            "old (dulled) tennis ball waiting in line. The old ball is "
            "driven 0.30 m west; the new ball is kicked 0.74 m east."
        ),
        "case_dir": str(source_dir.resolve()),
        "status": "pending",
    }
    manifest["source"] = source_record
    write_json(manifest_path, manifest)

    expected_source_video = source_dir / "video.mp4"
    if args.skip_existing and expected_source_video.exists():
        source_record["status"] = "skipped_existing"
        source_record["outputs"] = standardize_render_outputs(source_dir, has_overrides=False)
    elif args.dry_run:
        render_case(args, case_dir=source_dir, seed=11001, overrides_path=None,
                    duration_sec=float(args.duration_sec))
        source_record["status"] = "dry_run"
    else:
        t0 = time.perf_counter()
        print(f"[suite] render source {SOURCE_CASE_ID}")
        render_case(args, case_dir=source_dir, seed=11001, overrides_path=None,
                    duration_sec=float(args.duration_sec))
        source_record["outputs"] = standardize_render_outputs(source_dir, has_overrides=False)
        source_record["elapsed_sec"] = round(time.perf_counter() - t0, 3)
        source_record["status"] = "completed"
    write_json(manifest_path, manifest)

    # -------------------------------------------------------------- edits
    for case in EDIT_CASES:
        case_dir = args.out_root / "cases" / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        edit_info = build_edit_record(case)

        overrides_payload = {"physics": edit_info["physics_override"]}
        overrides_path = case_dir / "scenario_overrides.json"
        write_json(overrides_path, overrides_payload)
        prompts_path = write_prompt_file(case_dir, case, edit_info)

        duration_sec = case.duration_sec if case.duration_sec is not None else float(args.duration_sec)

        record: dict[str, Any] = {
            "case_id": case.case_id,
            "kind": "edit",
            "source_case_id": case.source_case_id,
            "seed": case.seed,
            "duration_sec": duration_sec,
            "case_dir": str(case_dir.resolve()),
            "scenario_overrides_json": str(overrides_path.resolve()),
            "prompts_json": str(prompts_path.resolve()),
            "edit_dsl": edit_info["edit_dsl"],
            "edit_summary": edit_info["edit_summary"],
            "physics_diff": edit_info["physics_diff"],
            "prompts": edit_info["prompts"],
            "status": "pending",
        }
        manifest["edits"].append(record)
        write_json(manifest_path, manifest)

        expected_video = case_dir / "video.mp4"
        if args.skip_existing and expected_video.exists():
            record["status"] = "skipped_existing"
            record["outputs"] = standardize_render_outputs(case_dir, has_overrides=True)
            write_json(manifest_path, manifest)
            print(f"[suite] skip existing {case.case_id}")
            continue

        if args.dry_run:
            render_case(args, case_dir=case_dir, seed=case.seed, overrides_path=overrides_path,
                        duration_sec=duration_sec)
            record["status"] = "dry_run"
            write_json(manifest_path, manifest)
            continue

        t0 = time.perf_counter()
        print(f"[suite] render edit {case.case_id}")
        try:
            render_case(args, case_dir=case_dir, seed=case.seed, overrides_path=overrides_path,
                        duration_sec=duration_sec)
        except subprocess.CalledProcessError:
            record["status"] = "failed"
            record["elapsed_sec"] = round(time.perf_counter() - t0, 3)
            write_json(manifest_path, manifest)
            raise
        record["outputs"] = standardize_render_outputs(case_dir, has_overrides=True)
        record["elapsed_sec"] = round(time.perf_counter() - t0, 3)
        record["status"] = "completed"
        write_json(manifest_path, manifest)
        print(f"[suite] completed {case.case_id} in {record['elapsed_sec']:.1f}s")

    print(f"[suite] manifest={manifest_path.resolve()}")


if __name__ == "__main__":
    main()
