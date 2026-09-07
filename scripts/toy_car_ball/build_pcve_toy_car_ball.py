"""Build the toy_car_ball PCVE suite.

Each edit is declared as a single ``edit_dsl`` string. Prompts (precise +
vague, zh + en), scenario overrides, and the physics diff for the manifest
are all derived from that one string.

The scene: a small toy car is pushed along a wooden shelf at 0.6 m/s and
rear-ends a soft toy ball sitting near the far edge. At baseline the ball
is knocked off the shelf, falls to the floor, and comes to rest at
x=-1.08 m. The car keeps going and also topples off the edge behind it.

Every value below was picked by sweeping simulate_toy_car_ball.py directly;
the numbers in each edit_summary are that sweep's output at the suite's
defaults.

Notes on the edit surface:
- Only the ball can be DELETEd. The car drives the whole simulation, so
  a DELETE on it is not exposed.
- The car is the only object with a non-zero baseline velocity, so
  `initial_velocity` is bound only on it (via launch_speed).
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
RENDER_SCRIPT = WORKSPACE_DIR / "scripts" / "toy_car_ball" / "render_toy_car_ball.py"


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


SOURCE_CASE_ID = "toy_car_ball_baseline"


EDIT_CASES: tuple[EditCase, ...] = (
    EditCase(
        case_id="edit_soft_push",
        source_case_id=SOURCE_CASE_ID,
        seed=15101,
        dsl="SET toy_car.initial_velocity TIMES 0.6",
        edit_summary=(
            "Car pushed more gently. Table friction bleeds off the push "
            "before the car ever reaches the ball: it coasts 0.15 m and "
            "stops, still on the table, and the ball sits untouched near "
            "the edge for the whole shot -- no collision and no fall, "
            "where the baseline knocks it 0.89 m off the shelf."
        ),
    ),
    EditCase(
        case_id="edit_hard_push",
        source_case_id=SOURCE_CASE_ID,
        seed=15102,
        dsl="SET toy_car.initial_velocity TIMES 1.5",
        edit_summary=(
            "Car pushed 50% harder (0.6 -> 0.9 m/s). The heavier hit sends "
            "the ball flying off the edge much further: it lands and rolls "
            "out to x=-2.42 m, more than twice the baseline's 1.08 m. The "
            "car itself also flies off behind it."
        ),
    ),
    EditCase(
        case_id="edit_heavy_ball",
        source_case_id=SOURCE_CASE_ID,
        seed=15103,
        dsl="SET toy_ball.mass TIMES 20",
        edit_summary=(
            "Ball made twenty times heavier, 0.05 kg to 1.0 kg, which puts it "
            "at three times the car's own mass. The car still crosses the "
            "shelf and reaches it -- 0.36 m of travel against the baseline's "
            "0.40 m, stopping against the ball rather than driving through "
            "it -- but it no longer has the momentum to move it: the ball "
            "shifts 24 mm and stays on the shelf, where the baseline sends it "
            "over the edge and 0.89 m down to the floor. What separates this "
            "from edit_soft_push is that the collision does happen; the car "
            "arrives and is stopped by the ball instead of running out of "
            "push half-way there."
        ),
    ),
    EditCase(
        case_id="edit_bouncy_ball",
        source_case_id=SOURCE_CASE_ID,
        seed=15104,
        dsl="SET toy_ball.restitution TIMES 1.5",
        edit_summary=(
            "Ball's restitution raised by half, to a near-elastic 0.90. "
            "It is still knocked off the shelf, but instead of settling "
            "on the floor it keeps bouncing through the rest of the shot "
            "-- five hops against the baseline's two -- and is still 0.17 "
            "m up and moving at 0.23 m/s when the clip ends."
        ),
    ),
    EditCase(
        case_id="edit_remove_toy_ball",
        source_case_id=SOURCE_CASE_ID,
        seed=15105,
        dsl="DELETE toy_ball",
        edit_summary=(
            "Ball removed. The car drives across an empty shelf and reaches "
            "the far edge without hitting anything; no fall, no collision. "
            "The car itself comes to rest near the edge instead of being "
            "knocked off behind the ball as in the baseline."
        ),
    ),
    # The one edit in this suite that does not hold for the whole clip, and
    # deliberately the same DELETE as the case above: they differ by the AT
    # FRAME clause alone. Taking the ball away after the car has already
    # pushed it is what makes the timing readable -- removing it beforehand
    # would leave nothing for the car to push and a different run entirely.
    EditCase(
        case_id="edit_remove_toy_ball_after_push",
        source_case_id=SOURCE_CASE_ID,
        seed=15106,
        dsl="DELETE toy_ball AT FRAME 18",
        edit_summary=(
            "Ball removed at frame 18, six frames after the car reaches it at "
            "frame 12. Frames 1-17 are the source video frame for frame: the "
            "car noses into the ball and starts it rolling, and the ball is "
            "0.07 m along when it disappears -- well short of the table edge "
            "it rolls off in the source. The car carries on exactly as in the "
            "source, stopping at x=-0.27. The whole-clip version of the same "
            "delete has nothing to push against at all: the car runs 0.05 m "
            "further, to x=-0.32."
        ),
    ),
)


# --------------------------------------------------------------------- CLI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the toy_car_ball PCVE suite (1 source + N edits)."
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=WORKSPACE_DIR / "renders" / "pcve_toy_car_ball_suite",
    )
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--resolution", nargs=2, type=int, default=(1280, 720))
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-sec", type=float, default=4.0)
    parser.add_argument("--samples", type=int, default=32)
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
        "--duration-sec", str(float(args.duration_sec)),
        "--samples", str(int(args.samples)),
        "--device", str(args.device),
        "--seed", str(int(seed)),
    ]
    if overrides_path is not None:
        cmd += ["--scenario-overrides-json", str(overrides_path.resolve())]
    return cmd


def standardize_render_outputs(case_dir: Path, *, has_overrides: bool) -> dict[str, str]:
    video_source = case_dir / "toy_car_ball.mp4"
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
) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    cmd = render_command(args, seed, case_dir=case_dir, overrides_path=overrides_path)
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
    # The scenario override, not the raw parameter dict: an edit that lands
    # partway through ships a schedule the simulator applies at its frame,
    # leaving the frames before it on the source video's own physics.
    physics = dsl.to_scenario_override(parsed, VOCAB)
    if isinstance(parsed, dsl.SetEdit):
        diff = {f"{parsed.property_name} ({parsed.object_id})":
                {"from": dsl.baseline_value_for(parsed, VOCAB), "to": parsed.to_value}}
    else:
        diff = {parsed.object_id: {"from": "present", "to": "removed"}}
    diff["timing"] = dsl.timing_diff(parsed, VOCAB)
    return {
        "edit_dsl": case.dsl,
        "edit_summary": case.edit_summary,
        "applies_from_frame": dsl.starts_at_frame(parsed),
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
        "applies_from_frame": edit_info["applies_from_frame"],
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
    # Timed edits name a frame, and the vocabulary is where that number is
    # bounded and turned into prompt wording. If the render length ever drifts
    # away from it, every "AT FRAME n" in the suite quietly means something
    # else, so it is checked here rather than discovered in a video.
    rendered_frames = int(round(float(args.duration_sec) * int(args.fps)))
    if rendered_frames != edit_vocab.TOTAL_FRAMES:
        raise SystemExit(
            f"{args.duration_sec}s at {args.fps} fps renders {rendered_frames} "
            f"frames, but edit_vocab.TOTAL_FRAMES says "
            f"{edit_vocab.TOTAL_FRAMES}. Update one to match the other."
        )
    args.out_root.mkdir(parents=True, exist_ok=True)

    keep_ids = {SOURCE_CASE_ID, *(c.case_id for c in EDIT_CASES)}
    if args.clean_stale_cases and not args.dry_run:
        clean_stale(args.out_root, keep_ids)

    manifest_path = args.out_root / "suite_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 3,
        "suite_name": "pcve_toy_car_ball_suite",
        "description": (
            "One source toy-car-hits-toy-ball video plus N edited variants. "
            "A toy car is pushed along a shelf and rear-ends a toy ball off "
            "the edge. Each edit is declared with a single edit-DSL string; "
            "prompts (precise + vague, zh + en), scenario overrides, and "
            "the physics diff are all derived from that one string."
        ),
        "baseline_physics": BASELINE_PHYSICS,
        "total_frames": edit_vocab.TOTAL_FRAMES,
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
        "description": {
            "vague": {
                "en": (
                    "The toy car is pushed along a shelf and rear-ends the "
                    "toy ball sitting near the far edge. The toy ball is "
                    "knocked over the edge and lands on the floor; the car "
                    "stays on the shelf."
                ),
                "zh": (
                    "玩具小车沿架子被推出,追尾停在远端边缘的玩具球。玩具球被撞下"
                    "架子落到地面;小车留在架子上。"
                ),
            },
            "quantitative": {
                "en": (
                    "The toy car, 0.35 kg, is pushed at 0.6 m/s along a shelf "
                    "and rear-ends the toy ball, 0.05 kg, sitting near the "
                    "far edge. The toy ball is knocked over the edge and "
                    "lands on the floor at x=-0.90, 0.89 m from where it sat. "
                    "The toy car stays on the shelf, stopping 0.40 m along."
                ),
                "zh": (
                    "0.35 kg 的玩具小车以 0.6 m/s 沿架子推出,追"
                    "尾停在远端边缘的 0.05 kg 玩具球。玩具球被撞下架子,"
                    "落到地面 x=-0.90 处,相对原位移动 0.89 m。玩"
                    "具小车留在架子上,前进 0.40 m 后停下。"
                ),
            },
        },
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
        render_case(args, case_dir=source_dir, seed=15001, overrides_path=None)
        source_record["status"] = "dry_run"
    else:
        t0 = time.perf_counter()
        print(f"[suite] render source {SOURCE_CASE_ID}")
        render_case(args, case_dir=source_dir, seed=15001, overrides_path=None)
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

        record: dict[str, Any] = {
            "case_id": case.case_id,
            "kind": "edit",
            "source_case_id": case.source_case_id,
            "seed": case.seed,
            "case_dir": str(case_dir.resolve()),
            "scenario_overrides_json": str(overrides_path.resolve()),
            "prompts_json": str(prompts_path.resolve()),
            "edit_dsl": edit_info["edit_dsl"],
            "edit_summary": edit_info["edit_summary"],
            "applies_from_frame": edit_info["applies_from_frame"],
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
            render_case(args, case_dir=case_dir, seed=case.seed, overrides_path=overrides_path)
            record["status"] = "dry_run"
            write_json(manifest_path, manifest)
            continue

        t0 = time.perf_counter()
        print(f"[suite] render edit {case.case_id}")
        try:
            render_case(args, case_dir=case_dir, seed=case.seed, overrides_path=overrides_path)
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
