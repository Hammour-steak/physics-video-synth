"""Build the picnic_apple_ball PCVE suite.

Each edit is declared as a single ``edit_dsl`` string. Prompts (precise +
vague, zh + en), scenario overrides, and the physics diff for the manifest
are all derived from that one string.

The scene: an apple falls onto a soccer ball resting on grass, hits the
top-side of the ball off-center, and torques it into a right-to-left roll
until grass friction brings it to rest. In the baseline the ball rolls
~0.54 m. Every value below was picked by sweeping simulate_picnic_apple_ball.py
directly; the numbers in each edit_summary are that sweep's output at the
suite's defaults.

Notes on the edit surface:
- The ball's `friction` is a CompoundBinding: the DSL edit scales both the
  lateral coefficient and the rolling coefficient in lockstep, since a
  rolling ball's decel comes from rolling friction and a lateral-only edit
  would leave the ball rolling forever.
- Restitution edits (apple or ball) barely change the roll distance in the
  sweep -- the collision is soft and momentum-driven -- so no restitution
  edit is in the suite.
- Both objects start at rest; there is no baseline direction for an
  `initial_velocity` edit on either.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))          # this scene
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # scripts root
import pcve_edit_dsl as dsl  # noqa: E402
import edit_vocab             # noqa: E402


WORKSPACE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_BLENDER = WORKSPACE_DIR / "tools" / "blender-3.6.23-linux-x64" / "blender"
RENDER_SCRIPT = WORKSPACE_DIR / "scripts" / "picnic_apple_ball" / "render_picnic_apple_ball.py"


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


SOURCE_CASE_ID = "picnic_apple_ball_baseline"


EDIT_CASES: tuple[EditCase, ...] = (
    EditCase(
        case_id="edit_heavy_soccer_ball",
        source_case_id=SOURCE_CASE_ID,
        seed=8101,
        dsl="SET soccer_ball.mass TIMES 4",
        edit_summary=(
            "Soccer ball made 4x heavier. The same apple impact now delivers "
            "a much smaller change of momentum, so the ball barely reacts: "
            "it rolls only 0.04 m before stopping instead of the baseline's "
            "0.54 m."
        ),
    ),
    EditCase(
        case_id="edit_slick_soccer_ball",
        source_case_id=SOURCE_CASE_ID,
        seed=8102,
        dsl="SET soccer_ball.friction TIMES 0.4",
        edit_summary=(
            "Soccer ball's friction cut 2.5x. Both the lateral coefficient and "
            "the rolling one scale together, so after the same apple impact "
            "the ball meets much less grip on the grass and rolls 1.25 m -- "
            "more than 2x the 0.54 m baseline -- before settling within the shot."
        ),
    ),
    EditCase(
        case_id="edit_grippy_soccer_ball",
        source_case_id=SOURCE_CASE_ID,
        seed=8103,
        dsl="SET soccer_ball.friction TIMES 3",
        edit_summary=(
            "Soccer ball's friction tripled. The grass grips the ball hard "
            "enough to damp the roll almost immediately: it moves only 0.04 m "
            "before stopping, vs 0.54 m in the baseline. Different mechanism "
            "from the heavy-ball edit -- same visible outcome, but the ball "
            "here has normal inertia and just cannot overcome the friction."
        ),
    ),
    EditCase(
        case_id="edit_heavy_apple",
        source_case_id=SOURCE_CASE_ID,
        seed=8104,
        dsl="SET apple.mass TIMES 2",
        edit_summary=(
            "Apple made twice as heavy. The off-centre hit carries more "
            "momentum into the ball: it rolls 1.66 m, about three times "
            "the baseline's 0.54 m, and comes to rest before the shot "
            "ends. The apple itself is slowed by the exchange, covering "
            "2.07 m against 2.37 m."
        ),
    ),
    EditCase(
        case_id="edit_remove_apple",
        source_case_id=SOURCE_CASE_ID,
        seed=8105,
        dsl="DELETE apple",
        edit_summary=(
            "Apple removed. Nothing falls; the soccer ball sits undisturbed "
            "on the grass for the whole shot. Roll distance 0 m."
        ),
    ),
    EditCase(
        case_id="edit_remove_soccer_ball",
        source_case_id=SOURCE_CASE_ID,
        seed=8106,
        dsl="DELETE soccer_ball",
        edit_summary=(
            "Soccer ball removed. The apple falls the full 1.3 m from the "
            "branch and lands on the grass beside the picnic blanket instead "
            "of on top of a ball -- a single, isolated apple drop rather "
            "than an oblique hit and a roll."
        ),
    ),
    # The one edit in this suite that does not hold for the whole clip, and
    # deliberately the same DELETE as edit_remove_apple: they differ by the AT
    # FRAME clause alone. Taking the apple away after it has already struck
    # the ball is what makes the timing readable -- removing it beforehand
    # leaves a clip in which nothing moves at all.
    EditCase(
        case_id="edit_remove_apple_after_impact",
        source_case_id=SOURCE_CASE_ID,
        seed=10107,
        dsl="DELETE apple AT FRAME 18",
        edit_summary=(
            "Apple removed at frame 18, four frames after it reaches the "
            "soccer ball at frame 14. Frames 1-17 are the source video frame "
            "for frame, impact included: the ball is knocked back to 0.64 m/s "
            "and the apple is 0.28 m past its start when it disappears. The "
            "ball rolls on to x=-0.54 exactly as in the source, with nothing "
            "left on the grass to follow it. The whole-clip version of the "
            "same delete is a different video again: with no apple to arrive, "
            "the ball never moves at all."
        ),
    ),
)


# --------------------------------------------------------------------- CLI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the picnic_apple_ball PCVE suite (1 source + N edits)."
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=WORKSPACE_DIR / "renders" / "pcve_picnic_apple_ball_suite",
    )
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--resolution", nargs=2, type=int, default=(1280, 720))
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-sec", type=float, default=4.0)
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
    video_source = case_dir / "picnic_apple_ball.mp4"
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
        "suite_name": "pcve_picnic_apple_ball_suite",
        "description": (
            "One source picnic apple-onto-ball video plus N edited variants. "
            "Each edit is declared with a single edit-DSL string; prompts "
            "(precise + vague, zh + en), scenario overrides, and the physics "
            "diff are all derived from that one string."
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
                    "The apple falls from an overhead branch onto the soccer "
                    "ball resting on the grass. It lands off-centre, torquing "
                    "the ball into a roll until friction stops it, and "
                    "bounces away itself."
                ),
                "zh": (
                    "苹果从头顶的树枝落到草地上的足球上,落点偏离球心,把足球拧得"
                    "滚起来,滚一段后被摩擦停住;苹果自己则弹开。"
                ),
            },
            "quantitative": {
                "en": (
                    "The apple falls from an overhead branch onto the soccer "
                    "ball resting on the grass, landing 0.105 m off the "
                    "ball's centre. That lever arm is enough to torque the "
                    "ball into a roll: the soccer ball travels 0.54 m across "
                    "the grass before friction stops it, while the apple "
                    "bounces on to end 2.37 m from where it fell."
                ),
                "zh": (
                    "苹果从头顶的树枝落到草地上的足球上,落点偏离球心 0.105"
                    " m。这个力臂足以把球拧得滚起来:足球在草地上滚 0.54 "
                    "m 后被摩擦停住,苹果则弹开,相对落点移动 2.37 m。"
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
        render_case(args, case_dir=source_dir, seed=8001, overrides_path=None)
        source_record["status"] = "dry_run"
    else:
        t0 = time.perf_counter()
        print(f"[suite] render source {SOURCE_CASE_ID}")
        render_case(args, case_dir=source_dir, seed=8001, overrides_path=None)
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
