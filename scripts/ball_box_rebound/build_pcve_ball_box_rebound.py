"""Build the ball_box_rebound PCVE suite.

Each edit is declared as a single ``edit_dsl`` string. Prompts (precise +
vague, zh + en), scenario overrides, and the physics diff for the manifest
are all derived from that one string.

The scene is a three-beat shot: a ball is rolled across the boards, bounces off
the toy chest's front panel at 30 deg off the normal, and the rebound crosses
the room into a little football, which it knocks 0.30 m clear.

What the suite is built on is that a bounce scales the *normal* component of
the velocity and leaves the tangential one alone. So changing how lively the
bounce is changes the **direction** the ball leaves at, not just its speed --
the rebound comes off at 29 deg in the source and at 65 deg once the
ball's restitution drops, and at those angles it runs along the front of the chest and
past the football entirely. A model that reads "less bouncy" as "same path,
slower" gets those cases wrong.

One edit is deliberately a near-twin in outcome and opposite in cause:
`edit_soft_push` still reaches the football, but the rebound angle is untouched
(29.4 deg) -- speed does not bend the path. Only the approach speed and the
angle off the panel separate it from the restitution cases.

Every value below was picked by sweeping simulate_ball_box_rebound.py directly;
the numbers quoted in each edit_summary are that sweep's output at this suite's
own defaults.
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
RENDER_SCRIPT = (
    WORKSPACE_DIR / "scripts" / "ball_box_rebound" / "render_ball_box_rebound.py"
)


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


SOURCE_CASE_ID = "ball_box_rebound_baseline"
SOURCE_SEED = 6001


EDIT_CASES: tuple[EditCase, ...] = (
    EditCase(
        case_id="edit_dead_ball",
        source_case_id=SOURCE_CASE_ID,
        seed=6102,
        dsl="SET ball_a.restitution TIMES 0.25",
        edit_summary=(
            "Star ball made far less bouncy, so its rebound off the chest "
            "panel flattens out and it comes away on a much shallower "
            "line. It runs 1.66 m from its start against the baseline's "
            "1.10 m and misses the football entirely: the little football "
            "never moves, where the baseline nudges it 0.30 m."
        ),
    ),
    EditCase(
        case_id="edit_soft_push",
        source_case_id=SOURCE_CASE_ID,
        seed=6103,
        dsl="SET ball_a.initial_velocity TIMES 0.75",
        edit_summary=(
            "A gentler push, aimed identically. This is the suite's distractor: "
            "the bounce geometry is untouched -- 29.4 deg out against the "
            "source's 29.2, because speed does not bend the rebound -- and the "
            "ball still reaches the football. But everything happens later and "
            "weaker: the chest is met on frame 17 instead of 13, the football "
            "on frame 38 instead of 24, and it is nudged 0.05 m rather than "
            "knocked 0.30 m clear."
        ),
    ),
    EditCase(
        case_id="edit_draggy_ball",
        source_case_id=SOURCE_CASE_ID,
        seed=6104,
        dsl="SET ball_a.friction TIMES 5",
        edit_summary=(
            "The rolling ball made to drag on the boards -- a scuffed, tacky "
            "ball rather than a changed floor. Because Bullet builds rolling "
            "resistance from `rf_floor * lateral_ball` and the ball carries no "
            "rolling friction of its own, this raises the effective resistance "
            "from 0.006 to 0.030 while leaving the bounce off the chest alone. "
            "The ball loses most of the run-up before it arrives: it reaches "
            "the chest on frame 18 at 0.85 m/s instead of frame 13 at 2.36 m/s. "
            "It still bounces, and still at 30.0 deg, but comes off with so "
            "little left that it stops short of the football and settles by "
            "frame 22, less than a second into the shot."
        ),
    ),
    EditCase(
        case_id="edit_heavy_target",
        source_case_id=SOURCE_CASE_ID,
        seed=6105,
        dsl="SET ball_b.mass TIMES 10",
        edit_summary=(
            "The football made 10x heavier. The roll, the bounce and the "
            "rebound line are identical to the source, and the ball still "
            "arrives at 1.10 m/s -- but the football barely twitches, moving "
            "0.013 m instead of 0.300, and the star ball rebounds back off it "
            "at 0.57 m/s instead of stopping dead at 0.19."
        ),
    ),
    EditCase(
        case_id="edit_heavy_ball",
        source_case_id=SOURCE_CASE_ID,
        seed=6106,
        dsl="SET ball_a.mass TIMES 10",
        edit_summary=(
            "The rolling ball made 10x heavier. The bounce off the chest is "
            "unchanged -- restitution sets the rebound, not mass -- but the "
            "ball-on-ball hit is one-sided: the football leaves at 1.65 m/s "
            "instead of 0.91 and is driven 0.84 m instead of 0.30, while the "
            "star ball carries on through at 0.89 m/s rather than being "
            "stopped. Both balls are still rolling when the clip ends."
        ),
    ),
    EditCase(
        case_id="edit_remove_target",
        source_case_id=SOURCE_CASE_ID,
        seed=6107,
        dsl="DELETE ball_b",
        edit_summary=(
            "The football removed. The roll and the bounce are identical to the "
            "source, and then the rebound crosses empty floor: with nothing to "
            "hit, the ball keeps its 1.72 m/s and rolls on towards the camera, "
            "leaving frame at the bottom edge around frame 48 and coming to "
            "rest off-camera. The last second of the clip is bare floor."
        ),
    ),
    # The one edit in this suite that does not hold for the whole clip, and
    # deliberately the same DELETE as edit_remove_target: they differ by the
    # AT FRAME clause alone. Taking the football away after it has been hit is
    # what makes the timing readable -- removing it beforehand leaves the star
    # ball nothing to hand its speed to.
    EditCase(
        case_id="edit_remove_target_after_impact",
        source_case_id=SOURCE_CASE_ID,
        seed=12107,
        dsl="DELETE ball_b AT FRAME 32",
        edit_summary=(
            "Football removed at frame 32, eight frames after the star ball "
            "reaches it at frame 24. Frames 1-31 are the source video frame "
            "for frame: the star ball rebounds off the toy chest at frame 13, "
            "runs into the football and is left creeping at 0.07 m/s, and the "
            "football has been driven 0.18 m when it disappears. The star "
            "ball then settles at (+0.27, +0.62) exactly as in the source. "
            "The whole-clip version of the same delete is a different video "
            "again: with no football to hit, the star ball keeps 1.0 m/s off "
            "the rebound and coasts out to (+0.99, +1.39)."
        ),
    ),
)


# --------------------------------------------------------------------- CLI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the ball_box_rebound PCVE suite (1 source + N edits)."
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=WORKSPACE_DIR / "renders" / "pcve_ball_box_rebound_suite",
    )
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--resolution", nargs=2, type=int, default=(1280, 720))
    parser.add_argument("--fps", type=int, default=24)
    # One length for every case, source included. The edits are meant to be
    # compared frame against frame, so a per-case duration -- which the older
    # scenario suite used -- would put a difference in the pair that no edit
    # asked for. 3.0 s is long enough for the source to settle (frame 45).
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
    video_source = case_dir / "ball_box_rebound.mp4"
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
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )
    # The renderer runs its own physics sanity checks and prints a one-line
    # summary. Surface both, or a case that quietly degraded looks identical to
    # a good one in the manifest.
    for line in (result.stdout or "").splitlines():
        if line.startswith(("[WARN]", "[NOTE]", "[SIM]")):
            print(f"  {line}")


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
        "suite_name": "pcve_ball_box_rebound_suite",
        "description": (
            "One source ball-off-toy-chest rebound video plus N edited "
            "variants. Each edit is declared with a single edit-DSL string; "
            "prompts (precise + vague, zh + en), scenario overrides, and the "
            "physics diff are all derived from that one string."
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
                    "The star ball is rolled at the toy chest's front panel, "
                    "comes off it at an angle, and goes on to hit the little "
                    "football and knock it clear. Both have settled before "
                    "the clip ends."
                ),
                "zh": (
                    "星星球滚向玩具箱的正面板,斜着弹开后撞上小足球并把它撞开。片"
                    "尾前两球都已静止。"
                ),
            },
            "quantitative": {
                "en": (
                    "The star ball is rolled at 2.60 m/s at the toy chest's "
                    "front panel, comes off it at an angle, and goes on to "
                    "hit the little football, knocking it 0.30 m clear. The "
                    "star ball ends 1.10 m from where it started; both have "
                    "settled before the clip ends."
                ),
                "zh": (
                    "星星球以 2.60 m/s 滚向玩具箱的正面板,斜着弹开后撞"
                    "上小足球,把它撞开 0.30 m。星星球相对起点移动 1.1"
                    "0 m,片尾前两球都已静止。"
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
        source_record["outputs"] = standardize_render_outputs(
            source_dir, has_overrides=False
        )
    elif args.dry_run:
        render_case(args, case_dir=source_dir, seed=SOURCE_SEED, overrides_path=None)
        source_record["status"] = "dry_run"
    else:
        t0 = time.perf_counter()
        print(f"[suite] render source {SOURCE_CASE_ID}")
        render_case(args, case_dir=source_dir, seed=SOURCE_SEED, overrides_path=None)
        source_record["outputs"] = standardize_render_outputs(
            source_dir, has_overrides=False
        )
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
            render_case(
                args, case_dir=case_dir, seed=case.seed, overrides_path=overrides_path
            )
            record["status"] = "dry_run"
            write_json(manifest_path, manifest)
            continue

        t0 = time.perf_counter()
        print(f"[suite] render edit {case.case_id}")
        try:
            render_case(
                args, case_dir=case_dir, seed=case.seed, overrides_path=overrides_path
            )
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
