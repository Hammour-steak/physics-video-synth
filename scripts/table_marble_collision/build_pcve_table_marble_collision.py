"""Build the table_marble_collision PCVE suite.

Each edit is declared as a single ``edit_dsl`` string. Prompts (precise +
vague, zh + en), scenario overrides, and the physics diff for the manifest
are all derived from that one string.

The scene: a 100 mm glass marble is rolled along a bar table at 1.31 m/s
into a 50 mm glass marble sitting near the far end. Both marbles are the
same glass; at baseline the small one weighs 1/8 of the big one (mass goes
as the cube of the radius), and the big marble arrives at ~0.83 m/s and
sends the small one skittering forward while itself continuing on. The
suite varies mass ratio (directly via `mass`, not by resizing the marble),
push speed, and restitution to show the full range a two-body impact has
to offer -- from the small marble rocketing off, to it barely twitching,
to a clean miss when the push runs out.

Every value below was picked by sweeping simulate_table_marble_collision.py
directly; the numbers in each edit_summary are that sweep's output at the
suite's defaults.

Notes on the edit surface:
- Only the small marble can be DELETEd. The big one drives the roll off
  the far end, and the sim is not built to run without it.
- The big marble is the only one with a non-zero baseline velocity, so
  `initial_velocity` is bound only on it (via launch_speed).
- Restitution edits on the big marble also affect its floor bounce and
  are less isolated visually; only the small-marble restitution is in
  the suite.
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
RENDER_SCRIPT = WORKSPACE_DIR / "scripts" / "table_marble_collision" / "render_table_marble_collision.py"


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
    duration_sec: float | None = None


SOURCE_CASE_ID = "table_marble_collision_baseline"


EDIT_CASES: tuple[EditCase, ...] = (
    EditCase(
        case_id="edit_heavy_small_marble",
        source_case_id=SOURCE_CASE_ID,
        seed=17101,
        dsl="SET small_marble.mass TIMES 5",
        edit_summary=(
            "Small marble made five times heavier, so the mass ratio "
            "drops from 8:1 to about 1.6:1. It barely reacts to the hit "
            "-- 0.29 m against the baseline's 0.68 m -- while the big "
            "marble is checked hard on impact and rolls only 0.70 m "
            "instead of 0.89 m."
        ),
    ),
    EditCase(
        case_id="edit_heavy_big_marble",
        source_case_id=SOURCE_CASE_ID,
        seed=17102,
        dsl="SET big_marble.mass TIMES 5",
        edit_summary=(
            "Big marble made 5x heavier (1.309 -> 6.545 kg). The mass "
            "ratio is now 40:1 instead of 8:1, so the impact drives even "
            "more momentum into the small marble: it rockets 0.85 m along "
            "the table (vs 0.68 m baseline). The big marble follows through "
            "almost undisturbed."
        ),
    ),
    EditCase(
        case_id="edit_matched_pair",
        source_case_id=SOURCE_CASE_ID,
        seed=17103,
        dsl="SET big_marble.mass TIMES 0.125",
        edit_summary=(
            "Big marble made exactly as heavy as the small one: it is "
            "twice the radius, so an eighth of its mass is the small "
            "one's mass, and the ratio goes from 8:1 to 1:1. The hit "
            "becomes an equal-mass exchange -- the big one is stopped "
            "short at 0.66 m against the baseline's 0.89 m, and the small "
            "one is nudged only 0.18 m, because table friction damps most "
            "of the transfer between two light marbles."
        ),
    ),
    EditCase(
        case_id="edit_soft_push",
        source_case_id=SOURCE_CASE_ID,
        seed=17104,
        dsl="SET big_marble.initial_velocity TIMES 0.7",
        edit_summary=(
            "Big marble pushed more gently. It slows against table "
            "friction and stops short of the small marble -- 0.49 m of "
            "travel against the baseline's 0.89 -- so no contact happens "
            "at all. The small marble sits untouched for the whole shot."
        ),
    ),
    EditCase(
        case_id="edit_dead_small_marble",
        source_case_id=SOURCE_CASE_ID,
        seed=17105,
        dsl="SET small_marble.restitution TIMES 0.05",
        edit_summary=(
            "Small marble's restitution killed. The impact goes almost "
            "fully inelastic: the small marble absorbs the collision "
            "instead of springing away, moving 0.26 m against the "
            "baseline's 0.68 m, while the big marble is barely checked "
            "and drives on to 0.93 m rather than being handed off "
            "cleanly."
        ),
    ),
    EditCase(
        case_id="edit_remove_small_marble",
        source_case_id=SOURCE_CASE_ID,
        seed=17106,
        dsl="DELETE small_marble",
        edit_summary=(
            "Small marble removed. The big marble rolls across the bar "
            "table unobstructed -- 1.01 m of travel instead of the "
            "baseline's 0.89 m -- and never hits anything."
        ),
    ),
)


# --------------------------------------------------------------------- CLI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the table_marble_collision PCVE suite (1 source + N edits)."
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=WORKSPACE_DIR / "renders" / "pcve_table_marble_collision_suite",
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
    video_source = case_dir / "table_marble_collision.mp4"
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
        "suite_name": "pcve_table_marble_collision_suite",
        "description": (
            "One source big/small marble-collision video plus N edited "
            "variants. A 100 mm glass marble is rolled into a 50 mm one on "
            "the bar top. Each edit is declared with a single edit-DSL "
            "string; prompts (precise + vague, zh + en), scenario overrides, "
            "and the physics diff are all derived from that one string."
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
                    "The big marble is rolled along the bar table into the "
                    "small marble sitting near the far end. The small marble "
                    "is sent forward and the big marble carries on behind it."
                ),
                "zh": (
                    "大玻璃球沿吧台滚向停在远端的小玻璃球。小玻璃球被撞得向前跑,"
                    "大玻璃球在后面继续前进。"
                ),
            },
            "quantitative": {
                "en": (
                    "The big marble, 100 mm across, is rolled at 1.31 m/s "
                    "along the bar table into the small marble, 50 mm across, "
                    "sitting near the far end. Both are the same glass, so "
                    "the small marble is an eighth of the mass: the impact "
                    "sends it 0.68 m forward while the big marble carries on "
                    "for 0.89 m in all."
                ),
                "zh": (
                    "100 mm 的大玻璃球以 1.31 m/s 沿吧台滚向停在"
                    "远端的 50 mm 小玻璃球。两球是同一种玻璃,小玻璃球质量"
                    "只有八分之一:撞击把它送出 0.68 m,大玻璃球则继续前进"
                    ",全程 0.89 m。"
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
        render_case(args, case_dir=source_dir, seed=17001, overrides_path=None,
                    duration_sec=float(args.duration_sec))
        source_record["status"] = "dry_run"
    else:
        t0 = time.perf_counter()
        print(f"[suite] render source {SOURCE_CASE_ID}")
        render_case(args, case_dir=source_dir, seed=17001, overrides_path=None,
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
