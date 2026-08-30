"""Build the air_hockey_chain PCVE suite.

Each edit is declared as a single ``edit_dsl`` string. Prompts (precise +
vague, zh + en), scenario overrides, and the physics diff for the manifest
are all derived from that one string.

The scene is a relay: three identical air-hockey mallets on the quarter, half
and three-quarter points of the table, and one push on the far one. It works
only because of three properties of a real air-hockey table -- the mallets are
identical, their faces are hard, and the air cushion makes the surface nearly
frictionless -- and the edits below break them one disc at a time. Every value
was picked by sweeping simulate_air_hockey_chain.py directly; the numbers
quoted in each edit_summary are that sweep's output at the suite's defaults.
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
    WORKSPACE_DIR / "scripts" / "air_hockey_chain" / "render_air_hockey_chain.py"
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


SOURCE_CASE_ID = "air_hockey_chain_baseline"


EDIT_CASES: tuple[EditCase, ...] = (
    EditCase(
        case_id="edit_heavy_middle_mallet",
        source_case_id=SOURCE_CASE_ID,
        seed=4101,
        dsl="SET red_mallet.mass FROM 0.12 TO 0.48",
        edit_summary=(
            "Middle mallet made 4x heavier. Momentum is no longer cancelled at "
            "the first impact: blue rebounds backwards up the table instead of "
            "stopping dead on red, ending further from the camera than it "
            "started. Red leaves with 0.30 m/s instead of 0.75, and the chain "
            "still reaches white but carries only 31% of the push instead of 88%."
        ),
    ),
    EditCase(
        case_id="edit_heavy_last_mallet",
        source_case_id=SOURCE_CASE_ID,
        seed=4102,
        dsl="SET white_mallet.mass FROM 0.12 TO 1.2",
        edit_summary=(
            "Last mallet made 10x heavier. The first handoff is untouched -- "
            "blue still stops dead on red -- but the second one fails: red "
            "bounces back up the table off white instead of stopping, and white "
            "only creeps forward at 0.13 m/s rather than carrying the push into "
            "the near rail."
        ),
    ),
    EditCase(
        case_id="edit_heavy_striker",
        source_case_id=SOURCE_CASE_ID,
        seed=4103,
        dsl="SET blue_mallet.mass FROM 0.12 TO 1.2",
        edit_summary=(
            "The pushed mallet made 10x heavier. Blue keeps 80% of its speed "
            "through the impact and drives on behind red rather than stopping, "
            "and red leaves faster than blue arrived (1.37 m/s against a 0.8 m/s "
            "push). The whole chain accelerates and all three finish bunched "
            "together against the near rail instead of strung out along the "
            "table where each handoff happened."
        ),
    ),
    EditCase(
        case_id="edit_dead_striker_face",
        source_case_id=SOURCE_CASE_ID,
        seed=4104,
        dsl="SET blue_mallet.restitution FROM 0.95 TO 0.35",
        edit_summary=(
            "Soft face on the pushed mallet only, so the first impact is a "
            "shove (pair restitution 0.33 instead of 0.90) and the second is "
            "unchanged. Blue keeps 35% of its speed instead of 3% -- it does not "
            "stop, it slides on behind red -- and red takes only 0.50 m/s down "
            "the table instead of 0.75."
        ),
    ),
    EditCase(
        case_id="edit_grippy_striker",
        source_case_id=SOURCE_CASE_ID,
        seed=4105,
        dsl="SET blue_mallet.friction FROM 0.06 TO 1.5",
        edit_summary=(
            "The pushed mallet's base made to grip instead of glide -- as if it "
            "alone had lost its air cushion. PyBullet multiplies the pair, so "
            "the effective coefficient against the table goes from 0.0036 to "
            "0.09. Blue decelerates and stops after 36 cm, well short of red. "
            "No collision happens at all: the disc riding on almost nothing is "
            "what makes the relay possible in the first place, not a detail of "
            "it. Red and white never move, and both still glide freely."
        ),
    ),
    EditCase(
        case_id="edit_soft_push",
        source_case_id=SOURCE_CASE_ID,
        seed=4106,
        dsl="SET blue_mallet.initial_velocity FROM 0.8 TO 0.3",
        edit_summary=(
            "Everything correct but the push is too gentle. Blue still coasts "
            "the length of the gap and still stops dead on red -- the collision "
            "physics is untouched -- but red leaves with only 0.22 m/s, runs out "
            "of momentum 24 cm short of white, and the chain dies one link "
            "early. White never moves."
        ),
    ),
    EditCase(
        case_id="edit_remove_middle_mallet",
        source_case_id=SOURCE_CASE_ID,
        seed=4107,
        dsl="DELETE red_mallet",
        edit_summary=(
            "Middle mallet removed. Blue coasts through the empty middle of the "
            "table and strikes white directly, losing so little to the surface "
            "on the way that the single handoff is cleaner than either of the "
            "baseline's two: white leaves with 90% of the original push. One "
            "impact instead of two, and it happens later and nearer the camera."
        ),
    ),
    EditCase(
        case_id="edit_remove_last_mallet",
        source_case_id=SOURCE_CASE_ID,
        seed=4108,
        dsl="DELETE white_mallet",
        edit_summary=(
            "Last mallet removed. The first handoff is identical to the "
            "baseline -- blue stops dead, red leaves at 0.75 m/s -- but red then "
            "runs the rest of the table unobstructed and into the near rail "
            "instead of handing over at the three-quarter point."
        ),
    ),
)


# --------------------------------------------------------------------- CLI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the air_hockey_chain PCVE suite (1 source + N edits)."
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=WORKSPACE_DIR / "renders" / "pcve_air_hockey_chain_suite",
    )
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--resolution", nargs=2, type=int, default=(1280, 720))
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-sec", type=float, default=4.0)
    # 64, not the 32 most scenes use: this one is lit by a single practical
    # pendant in a dark room, and at 32 the shadowed side of the cabinet and
    # the tiled floor still carry visible denoiser blotching.
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
    video_source = case_dir / "air_hockey_chain.mp4"
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
        "suite_name": "pcve_air_hockey_chain_suite",
        "description": (
            "One source air-hockey relay video plus N edited variants. Each edit "
            "is declared with a single edit-DSL string; prompts (precise + vague, "
            "zh + en), scenario overrides, and the physics diff are all derived "
            "from that one string."
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
            "Source video: default parameters. Three identical mallets sit on "
            "the quarter, half and three-quarter points of an arcade air-hockey "
            "table; the far one is pushed at 0.8 m/s, and each striker stops "
            "dead where it hits, handing its speed to the next. White carries "
            "88% of the original push into the near rail."
        ),
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
        render_case(args, case_dir=source_dir, seed=4001, overrides_path=None)
        source_record["status"] = "dry_run"
    else:
        t0 = time.perf_counter()
        print(f"[suite] render source {SOURCE_CASE_ID}")
        render_case(args, case_dir=source_dir, seed=4001, overrides_path=None)
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
