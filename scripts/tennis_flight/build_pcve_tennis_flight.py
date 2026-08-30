"""Build the tennis_flight PCVE suite.

Each edit is declared as a single ``edit_dsl`` string. Prompts (precise +
vague, zh + en), scenario overrides, and the physics diff for the manifest
are all derived from that one string.

The scene: a tennis ball is launched from (-4, 0, 1.5) with initial
velocity (4.5, 0, 4.5) m/s. It arcs to about z=2.5 m over the net at
x=0, first touches down at x≈1.05 (t≈1.17 s), and rolls out to x≈3.3
before stopping. Every value below was picked by sweeping
simulate_tennis_flight.py directly at the suite defaults
(duration 4 s, gravity -9.8 m/s²); the numbers in each edit_summary
are that sweep's output.

Notes on the edit surface:
- No DELETE binding: the scene has one moving object and removing it
  leaves an empty court, which is not a physically meaningful edit.
- Mass edits are intentionally not in the suite. In this scene the ball
  falls under gravity (mass-independent trajectory) and bounces off an
  infinite-mass floor (impulse independent of ball mass), so changing
  the ball's mass alone produces no visible difference -- a null edit
  under skill hard rule 5. `friction`, `restitution`, and
  `initial_velocity` are what govern the flight and the roll-out.
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
RENDER_SCRIPT = WORKSPACE_DIR / "scripts" / "tennis_flight" / "render_tennis_flight.py"


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


SOURCE_CASE_ID = "tennis_flight_baseline"


EDIT_CASES: tuple[EditCase, ...] = (
    EditCase(
        case_id="edit_slick_ball",
        source_case_id=SOURCE_CASE_ID,
        seed=13101,
        dsl="SET ball.friction FROM 0.5 TO 0.1",
        edit_summary=(
            "Ball's friction cut 5x -- both lateral and rolling coefficients "
            "scale together. After the same landing at x=1.06 the ball meets "
            "almost no grip on the court and is still rolling fast at the "
            "end of the 4 s window, having reached x=8.8 m -- roughly 2.6x "
            "the baseline's 3.3 m roll-out."
        ),
    ),
    EditCase(
        case_id="edit_grippy_ball",
        source_case_id=SOURCE_CASE_ID,
        seed=13102,
        dsl="SET ball.friction FROM 0.5 TO 1.5",
        edit_summary=(
            "Ball's friction tripled. After the same landing at x=1.05 the "
            "high-friction contact eats the roll almost immediately: the "
            "ball stops essentially at its landing point (final x=1.38 m "
            "vs 3.31 m baseline), no meaningful run-out."
        ),
    ),
    EditCase(
        case_id="edit_bouncy_ball",
        source_case_id=SOURCE_CASE_ID,
        seed=13103,
        dsl="SET ball.restitution FROM 0.05 TO 0.9",
        edit_summary=(
            "Ball's restitution 18x higher -- from a dead thud (0.05) to a "
            "near-perfect bounce (0.9). The first landing at x=1.05 is the "
            "same, but instead of pancaking the ball hops several times "
            "along +X before settling (final x=3.48 m vs 3.31 m baseline), "
            "with the visible bounces being the tell."
        ),
    ),
    EditCase(
        case_id="edit_soft_serve",
        source_case_id=SOURCE_CASE_ID,
        seed=13104,
        dsl="SET ball.initial_velocity FROM 4.5 TO 2.5",
        edit_summary=(
            "Launch speed cut nearly in half (4.5 -> 2.5 m/s along +X). "
            "The ball no longer clears the net: it lands short at x=-1.16 m "
            "(net is at x=0) and rolls out to only x=-0.49 m, versus the "
            "baseline landing at x=1.05 m and finishing at x=3.31 m."
        ),
    ),
    EditCase(
        case_id="edit_hard_serve",
        source_case_id=SOURCE_CASE_ID,
        seed=13105,
        dsl="SET ball.initial_velocity FROM 4.5 TO 6.5",
        edit_summary=(
            "Launch speed ~1.4x faster (4.5 -> 6.5 m/s). The ball flies "
            "further before touching down (landing at x=3.23 m instead of "
            "x=1.05 m) and, still carrying energy after landing, rolls out "
            "to x=7.11 m by the end of the 4 s window -- more than 2x the "
            "baseline's 3.31 m final position."
        ),
    ),
)


# --------------------------------------------------------------------- CLI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the tennis_flight PCVE suite (1 source + N edits)."
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=WORKSPACE_DIR / "renders" / "pcve_tennis_flight_suite",
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
    video_source = case_dir / "tennis_flight.mp4"
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
        "suite_name": "pcve_tennis_flight_suite",
        "description": (
            "One source tennis-flight video plus N edited variants. A tennis "
            "ball is launched with horizontal velocity across a court and "
            "arcs under gravity onto the ground. Each edit is declared with a "
            "single edit-DSL string; prompts (precise + vague, zh + en), "
            "scenario overrides, and the physics diff are all derived from "
            "that one string."
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
            "Source video: default parameters. A tennis ball is launched "
            "from (-4, 0, 1.5) with initial velocity (4.5, 0, 4.5) m/s. "
            "It arcs to about z=2.5 m over the net at x=0, first touches "
            "down at x=1.05 (t=1.17 s), and rolls out to x=3.31 by the "
            "end of the 4 s window."
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
        render_case(args, case_dir=source_dir, seed=13001, overrides_path=None)
        source_record["status"] = "dry_run"
    else:
        t0 = time.perf_counter()
        print(f"[suite] render source {SOURCE_CASE_ID}")
        render_case(args, case_dir=source_dir, seed=13001, overrides_path=None)
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
