"""Build the ball_block PCVE suite.

Each edit is declared as a single ``edit_dsl`` string. Prompts (precise +
vague, zh + en), scenario overrides, and the physics diff for the manifest
are all derived from that one string.

The scene is a single collision: a rubber ball rolls across a wooden floor at
6 m/s and shoves a wooden block roughly 1.5 m. Every edit below changes exactly
one scalar on one object (or removes the block) and was chosen by sweeping
simulate_ball_block_impact.py directly -- the numbers in each edit_summary are
that sweep's output at the suite's own defaults.

Clips are 3 s. The block comes to rest well inside that in every case, so the
block-displacement figures below are final; the ball does not always, and where
it is still rolling when the clip ends the summary says so rather than quoting
a resting place the video never shows.

The suite renders with ``--physics-jitter 0``. That is not a detail: the
renderer's create_scenario() jitters most physics values, and with jitter on,
an edit's declared "from" value would not be the value the source video
actually used.

Friction is one knob per object, not split into lateral and rolling: an edit
to ``ball.friction`` scales both the ball's lateral and rolling coefficients
by the same ratio, and the value shown is the object's lateral friction. This
means the older ``edit_draggy_ball`` case -- which relied on tuning rolling
friction *alone* to leave the collision untouched and change only the ball's
run-out -- is no longer expressible; scaling both together dampens the
approach too. It has been dropped rather than turned into a duplicate of
``edit_slow_ball``.
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
RENDER_SCRIPT = WORKSPACE_DIR / "scripts" / "ball_block" / "render_ball_block_impact.py"

# The renderer supports two motions; this suite is built entirely on the side
# impact, because a rolling collision is what the edits below are about.
MOTION = "side_impact"
BLOCK_TEXTURE_ASSET = "wood_table"


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


SOURCE_CASE_ID = "ball_block_baseline"
SOURCE_SEED = 5001


EDIT_CASES: tuple[EditCase, ...] = (
    EditCase(
        case_id="edit_heavy_block",
        source_case_id=SOURCE_CASE_ID,
        seed=5101,
        dsl="SET wood_block.mass FROM 0.65 TO 6.5",
        edit_summary=(
            "Block made 10x heavier. It barely moves -- 6 cm instead of 1.47 m -- "
            "and the ball rebounds off it back the way it came instead of "
            "following it across the floor. The contact happens at the same "
            "moment and the same place as in the source."
        ),
    ),
    EditCase(
        case_id="edit_heavy_ball",
        source_case_id=SOURCE_CASE_ID,
        seed=5102,
        dsl="SET ball.mass FROM 0.58 TO 5.8",
        edit_summary=(
            "Ball made 10x heavier. It plows through the impact barely slowed "
            "(3.35 m/s out of 4.38 m/s in, against 0.65 m/s in the source) and "
            "launches the block at 5.37 m/s, driving it 4.73 m instead of "
            "1.47 m -- the block leaves frame rather than sliding to a stop. "
            "Both it and the ball are still travelling when the clip ends."
        ),
    ),
    EditCase(
        case_id="edit_slow_ball",
        source_case_id=SOURCE_CASE_ID,
        seed=5103,
        dsl="SET ball.initial_velocity FROM 6 TO 4",
        edit_summary=(
            "Ball launched at 4 m/s instead of 6. Rolling resistance eats most "
            "of the difference on the way across, so it arrives at 1.68 m/s "
            "rather than 4.38 -- the contact is a nudge, not an impact. It "
            "lands at frame 26 instead of 14 and the block shifts 28 cm instead "
            "of 1.47 m."
        ),
    ),
EditCase(
        case_id="edit_grippy_block",
        source_case_id=SOURCE_CASE_ID,
        seed=5105,
        dsl="SET wood_block.friction FROM 0.32 TO 1.5",
        edit_summary=(
            "Block's friction against the floor raised, as if it had a rubber "
            "underside. It is struck just as hard -- it still leaves the "
            "collision at 2.44 m/s -- but grinds to a halt after 30 cm instead "
            "of coasting 1.47 m."
        ),
    ),
    EditCase(
        case_id="edit_bouncy_block",
        source_case_id=SOURCE_CASE_ID,
        seed=5106,
        dsl="SET wood_block.restitution FROM 0.55 TO 1",
        edit_summary=(
            "Block's faces made perfectly elastic, so the pair restitution goes "
            "from 0.43 to 0.78. More of the ball's momentum goes into the block "
            "and less stays with the ball: the block is driven 2.11 m instead "
            "of 1.47 m, and the ball is checked almost dead at the impact "
            "(0.18 m/s, against 0.65) and ends up behind where it hit."
        ),
    ),
    EditCase(
        case_id="edit_remove_block",
        source_case_id=SOURCE_CASE_ID,
        seed=5107,
        dsl="DELETE wood_block",
        edit_summary=(
            "Block removed. The ball rolls straight through the spot where it "
            "used to stand and on out of frame, with nothing in the video ever "
            "interrupting it. No collision at all."
        ),
    ),
)


# --------------------------------------------------------------------- CLI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the ball_block PCVE suite (1 source + N edits)."
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=WORKSPACE_DIR / "renders" / "pcve_ball_block_suite",
    )
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--resolution", nargs=2, type=int, default=(1280, 720))
    parser.add_argument("--fps", type=int, default=24)
    # 3 s. The collision lands on frame 14 and the block has always stopped by
    # the end; the ball sometimes has not, which is itself part of what
    # separates several of the cases.
    parser.add_argument("--duration-sec", type=float, default=4.0)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--surface-marks", choices=("none", "subtle", "full"), default="none")
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
        "--motion", MOTION,
        "--block-texture-asset", BLOCK_TEXTURE_ASSET,
        "--surface-marks", str(args.surface_marks),
        # Both jitters off. The source and every edit must differ only by the
        # one property the edit names -- a jittered camera or a jittered mass
        # would put uncontrolled differences in the frame alongside it.
        "--physics-jitter", "0",
        "--camera-jitter", "0",
    ]
    if overrides_path is not None:
        cmd += ["--scenario-overrides-json", str(overrides_path.resolve())]
    return cmd


def preferred_video_path(case_dir: Path) -> Path:
    preferred = (
        case_dir / "ball_block_impact.mp4",
        case_dir / "ball_block_impact_cycles.mp4",
    )
    for path in preferred:
        if path.exists():
            return path
    matches = sorted(case_dir.glob("*.mp4"))
    if not matches:
        raise FileNotFoundError(f"No mp4 found in {case_dir}")
    return matches[0]


def standardize_render_outputs(case_dir: Path, *, has_overrides: bool) -> dict[str, str]:
    video_source = preferred_video_path(case_dir)
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
        "suite_name": "pcve_ball_block_suite",
        "description": (
            "One source ball-versus-block collision video plus N edited "
            "variants. Each edit is declared with a single edit-DSL string; "
            "prompts (precise + vague, zh + en), scenario overrides, and the "
            "physics diff are all derived from that one string."
        ),
        "motion": MOTION,
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
            "Source video: default parameters, no jitter. The ball rolls in at "
            "6 m/s, arrives at the block at 4.38 m/s on frame 14, is checked to "
            "0.65 m/s, and shoves the block 1.47 m across the floor, where it "
            "comes to rest. The ball follows it and is still rolling slowly "
            "when the 3 s clip ends."
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
