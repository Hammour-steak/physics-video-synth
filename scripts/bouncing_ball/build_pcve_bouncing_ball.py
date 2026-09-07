"""Build the bouncing_ball PCVE suite.

One source video (red ball dropped from ~1.32 m with a gentle 0.5 m/s
horizontal nudge, bounces to ~0.86 m) plus N edited variants. Each edit is
declared as a single edit-DSL string; prompts, scenario overrides and the
physics diff are all derived from that one string.
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
import edit_vocab              # noqa: E402


WORKSPACE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_BLENDER = WORKSPACE_DIR / 'tools' / 'blender-3.6.23-linux-x64' / 'blender'
RENDER_SCRIPT = WORKSPACE_DIR / 'scripts' / 'bouncing_ball' / 'render_bouncing_ball.py'


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


SOURCE_CASE_ID = 'bouncing_ball_baseline'


# Values were fixed by sweeping simulate_bouncing_ball.py, not guessed:
#   baseline         max_z_after_first_bounce ~ 0.86 m, final_x ~ +0.29 m
#   restitution 0.3  max_z ~ 0.44 m, dies in ~2 bounces
#   restitution 0.95 max_z ~ 1.25 m, bounces higher than the drop
#   init_vel 0.0     final_x ~  0.00 m, drops in place
#   init_vel 3.0     final_x ~ +4.37 m, flies across the room
# ball_mass and ball_friction produce no visible change on this clip length
# (mass cancels out in restitution/gravity; the ball is airborne most of the
# 3 s so lateral friction never gets to do work) -- deliberately excluded.
EDIT_CASES: tuple[EditCase, ...] = (
    EditCase(
        case_id='edit_dead_ball',
        source_case_id=SOURCE_CASE_ID,
        seed=4101,
        dsl='SET ball.restitution TIMES 0.4',
        edit_summary=(
            "Ball made much less elastic. The bounce chain collapses: two "
            "hops instead of seven, a 2.04 m path over the clip against "
            "the baseline's 5.98 m, and the ball is down to 0.07 m/s at "
            "the end where the baseline is still running at 0.87. First "
            "floor contact is unchanged at frame 13."
        ),
    ),
    EditCase(
        case_id='edit_super_bouncy',
        source_case_id=SOURCE_CASE_ID,
        seed=4102,
        dsl='SET ball.restitution TIMES 1.2',
        edit_summary=(
            "Ball made near-elastic. The bounce chain barely decays: the "
            "ball covers a 9.00 m path against the baseline's 5.98 m, and "
            "at the end of the clip it is still airborne at 1.25 m -- "
            "most of the way back to its 1.57 m release height -- and "
            "still moving at 1.09 m/s."
        ),
    ),
    EditCase(
        case_id='edit_no_push',
        source_case_id=SOURCE_CASE_ID,
        seed=4103,
        dsl='SET ball.initial_velocity TIMES 0',
        edit_summary=(
            'Horizontal push removed. The ball drops straight down and bounces '
            'in place at x ~ 0 instead of drifting the ~0.29 m across the floor '
            'seen in the baseline.'
        ),
    ),
    EditCase(
        case_id='edit_strong_push',
        source_case_id=SOURCE_CASE_ID,
        seed=4104,
        dsl='SET ball.initial_velocity TIMES 6',
        edit_summary=(
            'Horizontal push made 6x stronger. The ball crosses the room and '
            'ends the clip at x ~ +4.4 m instead of the baseline +0.29 m, '
            'bouncing along the way.'
        ),
    ),
    # The one edit in this suite that does not hold for the whole clip. It
    # lands while the ball is in the air after its first bounce, so the first
    # bounce is the source video's and every bounce after it is the edit's --
    # which is something no whole-clip edit can show.
    EditCase(
        case_id='edit_dead_ball_after_first_bounce',
        source_case_id=SOURCE_CASE_ID,
        seed=4105,
        dsl='SET ball.restitution TIMES 0.1 AT FRAME 20',
        edit_summary=(
            "Ball made much less elastic at frame 20, while it is airborne "
            "between its first and second bounce. Frames 1-19 are the source "
            "video frame for frame: the ball falls, hits the floor at frame "
            "13 and rebounds to the same 0.86 m apex at frame 23. The second "
            "landing is where the edit shows -- it barely leaves the floor "
            "(0.06 m at frame 34) and the ball is at rest for the rest of the "
            "clip, against the baseline's six apexes. The whole-clip version "
            "of the same edit kills the first bounce too: the ball never gets "
            "above 0.06 m at all."
        ),
    ),
)


# --------------------------------------------------------------------- CLI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build the bouncing_ball PCVE suite (1 source + N edits).'
    )
    parser.add_argument(
        '--out-root',
        type=Path,
        default=WORKSPACE_DIR / 'renders' / 'pcve_bouncing_ball_suite',
    )
    parser.add_argument('--blender', type=Path, default=DEFAULT_BLENDER)
    parser.add_argument('--resolution', nargs=2, type=int, default=(1280, 720))
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('--duration-sec', type=float, default=4.0)
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--device', choices=('auto', 'cpu'), default='auto')
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--verbose-render', action='store_true')
    parser.add_argument(
        '--clean-stale-cases',
        action='store_true',
        help='Delete case directories that are no longer part of this suite.',
    )
    return parser.parse_args()


# --------------------------------------------------------------- render glue


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def render_command(
    args: argparse.Namespace,
    seed: int,
    *,
    case_dir: Path,
    overrides_path: Path | None,
) -> list[str]:
    cmd = [
        str(args.blender.expanduser().resolve()),
        '-b',
        '--python',
        str(RENDER_SCRIPT.resolve()),
        '--',
        '--mode', 'animation',
        '--out-dir', str(case_dir.resolve()),
        '--resolution', str(int(args.resolution[0])), str(int(args.resolution[1])),
        '--fps', str(int(args.fps)),
        '--duration-sec', str(float(args.duration_sec)),
        '--samples', str(int(args.samples)),
        '--device', str(args.device),
        '--seed', str(int(seed)),
    ]
    if overrides_path is not None:
        cmd += ['--scenario-overrides-json', str(overrides_path.resolve())]
    return cmd


def standardize_render_outputs(case_dir: Path, *, has_overrides: bool) -> dict[str, str]:
    video_source = case_dir / 'bouncing_ball.mp4'
    if not video_source.exists():
        candidates = sorted(case_dir.glob('*.mp4'))
        if not candidates:
            raise FileNotFoundError(f'No mp4 found in {case_dir}')
        video_source = candidates[0]
    video_target = case_dir / 'video.mp4'
    if video_source.resolve() != video_target.resolve():
        shutil.copy2(video_source, video_target)

    outputs: dict[str, Path] = {
        'video': video_target,
        'ground_truth': case_dir / 'ground_truth_transforms.json',
        'scenario_metadata': case_dir / 'scenario_metadata.json',
    }
    if has_overrides:
        outputs['scenario_overrides'] = case_dir / 'scenario_overrides.json'
    for key, path in outputs.items():
        if not path.exists():
            raise FileNotFoundError(f'Missing rendered {key}: {path}')
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
        print(' '.join(cmd))
        return
    if args.verbose_render:
        subprocess.run(cmd, check=True)
        return
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        tail = '\n'.join((result.stderr or '').splitlines()[-40:])
        print(f'[suite] render failed; stderr tail:\n{tail}')
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)


def build_edit_record(case: EditCase) -> dict[str, Any]:
    parsed = dsl.parse(case.dsl, VOCAB)
    # The scenario override, not the raw parameter dict: an edit that lands
    # partway through ships a schedule the simulator applies at its frame,
    # leaving the frames before it on the source video's own physics.
    physics = dsl.to_scenario_override(parsed, VOCAB)
    if isinstance(parsed, dsl.SetEdit):
        diff = {f'{parsed.property_name} ({parsed.object_id})':
                {'from': dsl.baseline_value_for(parsed, VOCAB), 'to': parsed.to_value}}
    else:
        diff = {parsed.object_id: {'from': 'present', 'to': 'removed'}}
    diff['timing'] = dsl.timing_diff(parsed, VOCAB)
    return {
        'edit_dsl': case.dsl,
        'edit_summary': case.edit_summary,
        'applies_from_frame': dsl.starts_at_frame(parsed),
        'prompts': dsl.make_prompts(parsed, VOCAB),
        'physics_diff': diff,
        'physics_override': physics,
    }


def write_prompt_file(case_dir: Path, case: EditCase, edit_info: dict[str, Any]) -> Path:
    path = case_dir / 'prompts.json'
    write_json(path, {
        'schema_version': 2,
        'case_id': case.case_id,
        'source_case_id': case.source_case_id,
        'edit_dsl': edit_info['edit_dsl'],
        'edit_summary': edit_info['edit_summary'],
        'applies_from_frame': edit_info['applies_from_frame'],
        'physics_diff': edit_info['physics_diff'],
        'prompts': edit_info['prompts'],
    })
    return path


def clean_stale(out_root: Path, keep_ids: set[str]) -> None:
    cases_dir = out_root / 'cases'
    if not cases_dir.exists():
        return
    for path in sorted(cases_dir.iterdir()):
        if not path.is_dir() or path.name in keep_ids:
            continue
        print(f'[suite] remove stale case directory {path}')
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

    manifest_path = args.out_root / 'suite_manifest.json'
    manifest: dict[str, Any] = {
        'schema_version': 3,
        'suite_name': 'pcve_bouncing_ball_suite',
        'description': (
            'One source bouncing-ball video plus N edited variants. The red '
            'ball is the only moving object; each edit is one line of edit-DSL '
            'against it, and prompts (precise + vague, zh + en), scenario '
            'overrides and the physics diff are all derived from that one '
            'string.'
        ),
        'baseline_physics': BASELINE_PHYSICS,
        'total_frames': edit_vocab.TOTAL_FRAMES,
        'resolution': [int(args.resolution[0]), int(args.resolution[1])],
        'fps': int(args.fps),
        'duration_sec': float(args.duration_sec),
        'samples': int(args.samples),
        'source': None,
        'edits': [],
    }
    write_json(manifest_path, manifest)

    # ------------------------------------------------------------- source
    source_dir = args.out_root / 'cases' / SOURCE_CASE_ID
    source_record: dict[str, Any] = {
        'case_id': SOURCE_CASE_ID,
        'kind': 'source',
        "description": {
            "vague": {
                "en": (
                    "The red ball is dropped with a gentle sideways push, "
                    "meets the floor and bounces its way across the room. The "
                    "bounces get smaller and it is still moving when the clip "
                    "ends."
                ),
                "zh": (
                    "红球带着轻微的水平初速落下,触地后一路弹跳着横穿房间。弹跳逐"
                    "次变小,片尾时仍在运动。"
                ),
            },
            "quantitative": {
                "en": (
                    "The red ball is released from 1.57 m with a gentle 0.5 "
                    "m/s horizontal push, first meets the floor on frame 13, "
                    "and bounces its way across the room. The chain decays "
                    "over the clip and the ball is still moving when it ends."
                ),
                "zh": (
                    "红球从 1.57 m 高处释放,带 0.5 m/s 的水平初"
                    "速,第 13 帧首次触地,之后一路弹跳着横穿房间。弹跳逐次衰"
                    "减,片尾时球仍在运动。"
                ),
            },
        },
        'case_dir': str(source_dir.resolve()),
        'status': 'pending',
    }
    manifest['source'] = source_record
    write_json(manifest_path, manifest)

    expected_source_video = source_dir / 'video.mp4'
    if args.skip_existing and expected_source_video.exists():
        source_record['status'] = 'skipped_existing'
        source_record['outputs'] = standardize_render_outputs(source_dir, has_overrides=False)
    elif args.dry_run:
        render_case(args, case_dir=source_dir, seed=4001, overrides_path=None)
        source_record['status'] = 'dry_run'
    else:
        t0 = time.perf_counter()
        print(f'[suite] render source {SOURCE_CASE_ID}')
        render_case(args, case_dir=source_dir, seed=4001, overrides_path=None)
        source_record['outputs'] = standardize_render_outputs(source_dir, has_overrides=False)
        source_record['elapsed_sec'] = round(time.perf_counter() - t0, 3)
        source_record['status'] = 'completed'
    write_json(manifest_path, manifest)

    # -------------------------------------------------------------- edits
    for case in EDIT_CASES:
        case_dir = args.out_root / 'cases' / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        edit_info = build_edit_record(case)

        overrides_payload = {'physics': edit_info['physics_override']}
        overrides_path = case_dir / 'scenario_overrides.json'
        write_json(overrides_path, overrides_payload)
        prompts_path = write_prompt_file(case_dir, case, edit_info)

        record: dict[str, Any] = {
            'case_id': case.case_id,
            'kind': 'edit',
            'source_case_id': case.source_case_id,
            'seed': case.seed,
            'case_dir': str(case_dir.resolve()),
            'scenario_overrides_json': str(overrides_path.resolve()),
            'prompts_json': str(prompts_path.resolve()),
            'edit_dsl': edit_info['edit_dsl'],
            'edit_summary': edit_info['edit_summary'],
            'applies_from_frame': edit_info['applies_from_frame'],
            'physics_diff': edit_info['physics_diff'],
            'prompts': edit_info['prompts'],
            'status': 'pending',
        }
        manifest['edits'].append(record)
        write_json(manifest_path, manifest)

        expected_video = case_dir / 'video.mp4'
        if args.skip_existing and expected_video.exists():
            record['status'] = 'skipped_existing'
            record['outputs'] = standardize_render_outputs(case_dir, has_overrides=True)
            write_json(manifest_path, manifest)
            print(f'[suite] skip existing {case.case_id}')
            continue

        if args.dry_run:
            render_case(args, case_dir=case_dir, seed=case.seed, overrides_path=overrides_path)
            record['status'] = 'dry_run'
            write_json(manifest_path, manifest)
            continue

        t0 = time.perf_counter()
        print(f'[suite] render edit {case.case_id}')
        try:
            render_case(args, case_dir=case_dir, seed=case.seed, overrides_path=overrides_path)
        except subprocess.CalledProcessError:
            record['status'] = 'failed'
            record['elapsed_sec'] = round(time.perf_counter() - t0, 3)
            write_json(manifest_path, manifest)
            raise
        record['outputs'] = standardize_render_outputs(case_dir, has_overrides=True)
        record['elapsed_sec'] = round(time.perf_counter() - t0, 3)
        record['status'] = 'completed'
        write_json(manifest_path, manifest)
        print(f'[suite] completed {case.case_id} in {record["elapsed_sec"]:.1f}s')

    print(f'[suite] manifest={manifest_path.resolve()}')


if __name__ == '__main__':
    main()
