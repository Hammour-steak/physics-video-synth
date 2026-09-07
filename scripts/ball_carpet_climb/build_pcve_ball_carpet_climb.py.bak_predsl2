"""Build the ball_carpet_climb PCVE suite.

One source video (volleyball rolled at 2.05 m/s across the bare floor onto a
flush rug, comes to rest ~0.49 m onto the rug) plus N edited variants. Each
edit is declared as a single edit-DSL string; prompts (precise + vague, zh +
en), scenario overrides and the physics diff for the manifest are all derived
from that one string.
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
RENDER_SCRIPT = WORKSPACE_DIR / 'scripts' / 'ball_carpet_climb' / 'render_ball_carpet_climb.py'


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


SOURCE_CASE_ID = 'ball_carpet_climb_baseline'


# Values were fixed by sweeping simulate_ball_carpet_climb.py, not guessed:
#   baseline           carpet_travel=0.49 m, stops mid-rug
#   ball.friction 3.0  carpet_travel=0.16 m, stalls at the rug's near edge
#   ball.friction 0.3  carpet_travel=1.11 m, coasts the full rug width
#   launch_speed 1.30  carpet_travel=0.19 m, dies just onto the rug
#   launch_speed 3.00  carpet_travel=1.06 m, coasts the full rug to the sofa
EDIT_CASES: tuple[EditCase, ...] = (
    EditCase(
        case_id='edit_grippy_ball',
        source_case_id=SOURCE_CASE_ID,
        seed=17101,
        dsl='SET ball.friction FROM 1.0 TO 3.0',
        edit_summary=(
            'Ball made much grippier (friction 3x). It rolls the same bare floor '
            'in, but the moment it touches the rug the extra rolling resistance '
            'kills it: instead of the 0.49 m mid-rug rest it barely creeps 0.16 m '
            'past the rug edge before stopping.'
        ),
    ),
    EditCase(
        case_id='edit_slippy_ball',
        source_case_id=SOURCE_CASE_ID,
        seed=17102,
        dsl='SET ball.friction FROM 1.0 TO 0.3',
        edit_summary=(
            'Ball made much slicker (friction 0.3x). The rug barely slows it: it '
            'coasts across the full 1.11 m of pile and comes to rest against the '
            'front of the sofa, rather than stopping halfway across the rug.'
        ),
    ),
    EditCase(
        case_id='edit_soft_push',
        source_case_id=SOURCE_CASE_ID,
        seed=17103,
        dsl='SET ball.initial_velocity FROM 2.05 TO 1.30',
        edit_summary=(
            'Gentler push. The ball reaches the rug at 1.25 m/s instead of the '
            "baseline's 2.0 m/s and dies almost immediately after the border, "
            'stopping ~0.19 m in -- a rug-edge stall driven by the weaker start, '
            'not by a grippier surface.'
        ),
    ),
    EditCase(
        case_id='edit_hard_push',
        source_case_id=SOURCE_CASE_ID,
        seed=17104,
        dsl='SET ball.initial_velocity FROM 2.05 TO 3.00',
        edit_summary=(
            'Firmer push. The ball hits the rug at ~2.9 m/s and carries all the '
            'way across the pile, stopping against the front of the sofa instead '
            'of half a metre in.'
        ),
    ),
)


# --------------------------------------------------------------------- CLI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build the ball_carpet_climb PCVE suite (1 source + N edits).'
    )
    parser.add_argument(
        '--out-root',
        type=Path,
        default=WORKSPACE_DIR / 'renders' / 'pcve_ball_carpet_climb_suite',
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
    video_source = case_dir / 'ball_carpet_climb.mp4'
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
    physics = dsl.to_physics_override(parsed, VOCAB)
    if isinstance(parsed, dsl.SetEdit):
        diff = {f'{parsed.property_name} ({parsed.object_id})':
                {'from': dsl.baseline_value_for(parsed, VOCAB), 'to': parsed.to_value}}
    else:
        diff = {parsed.object_id: {'from': 'present', 'to': 'removed'}}
    return {
        'edit_dsl': case.dsl,
        'edit_summary': case.edit_summary,
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
    args.out_root.mkdir(parents=True, exist_ok=True)

    keep_ids = {SOURCE_CASE_ID, *(c.case_id for c in EDIT_CASES)}
    if args.clean_stale_cases and not args.dry_run:
        clean_stale(args.out_root, keep_ids)

    manifest_path = args.out_root / 'suite_manifest.json'
    manifest: dict[str, Any] = {
        'schema_version': 3,
        'suite_name': 'pcve_ball_carpet_climb_suite',
        'description': (
            'One source ball-onto-rug video plus N edited variants. The ball is '
            'the only moving object; each edit is one line of edit-DSL against '
            'it, and prompts (precise + vague, zh + en), scenario overrides and '
            'the physics diff are all derived from that one string.'
        ),
        'baseline_physics': BASELINE_PHYSICS,
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
        'description': (
            'Source video: baseline parameters. The volleyball is rolled at '
            '2.05 m/s across ~0.5 m of bare floor, reaches the flush rug at '
            '~2.0 m/s and comes to rest ~0.49 m onto the rug, next to the '
            'coffee table.'
        ),
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
        render_case(args, case_dir=source_dir, seed=17001, overrides_path=None)
        source_record['status'] = 'dry_run'
    else:
        t0 = time.perf_counter()
        print(f'[suite] render source {SOURCE_CASE_ID}')
        render_case(args, case_dir=source_dir, seed=17001, overrides_path=None)
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
