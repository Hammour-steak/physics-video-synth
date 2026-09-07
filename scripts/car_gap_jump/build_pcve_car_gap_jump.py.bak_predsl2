"""Build the car_gap_jump PCVE suite.

One source video (toy car pushed at 1.9 m/s across the book stack, clears the
0.28 m gap and lands upright on the far table, coming to rest ~0.78 m past
its edge) plus N edited variants. Each edit is one line of edit-DSL against
the car; prompts, scenario overrides and the physics diff are all derived
from that one string.

The render script does not consume a scenario_overrides.json (it only takes
--launch-speed / --gap-width / --car-friction on its command line), so this
suite translates each edit's physics_override dict into the matching render
CLI flags rather than writing a JSON overrides file.
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
RENDER_SCRIPT = WORKSPACE_DIR / 'scripts' / 'car_gap_jump' / 'render_car_gap_jump.py'


VOCAB = edit_vocab.VOCAB
BASELINE_PHYSICS = edit_vocab.BASELINE_PHYSICS


# Map physics-key -> render-CLI flag. Only these keys can flow through the
# render script; anything else in a physics_override is a bug in the vocab.
SIM_KEY_TO_CLI_FLAG = {
    'launch_speed': '--launch-speed',
    'car_friction': '--car-friction',
    'gap_width':    '--gap-width',
}


# --------------------------------------------------------------- edit cases


@dataclass(frozen=True)
class EditCase:
    case_id: str
    source_case_id: str
    seed: int
    dsl: str
    edit_summary: str


SOURCE_CASE_ID = 'car_gap_jump_baseline'


# All values were fixed by sweeping simulate_car_gap_jump.py:
#   baseline (1.9 m/s, friction 0.45)  clears=1, final_x=+0.78
#   launch_speed 1.4                   clears=0, fell_into_chasm=1
#   launch_speed 3.5                   clears=0, overshoots, final_x=+2.81, drops off far end
#   car_friction 0.02                  clears=0, skates off far end (final_x=+2.61)
#   car_friction 3.0                   clears=0, drags to stop and falls into chasm
EDIT_CASES: tuple[EditCase, ...] = (
    EditCase(
        case_id='edit_underpowered',
        source_case_id=SOURCE_CASE_ID,
        seed=32,
        dsl='SET car.initial_velocity FROM 1.9 TO 1.4',
        edit_summary=(
            'Push weakened below the ~1.55 m/s clearance threshold. The car '
            'noses off the book stack but its arc falls short of the far '
            'table and it drops into the gap onto the room floor 0.71 m below '
            "(baseline lands upright on the far table at final_x=+0.78)."
        ),
    ),
    EditCase(
        case_id='edit_overpowered',
        source_case_id=SOURCE_CASE_ID,
        seed=33,
        dsl='SET car.initial_velocity FROM 1.9 TO 3.5',
        edit_summary=(
            'Push nearly doubled. The car clears the gap easily but has so '
            'much speed left that it skids straight off the far table and '
            'ends the clip on the floor beyond it (final_x=+2.81, final_z '
            "drops below the tabletop)."
        ),
    ),
    EditCase(
        case_id='edit_slippery_wheels',
        source_case_id=SOURCE_CASE_ID,
        seed=34,
        dsl='SET car.friction FROM 0.45 TO 0.02',
        edit_summary=(
            'Car body/wheel friction dropped to near-zero. On the launch '
            'deck it coasts almost losslessly, so it reaches the edge with '
            'far more speed than the source and, after clearing, keeps '
            'sliding off the far table -- final_x=+2.61, ends the clip on '
            'the floor beyond it.'
        ),
    ),
    EditCase(
        case_id='edit_grippy_wheels',
        source_case_id=SOURCE_CASE_ID,
        seed=35,
        dsl='SET car.friction FROM 0.45 TO 3.0',
        edit_summary=(
            'Car friction cranked up 6.7x. The deck now scrubs so much speed '
            'off the roll-up that the car barely leaves the stack -- it '
            'falls off the near edge into the gap instead of arcing across '
            '(final_x=+0.17, drops 0.71 m).'
        ),
    ),
)


# --------------------------------------------------------------------- CLI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build the car_gap_jump PCVE suite (1 source + N edits).'
    )
    parser.add_argument(
        '--out-root',
        type=Path,
        default=WORKSPACE_DIR / 'renders' / 'pcve_car_gap_jump_suite',
    )
    parser.add_argument('--blender', type=Path, default=DEFAULT_BLENDER)
    parser.add_argument('--resolution', nargs=2, type=int, default=(1280, 720))
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('--duration-sec', type=float, default=4.0)
    parser.add_argument('--samples', type=int, default=128)
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


def physics_override_to_cli(override: dict[str, Any]) -> list[str]:
    """Turn a vocab-produced physics_override dict into render CLI flags."""
    flags: list[str] = []
    for key, value in override.items():
        if key not in SIM_KEY_TO_CLI_FLAG:
            raise ValueError(
                f'physics_override key {key!r} has no render CLI flag; '
                'either extend SIM_KEY_TO_CLI_FLAG or drop the binding.'
            )
        flags += [SIM_KEY_TO_CLI_FLAG[key], str(float(value))]
    return flags


def render_command(
    args: argparse.Namespace,
    seed: int,
    *,
    case_dir: Path,
    physics_override_cli: list[str],
) -> list[str]:
    return [
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
        *physics_override_cli,
    ]


def standardize_render_outputs(case_dir: Path) -> dict[str, str]:
    video_source = case_dir / 'car_gap_jump.mp4'
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
    return {key: str(path.resolve()) for key, path in outputs.items() if path.exists()}


def render_case(
    args: argparse.Namespace,
    *,
    case_dir: Path,
    seed: int,
    physics_override_cli: list[str],
) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    cmd = render_command(args, seed, case_dir=case_dir, physics_override_cli=physics_override_cli)
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
        'suite_name': 'pcve_car_gap_jump_suite',
        'description': (
            'One source toy-car gap-jump video plus N edited variants. The '
            'car is the only moving object; each edit is one line of edit-DSL '
            'against it, and prompts (precise + vague, zh + en), the physics '
            'diff and the render CLI flags are all derived from that one '
            'string.'
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
            'Source video: baseline parameters. The toy car is pushed at '
            '1.9 m/s across the book stack, clears the 0.28 m gap and lands '
            'upright on the far table, coming to rest at final_x ~ +0.78 m.'
        ),
        'case_dir': str(source_dir.resolve()),
        'status': 'pending',
    }
    manifest['source'] = source_record
    write_json(manifest_path, manifest)

    expected_source_video = source_dir / 'video.mp4'
    if args.skip_existing and expected_source_video.exists():
        source_record['status'] = 'skipped_existing'
        source_record['outputs'] = standardize_render_outputs(source_dir)
    elif args.dry_run:
        render_case(args, case_dir=source_dir, seed=31, physics_override_cli=[])
        source_record['status'] = 'dry_run'
    else:
        t0 = time.perf_counter()
        print(f'[suite] render source {SOURCE_CASE_ID}')
        render_case(args, case_dir=source_dir, seed=31, physics_override_cli=[])
        source_record['outputs'] = standardize_render_outputs(source_dir)
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

        physics_cli = physics_override_to_cli(edit_info['physics_override'])

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
            'render_cli_flags': physics_cli,
            'status': 'pending',
        }
        manifest['edits'].append(record)
        write_json(manifest_path, manifest)

        expected_video = case_dir / 'video.mp4'
        if args.skip_existing and expected_video.exists():
            record['status'] = 'skipped_existing'
            record['outputs'] = standardize_render_outputs(case_dir)
            write_json(manifest_path, manifest)
            print(f'[suite] skip existing {case.case_id}')
            continue

        if args.dry_run:
            render_case(args, case_dir=case_dir, seed=case.seed, physics_override_cli=physics_cli)
            record['status'] = 'dry_run'
            write_json(manifest_path, manifest)
            continue

        t0 = time.perf_counter()
        print(f'[suite] render edit {case.case_id}')
        try:
            render_case(args, case_dir=case_dir, seed=case.seed, physics_override_cli=physics_cli)
        except subprocess.CalledProcessError:
            record['status'] = 'failed'
            record['elapsed_sec'] = round(time.perf_counter() - t0, 3)
            write_json(manifest_path, manifest)
            raise
        record['outputs'] = standardize_render_outputs(case_dir)
        record['elapsed_sec'] = round(time.perf_counter() - t0, 3)
        record['status'] = 'completed'
        write_json(manifest_path, manifest)
        print(f'[suite] completed {case.case_id} in {record["elapsed_sec"]:.1f}s')

    print(f'[suite] manifest={manifest_path.resolve()}')


if __name__ == '__main__':
    main()
