"""Build the curling_collision PCVE suite.

One source video (red and yellow 20 kg stones launched at each other at
0.9 m/s from 5 m apart, meet head-on and both come to rest at the point of
impact) plus N edited variants. Each edit is one line of edit-DSL; prompts,
scenario overrides and the physics diff are all derived from that one string.
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
RENDER_SCRIPT = WORKSPACE_DIR / 'scripts' / 'curling_collision' / 'render_curling_collision.py'


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


SOURCE_CASE_ID = 'curling_collision_baseline'


# Sweep numbers (simulate_curling_collision.py, baseline v=0.9, m1=m2=20):
#   baseline           collided=True,  s1_final=-0.25 s2_final=+0.25 (dead-stop at centre)
#   stones.v 0.9->0.4  collided=False, stones stop 0.3 m apart, no impact
#   stones.rest 0->0.95 collided=True, stones rebound to +/-0.57 (baseline +/-0.25)
#   yellow.mass 20->60 collided=True, red rebounds back to -0.83, yellow -0.31
#   yellow.mass 20->4  collided=True, yellow shoved to +1.20, red continues to +0.68
# ice_friction and start_separation would also flip outcomes but they are
# scene/set edits (see edit_vocab.py); stone_friction is a visible-null on
# this clip length.
EDIT_CASES: tuple[EditCase, ...] = (
    # --- symmetric edits ------------------------------------------------
    EditCase(
        case_id='edit_gentle_launch',
        source_case_id=SOURCE_CASE_ID,
        seed=9101,
        dsl='SET stones.initial_velocity FROM 1.5 TO 0.5',
        edit_summary=(
            'Both stones launched at one-third of the source speed. Ice '
            'friction bleeds them off before they meet: they stop about '
            '2.2 m apart at +/-1.12 m, never colliding at all, instead of '
            'meeting head-on around frame 39.'
        ),
    ),
    # --- per-stone mass edits ------------------------------------------
    EditCase(
        case_id='edit_heavy_yellow',
        source_case_id=SOURCE_CASE_ID,
        seed=9102,
        dsl='SET yellow_stone.mass FROM 20.0 TO 60.0',
        edit_summary=(
            'Yellow stone made 3x heavier. The head-on impact is no longer '
            'balanced: yellow only gives ground to -0.78 m, while red rebounds '
            'clear past its own side to -1.30 m -- the opposite of the source, '
            'where both stones die at centre.'
        ),
    ),
    EditCase(
        case_id='edit_light_yellow',
        source_case_id=SOURCE_CASE_ID,
        seed=9103,
        dsl='SET yellow_stone.mass FROM 20.0 TO 4.0',
        edit_summary=(
            'Yellow stone made 5x lighter. Momentum from the red stone shoves '
            'it clear across the ice to +1.76 m, while the red stone barely '
            'slows and keeps going to +1.11 m -- the mirror of '
            'edit_heavy_yellow, with the light stone doing the flying.'
        ),
    ),
    # --- per-stone velocity edits (asymmetric launch) -------------------
    EditCase(
        case_id='edit_red_hard_throw',
        source_case_id=SOURCE_CASE_ID,
        seed=9104,
        dsl='SET red_stone.initial_velocity FROM 1.5 TO 3.0',
        edit_summary=(
            'Red stone launched twice as fast; yellow still 1.5 m/s. The '
            'head-on impact is one-sided and happens earlier (frame 26): red '
            'carries far more momentum, and both stones end up well past '
            'centre on the yellow side (red +1.90, yellow +2.46) -- the '
            "source's centred stop is broken."
        ),
    ),
    EditCase(
        case_id='edit_red_soft_throw',
        source_case_id=SOURCE_CASE_ID,
        seed=9105,
        dsl='SET red_stone.initial_velocity FROM 1.5 TO 0.7',
        edit_summary=(
            'Red stone launched at less than half the yellow stone speed. '
            'Yellow now carries the momentum: contact is delayed to frame 53 '
            'and both stones end up on the red side of centre (red -1.43, '
            'yellow -0.93) -- the DSL-opposite of edit_red_hard_throw and a '
            'distinct visual (the yellow stone is what crosses centre).'
        ),
    ),
    EditCase(
        case_id='edit_yellow_at_rest',
        source_case_id=SOURCE_CASE_ID,
        seed=9106,
        dsl='SET yellow_stone.initial_velocity FROM 1.5 TO 0.0',
        edit_summary=(
            'Yellow stone starts at rest instead of sliding in. Red stone '
            'launched at the same 1.5 m/s from 5 m away rolls in alone -- but '
            'ice friction bleeds it off before it reaches yellow, so it stops '
            'at +1.84 m with 0.66 m still between them. No collision happens; '
            'yellow never moves.'
        ),
    ),
    # --- pair-shared elasticity ----------------------------------------
    EditCase(
        case_id='edit_bouncy_stones',
        source_case_id=SOURCE_CASE_ID,
        seed=9107,
        dsl='SET stones.restitution FROM 0.0 TO 0.95',
        edit_summary=(
            'Pair restitution raised from perfectly inelastic to near-elastic. '
            'The symmetric collision now rebounds cleanly: stones bounce apart '
            'and end the clip at +/-0.95 m (baseline +/-0.32) still drifting '
            'away from centre.'
        ),
    ),
)


# --------------------------------------------------------------------- CLI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build the curling_collision PCVE suite (1 source + N edits).'
    )
    parser.add_argument(
        '--out-root',
        type=Path,
        default=WORKSPACE_DIR / 'renders' / 'pcve_curling_collision_suite',
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
    video_source = case_dir / 'curling_collision.mp4'
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
        'suite_name': 'pcve_curling_collision_suite',
        'description': (
            'One source curling head-on collision video plus N edited '
            'variants. Both stones move (they are launched symmetrically at '
            'each other); each edit is one line of edit-DSL against a stone '
            'or the collective stones for pair-shared knobs, and prompts, '
            'overrides, and the physics diff are derived from that string.'
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
            'Source video: baseline parameters. Two equal 20 kg curling '
            'stones launched at each other at 0.9 m/s from 5 m apart, meet '
            'head-on at the centre of the sheet and both come to rest '
            'touching (perfectly inelastic collision, restitution 0.0).'
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
        render_case(args, case_dir=source_dir, seed=9001, overrides_path=None)
        source_record['status'] = 'dry_run'
    else:
        t0 = time.perf_counter()
        print(f'[suite] render source {SOURCE_CASE_ID}')
        render_case(args, case_dir=source_dir, seed=9001, overrides_path=None)
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
