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
        dsl='SET car.initial_velocity TIMES 0.75',
        edit_summary=(
            "Push weakened below the clearance threshold. The car noses "
            "off the book stack but its arc falls short of the far table "
            "and it drops into the gap, landing on the room floor at "
            "x=+0.01 against the baseline's upright landing on the far "
            "table at x=+0.87."
        ),
    ),
    EditCase(
        case_id='edit_overpowered',
        source_case_id=SOURCE_CASE_ID,
        seed=33,
        dsl='SET car.initial_velocity TIMES 2',
        edit_summary=(
            "Push doubled. The car clears the gap easily but has so much "
            "speed left that it skids straight off the far table: it ends "
            "the clip on the room floor at x=+3.22, 3.53 m from where it "
            "started, against the baseline's tidy landing on the far "
            "table at x=+0.87."
        ),
    ),
    EditCase(
        case_id='edit_slippery_wheels',
        source_case_id=SOURCE_CASE_ID,
        seed=34,
        dsl='SET car.friction TIMES 0.3',
        edit_summary=(
            "Car friction cut to less than a third. It scrubs almost "
            "nothing off on the launch deck and reaches the edge far "
            "faster than the source, so it clears the gap and keeps "
            "going: it slides straight off the far table and ends the "
            "clip upright on the room floor at x=+2.78, 3.11 m from where "
            "it started, against the baseline's tidy landing on the far "
            "table at x=+0.87."
        ),
    ),
    EditCase(
        case_id='edit_grippy_wheels',
        source_case_id=SOURCE_CASE_ID,
        seed=35,
        dsl='SET car.friction TIMES 7',
        edit_summary=(
            "Car friction cranked up sevenfold. The deck scrubs so much "
            "speed off the roll-up that the car cannot clear the gap: it "
            "drops off the near edge and lands upside down on the room "
            "floor 0.89 m below, at x=+0.22, where the baseline lands "
            "upright on the far table at x=+0.87."
        ),
    ),
    # The one edit in this suite that does not hold for the whole clip, and
    # deliberately the same factor as edit_grippy_wheels: they differ by the
    # AT FRAME clause alone. Landing it after take-off is what makes the
    # timing readable -- the whole-clip version never lets the car leave the
    # near table at all.
    EditCase(
        case_id='edit_grippy_wheels_mid_jump',
        source_case_id=SOURCE_CASE_ID,
        seed=14105,
        dsl='SET car.friction TIMES 7 AT FRAME 6',
        edit_summary=(
            "Wheel friction raised sevenfold at frame 6, with the car already "
            "off the near table and in the air. Frames 1-5 are the source "
            "video frame for frame, take-off included, and the flight is "
            "unchanged -- friction does nothing until something is touched. "
            "The landing is where it shows: instead of sliding out onto the "
            "far table and coasting to x=+0.78, the tyres grip the far lip, "
            "the car trips over it and drops into the gap, ending upside down "
            "at z=-0.71. The whole-clip version of the same factor is a "
            "different video again: the car never gets moving and falls "
            "straight into the gap from the near edge."
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


def physics_override_to_cli(override: dict[str, Any], case_dir: Path) -> list[str]:
    """Turn a vocab-produced physics_override dict into render CLI flags.

    An edit that lands partway through the clip is not a parameter value but a
    schedule, so it is written into the case directory and passed by path --
    the same file the other scenes carry inside their scenario_overrides.json.
    """
    flags: list[str] = []
    for key, value in override.items():
        if key == dsl.TIMED_EDITS_KEY:
            schedule_path = case_dir / 'timed_edits.json'
            write_json(schedule_path, value)
            flags += ['--timed-edits-json', str(schedule_path.resolve())]
            continue
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
        'suite_name': 'pcve_car_gap_jump_suite',
        'description': (
            'One source toy-car gap-jump video plus N edited variants. The '
            'car is the only moving object; each edit is one line of edit-DSL '
            'against it, and prompts (precise + vague, zh + en), the physics '
            'diff and the render CLI flags are all derived from that one '
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
                    "The toy car is pushed across a book stack, clears the "
                    "gap beyond it, and lands upright on the far table where "
                    "it comes to rest."
                ),
                "zh": (
                    "玩具车被推过书堆,越过前方的缺口,正面朝上落在对面桌子上并停"
                    "下。"
                ),
            },
            "quantitative": {
                "en": (
                    "The toy car is pushed across the book stack, clears the "
                    "gap, and lands upright on the far table, coming to rest "
                    "1.07 m from where it set off, at x=+0.87."
                ),
                "zh": (
                    "玩具车被推过书堆,越过缺口,正面朝上落在对面桌子上,相对起点"
                    "移动 1.07 m,停在 x=+0.87。"
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

        physics_cli = physics_override_to_cli(edit_info['physics_override'], case_dir)

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
