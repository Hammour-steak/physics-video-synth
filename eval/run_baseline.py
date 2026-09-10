"""Run a baseline over the whole (or filtered) benchmark.

Reads benchmark_manifest.json, iterates over `edits`, calls the chosen
baseline on each (source_video, prompt) pair, writes:

    predictions/{baseline_name}/videos/{scene}/{case_id}.mp4
    predictions/{baseline_name}/run_manifest.json

Idempotent: pass --skip-existing to skip cases whose output mp4 already
exists (useful after crashes / partial runs).

Usage:
    python run_baseline.py --baseline stub
    python run_baseline.py --baseline wan_vace_14b --limit 5
    python run_baseline.py --baseline wan_vace_14b \\
        --benchmark-root /path/to/pcve_benchmark_v1 \\
        --predictions-root /path/to/predictions \\
        --prompt-flavor quantitative --prompt-lang en \\
        --filter-property mass --skip-existing
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
import traceback
from pathlib import Path
from typing import Any

from baselines import BaselineModel
from baselines.stub import StubBaseline


# Register baselines here. Each entry maps `name` -> (module path, class).
# Modules are imported lazily so `--baseline stub` never imports diffusers.
BASELINES: dict[str, tuple[str, str]] = {
    "stub":         ("baselines.stub",     "StubBaseline"),
    "wan_vace_14b": ("baselines.wan_vace", "WanVACEBaseline"),
    # DELETE edits only -- VOID is an object-removal inpainter, so SET and ADD
    # are outside what it does. Run it with --filter-kind DELETE.
    "void":         ("baselines.void",     "VoidBaseline"),
    "ditto":        ("baselines.ditto",    "DittoBaseline"),
}


def load_baseline(name: str, **kwargs: Any) -> BaselineModel:
    if name not in BASELINES:
        raise SystemExit(f"Unknown baseline {name!r}. "
                         f"Available: {list(BASELINES)}")
    module_path, class_name = BASELINES[name]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--benchmark-root", type=Path, required=True,
                        help="Directory containing benchmark_manifest.json.")
    parser.add_argument("--predictions-root", type=Path, default=None,
                        help="Where to write predictions/{baseline}/... "
                             "Defaults to {benchmark_root}/predictions/.")
    parser.add_argument("--baseline", type=str, default="stub",
                        choices=sorted(BASELINES),
                        help="Which baseline to run.")

    parser.add_argument("--prompt-flavor", choices=("vague", "quantitative"),
                        default="quantitative")
    parser.add_argument("--prompt-lang", choices=("en", "zh"), default="en")

    parser.add_argument("--filter-property", type=str, default=None,
                        help="Only run edits whose property matches (mass, "
                             "friction, restitution, initial_velocity, "
                             "presence). Default: run everything.")
    parser.add_argument("--filter-scene", type=str, default=None,
                        help="Only run this scene.")
    parser.add_argument("--filter-kind", choices=("SET", "DELETE", "ADD"), default=None)
    parser.add_argument("--filter-timing", choices=("whole_clip", "partway"), default=None,
                        help="Restrict to timed / whole-clip edits. Matches the "
                             "`timing` field on each edit in the manifest.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap number of edits (after filters). Handy for "
                             "smoke tests.")

    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip cases whose output mp4 already exists.")

    parser.add_argument("--num-shards", type=int, default=1,
                        help="Split the (filtered) edit list into N shards so "
                             "several GPUs can run disjoint subsets. Videos go "
                             "to the same tree; each shard writes its own "
                             "run_manifest.shard{i}of{N}.json.")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="Which shard this process handles (0-based).")

    # Baseline hyperparameters (a subset -- everything the wan_vace baseline
    # exposes has a default in its __init__).
    parser.add_argument("--model-id", type=str, default=None,
                        help="Override the model id / local checkout path.")
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None,
                        help="Frames the baseline generates. ditto defaults to "
                             "73 (its LoRA's trained length); 96 covers the "
                             "whole benchmark clip.")
    parser.add_argument("--enable-cpu-offload", action="store_true",
                        help="Legacy alias for --offload-mode sequential.")
    parser.add_argument("--offload-mode",
                        choices=("sequential", "group", "model", "none"),
                        default=None,
                        help="Offload strategy for wan_vace. See baselines/wan_vace.py.")
    parser.add_argument("--num-blocks-per-group", type=int, default=None,
                        help="Blocks per group when --offload-mode group.")

    return parser.parse_args()


def collect_baseline_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kw: dict[str, Any] = {}
    for key in ("model_id", "num_inference_steps", "guidance_scale",
                "height", "width", "seed", "num_frames"):
        val = getattr(args, key)
        if val is not None:
            kw[key] = val
    if args.enable_cpu_offload:
        kw["enable_cpu_offload"] = True
    if args.offload_mode is not None:
        kw["offload_mode"] = args.offload_mode
    if args.num_blocks_per_group is not None:
        kw["num_blocks_per_group"] = args.num_blocks_per_group
    return kw


def pick_prompt(edit: dict[str, Any], flavor: str, lang: str) -> str:
    return edit["prompts"][flavor][lang]


def main() -> None:
    args = parse_args()
    manifest_path = args.benchmark_root / "benchmark_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    predictions_root = args.predictions_root or (args.benchmark_root / "predictions")
    baseline_root = predictions_root / args.baseline
    videos_dir = baseline_root / "videos"
    baseline_root.mkdir(parents=True, exist_ok=True)

    # Filter.
    edits = manifest["edits"]
    if args.filter_scene:
        edits = [e for e in edits if e["scene"] == args.filter_scene]
    if args.filter_property:
        edits = [e for e in edits if (e["property"] or "presence") == args.filter_property]
    if args.filter_kind:
        edits = [e for e in edits if e["edit_kind"] == args.filter_kind]
    if args.filter_timing:
        edits = [e for e in edits if e.get("timing") == args.filter_timing]
    if args.limit is not None:
        edits = edits[:args.limit]
    if args.num_shards > 1:
        if not 0 <= args.shard_index < args.num_shards:
            raise SystemExit(f"--shard-index must be in [0, {args.num_shards}), "
                             f"got {args.shard_index}")
        edits = edits[args.shard_index::args.num_shards]
    print(f"[run] {len(edits)} edits after filters "
          f"(scene={args.filter_scene}, property={args.filter_property}, "
          f"kind={args.filter_kind}, timing={args.filter_timing}, "
          f"limit={args.limit}, "
          f"shard={args.shard_index}/{args.num_shards})")

    baseline = load_baseline(args.baseline, **collect_baseline_kwargs(args))
    print(f"[run] baseline={baseline.name} config={baseline.config}")
    baseline.setup()

    run_records: list[dict[str, Any]] = []
    shard_suffix = ("" if args.num_shards == 1
                    else f".shard{args.shard_index}of{args.num_shards}")
    run_manifest_path = baseline_root / f"run_manifest{shard_suffix}.json"

    def flush() -> None:
        run_manifest_path.write_text(json.dumps({
            "baseline": args.baseline,
            "baseline_config": baseline.config,
            "prompt_flavor": args.prompt_flavor,
            "prompt_lang":   args.prompt_lang,
            "shard_index":   args.shard_index,
            "num_shards":    args.num_shards,
            "benchmark_root": str(args.benchmark_root.resolve()),
            "runs": run_records,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    t_start = time.perf_counter()
    try:
        for i, edit in enumerate(edits, 1):
            source_video = args.benchmark_root / edit["source_video"]
            out_path = videos_dir / f"{edit['scene']}/{edit['case_id']}.mp4"
            prompt = pick_prompt(edit, args.prompt_flavor, args.prompt_lang)

            if args.skip_existing and out_path.exists():
                run_records.append({
                    "global_id": edit["global_id"],
                    "status": "skipped_existing",
                    "output_video": str(out_path.resolve()),
                })
                print(f"[{i}/{len(edits)}] skip existing {edit['global_id']}")
                flush()
                continue

            try:
                print(f"[{i}/{len(edits)}] {edit['global_id']}  ({args.prompt_lang}/{args.prompt_flavor}) -> {out_path.name}")
                result = baseline.edit_video(
                    source_video=source_video, prompt=prompt,
                    output_path=out_path, edit_info=edit,
                )
                run_records.append({
                    "global_id": edit["global_id"],
                    "status": "completed",
                    "output_video": str(result.output_video.resolve()),
                    "elapsed_sec": round(result.elapsed_sec, 3),
                    "prompt": prompt,
                    "extra": result.extra,
                })
            except Exception as exc:                       # noqa: BLE001
                traceback.print_exc()
                run_records.append({
                    "global_id": edit["global_id"],
                    "status": "failed",
                    "error": str(exc),
                    "prompt": prompt,
                })
            flush()
    finally:
        baseline.teardown()

    elapsed = time.perf_counter() - t_start
    n_ok = sum(1 for r in run_records if r["status"] == "completed")
    n_fail = sum(1 for r in run_records if r["status"] == "failed")
    n_skip = sum(1 for r in run_records if r["status"] == "skipped_existing")
    print(f"\n[run] done in {elapsed/60:.1f} min: "
          f"{n_ok} completed, {n_fail} failed, {n_skip} skipped_existing")
    print(f"[run] manifest -> {run_manifest_path}")


if __name__ == "__main__":
    main()
