"""Consolidate every pcve_{scene}_suite render into a single benchmark bundle.

Reads /remote-home/chenyuanjie/physics-video-synth/renders/pcve_*_suite (schema v3)
and writes /remote-home/chenyuanjie/physics-video-synth/pcve_benchmark_v1/ with:

    benchmark_manifest.json          flat index of every source + edit
    scenes/{scene}/suite_manifest.json + cases/{case_id}/...  (copied verbatim)
    vocab/pcve_edit_dsl.py + vocab/{scene}_edit_vocab.py
    thumbnails/{scene}/{case_id}.jpg (mid-frame preview)
    splits/all.txt + splits/by_property/{mass,friction,...}.txt

Idempotent: safe to re-run; only rewrites the manifest + missing thumbnails.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path


def _find_ffmpeg() -> str | None:
    """Return an ffmpeg binary path, or None if none is available.

    Prefers the system PATH, then falls back to the imageio-ffmpeg bundled
    binary. Thumbnail generation is skipped silently when ffmpeg is missing.
    """
    for name in ("ffmpeg",):
        found = shutil.which(name)
        if found:
            return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


FFMPEG = _find_ffmpeg()

WORKSPACE = Path("/remote-home/chenyuanjie/physics-video-synth")
RENDERS   = WORKSPACE / "renders"
SCRIPTS   = WORKSPACE / "scripts"
OUT       = WORKSPACE / "pcve_benchmark_v1"
BENCHMARK_NAME = "pcve_benchmark_v1"

# Files copied per case. Everything else (scenario_metadata.json,
# scenario_overrides.json, per-scene suite_manifest.json, vocab modules) is
# reproduction-only material for the render pipeline and is dropped from the
# distributed benchmark; the top-level benchmark_manifest.json is authoritative.
CASE_FILES_KEEP = {"video.mp4", "prompts.json", "ground_truth_transforms.json"}


DSL_SET_RE = re.compile(r"^SET\s+(\w+)\.(\w+)\s+FROM\s+(\S.*?)\s+TO\s+(\S.*?)\s*(?:HINT.*)?$")
DSL_DEL_RE = re.compile(r"^DELETE\s+(\w+)\s*$")


def parse_dsl(dsl: str) -> dict:
    m = DSL_SET_RE.match(dsl.strip())
    if m:
        return {"kind": "SET", "object": m.group(1), "property": m.group(2),
                "from_value": m.group(3), "to_value": m.group(4)}
    m = DSL_DEL_RE.match(dsl.strip())
    if m:
        return {"kind": "DELETE", "object": m.group(1), "property": None,
                "from_value": None, "to_value": None}
    raise ValueError(f"Cannot parse DSL: {dsl!r}")


def classify_outcome(dsl_parsed: dict, from_v, to_v) -> str:
    """Coarse semantic label from the DSL alone (no video parsing).

    Used to bucket edits along `outcome_type` axis in the manifest. The label
    is deliberately coarse -- it says how the *input* was moved, not what the
    physics does. Anything more nuanced belongs in edit_summary.
    """
    kind = dsl_parsed["kind"]
    prop = dsl_parsed["property"]
    if kind == "DELETE":
        return "object_removed"
    try:
        fv = float(from_v); tv = float(to_v)
    except (TypeError, ValueError):
        return f"{prop}_change"
    delta = "increase" if tv > fv else "decrease"
    return f"{prop}_{delta}"


_DUR_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


def video_duration(video_path: Path) -> float | None:
    if not FFMPEG or not video_path.exists():
        return None
    # ffmpeg -i on stderr always prints "Duration: HH:MM:SS.ff". Cheaper than
    # adding ffprobe as a separate dependency.
    r = subprocess.run(
        [FFMPEG, "-i", str(video_path)],
        capture_output=True, text=True,
    )
    m = _DUR_RE.search(r.stderr)
    if not m:
        return None
    h, mi, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
    return h * 3600.0 + mi * 60.0 + s


def make_thumbnail(video_path: Path, thumb_path: Path, at_sec: float | None = None) -> None:
    if not FFMPEG or thumb_path.exists() or not video_path.exists():
        return
    thumb_path.parent.mkdir(parents=True, exist_ok=True)
    dur = video_duration(video_path) or 3.0
    ts = at_sec if at_sec is not None else max(0.05, dur / 2.0)
    subprocess.run(
        [FFMPEG, "-y", "-loglevel", "error",
         "-ss", f"{ts:.2f}", "-i", str(video_path),
         "-frames:v", "1", "-q:v", "3", str(thumb_path)],
        check=True,
    )


def main() -> None:
    suite_dirs = sorted(RENDERS.glob("pcve_*_suite"))
    if not suite_dirs:
        raise SystemExit(f"No suites found in {RENDERS}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scenes").mkdir(exist_ok=True)
    (OUT / "thumbnails").mkdir(exist_ok=True)
    (OUT / "splits" / "by_property").mkdir(parents=True, exist_ok=True)

    scenes: list[str] = []
    sources: list[dict] = []
    edits: list[dict] = []
    by_property: dict[str, list[str]] = defaultdict(list)

    for suite_dir in suite_dirs:
        scene = suite_dir.name[len("pcve_"):-len("_suite")]
        manifest_src = suite_dir / "suite_manifest.json"
        if not manifest_src.exists():
            print(f"[skip] {scene}: no suite_manifest.json")
            continue
        manifest = json.loads(manifest_src.read_text())
        if manifest.get("schema_version") != 3:
            print(f"[skip] {scene}: schema_version != 3 (got {manifest.get('schema_version')!r})")
            continue

        scenes.append(scene)
        scene_dst = OUT / "scenes" / scene
        scene_dst.mkdir(parents=True, exist_ok=True)

        # Copy only the eval-relevant per-case files (see CASE_FILES_KEEP).
        # Everything the render pipeline needs to reproduce a case
        # (scenario_metadata.json, scenario_overrides.json,
        # suite_manifest.json, vocab modules) stays in `renders/` and is
        # excluded from the distributed benchmark.
        cases_src = suite_dir / "cases"
        cases_dst = scene_dst / "cases"
        if cases_dst.exists():
            shutil.rmtree(cases_dst)
        cases_dst.mkdir(parents=True)
        for case_src in sorted(cases_src.iterdir()):
            if not case_src.is_dir():
                continue
            case_dst = cases_dst / case_src.name
            case_dst.mkdir(parents=True, exist_ok=True)
            for fname in CASE_FILES_KEEP:
                fsrc = case_src / fname
                if fsrc.exists():
                    shutil.copy2(fsrc, case_dst / fname)

        # ---- source ----
        source = manifest["source"]
        src_case_id = source["case_id"]
        src_video_rel = f"scenes/{scene}/cases/{src_case_id}/video.mp4"
        src_video_abs = OUT / src_video_rel
        sources.append({
            "global_id": f"{scene}/{src_case_id}",
            "scene": scene,
            "case_id": src_case_id,
            "video": src_video_rel,
            "duration_sec": video_duration(src_video_abs),
            "description": source.get("description", ""),
        })
        make_thumbnail(src_video_abs, OUT / "thumbnails" / scene / f"{src_case_id}.jpg")

        # ---- edits ----
        for e in manifest["edits"]:
            case_id = e["case_id"]
            edited_rel = f"scenes/{scene}/cases/{case_id}/video.mp4"
            edited_abs = OUT / edited_rel
            parsed = parse_dsl(e["edit_dsl"])
            diff = e.get("physics_diff", {})
            # Try to extract from/to for the outcome label. physics_diff is
            # {"prop (obj)": {"from":..., "to":...}} for SET, or
            # {"obj": {"from":"present","to":"removed"}} for DELETE.
            from_v = to_v = None
            for _, v in diff.items():
                from_v = v.get("from"); to_v = v.get("to"); break
            outcome = classify_outcome(parsed, from_v, to_v)

            global_id = f"{scene}/{case_id}"
            edits.append({
                "global_id": global_id,
                "scene": scene,
                "case_id": case_id,
                "source_case_id": e["source_case_id"],
                "source_video": f"scenes/{scene}/cases/{e['source_case_id']}/video.mp4",
                "edited_video": edited_rel,
                "duration_sec": video_duration(edited_abs),
                "prompts": e["prompts"],
                "edit_dsl": e["edit_dsl"],
                "edit_kind": parsed["kind"],
                "object_id": parsed["object"],
                "property": parsed["property"],           # None for DELETE
                "outcome_type": outcome,
                "physics_diff": diff,
                "edit_summary": e.get("edit_summary", ""),
                "ground_truth": f"scenes/{scene}/cases/{case_id}/ground_truth_transforms.json",
                "thumbnail": f"thumbnails/{scene}/{case_id}.jpg",
            })
            axis_key = parsed["property"] or "presence"
            by_property[axis_key].append(global_id)
            make_thumbnail(edited_abs, OUT / "thumbnails" / scene / f"{case_id}.jpg")

        print(f"[ok] {scene}: 1 source + {len(manifest['edits'])} edits")

    # Top-level manifest.
    manifest_out = {
        "schema_version": 1,
        "benchmark_name": BENCHMARK_NAME,
        "prompts_schema": {
            "flavors": ["vague.zh", "vague.en",
                        "quantitative.zh", "quantitative.en"],
            "notes": "quantitative includes the numerical from/to; vague is "
                     "direction-only (increase/decrease/activate/deactivate/change).",
        },
        "properties_vocab": ["mass", "friction", "restitution",
                             "initial_velocity", "presence"],
        "scenes": scenes,
        "counts": {
            "scenes": len(scenes),
            "sources": len(sources),
            "edits": len(edits),
            "by_property": {k: len(v) for k, v in sorted(by_property.items())},
            "by_kind": {
                "SET": sum(1 for e in edits if e["edit_kind"] == "SET"),
                "DELETE": sum(1 for e in edits if e["edit_kind"] == "DELETE"),
            },
        },
        "sources": sources,
        "edits": edits,
    }
    (OUT / "benchmark_manifest.json").write_text(
        json.dumps(manifest_out, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    # Splits: all + by_property.
    (OUT / "splits" / "all.txt").write_text(
        "\n".join(e["global_id"] for e in edits) + "\n", encoding="utf-8",
    )
    for prop, ids in by_property.items():
        (OUT / "splits" / "by_property" / f"{prop}.txt").write_text(
            "\n".join(ids) + "\n", encoding="utf-8",
        )

    # Terse README.
    readme = f"""# {BENCHMARK_NAME}

Physics-Consistent Video Editing benchmark. {len(scenes)} scenes,
{len(sources)} source videos, {len(edits)} edit tasks.

## Layout
- `benchmark_manifest.json`: flat index of every source + edit; the
  authoritative record. Every field the evaluator needs (prompts,
  physics_diff, edit_summary, video/gt paths) is inlined here.
- `scenes/{{scene}}/cases/{{case_id}}/`:
    - `video.mp4` -- the source (in the baseline case) or edited render.
    - `prompts.json` -- redundant with the top-level manifest; kept
      per-case so an evaluator that iterates directories has everything
      local.
    - `ground_truth_transforms.json` -- per-frame world matrices,
      linear/angular velocities and any object-specific quality metrics
      the physics sim produced. Use these for physics-consistency
      metrics; skip them if you are only doing pixel/perceptual eval.
- `thumbnails/{{scene}}/{{case_id}}.jpg`: mid-frame preview per case.
- `splits/all.txt`: every edit's `global_id`, one per line.
- `splits/by_property/`: same, bucketed by the edited property (mass /
  friction / restitution / initial_velocity / presence for DELETE edits).

## Prompts
Each edit has 4 flavors under `prompts`:
- `vague.zh` / `vague.en`: direction-only ("调大一点" / "Decrease X").
- `quantitative.zh` / `quantitative.en`: includes `from` and `to` numbers.

## Task
Given (`source_video`, prompt), generate a video that matches
`edited_video`. Compare against `ground_truth_transforms.json` for
per-frame object trajectories, or against `edited_video` for pixel /
perceptual metrics.

## Counts
- Edits by kind: {manifest_out['counts']['by_kind']}
- Edits by property: {manifest_out['counts']['by_property']}
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    print(f"\n[bench] {len(scenes)} scenes, {len(sources)} sources, {len(edits)} edits")
    print(f"[bench] wrote {OUT/'benchmark_manifest.json'}")
    print(f"[bench] by_property: {manifest_out['counts']['by_property']}")


if __name__ == "__main__":
    main()
