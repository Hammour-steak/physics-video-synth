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

import importlib
import json
import os
import re
import shutil
import subprocess
import sys
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


# The manifest carries the resolved from/to for every SET, so this only has
# to recognise the two shapes the edit string itself comes in: a factor off
# the baseline, or the absolute escape hatch for a baseline of zero.
DSL_SET_RE = re.compile(
    r"^SET\s+(\w+)\.(\w+)\s+(?:TIMES\s+(\d+\s*/\s*\d+|[-\d.eE+]+)"
    r"|TO\s+(\S.*?))\s*(?:HINT.*)?$"
)
DSL_DEL_RE = re.compile(r"^DELETE\s+(\w+)\s*$")
# "AT FRAME n" says the edit lands partway through the clip instead of holding
# from the first frame. It is lifted off before the statement is matched, the
# same way the DSL module parses it.
DSL_TIMING_RE = re.compile(
    r"^(?P<body>.*?)\s+AT\s+FRAME\s+(?P<frame>\d+)"
    r"(?P<hint>(?:\s+HINT\s+\"[^\"]*\")?)\s*$"
)
DSL_ADD_RE = re.compile(
    r"^ADD\s+(\w+)\s+BETWEEN\s+(\w+)\s+AND\s+(\w+)"
    r"\s+AT\s+(?:(MIDPOINT)|(\d+)\s*/\s*(\d+)\s+FROM\s+(\w+))\s*(?:HINT.*)?$"
)


def parse_dsl(dsl: str) -> dict:
    dsl = dsl.strip()
    at_frame = None
    m = DSL_TIMING_RE.match(dsl)
    if m:
        dsl = (m.group("body") + m.group("hint")).strip()
        at_frame = int(m.group("frame"))
    parsed = _parse_statement(dsl)
    parsed["at_frame"] = at_frame
    return parsed


def _parse_statement(dsl: str) -> dict:
    m = DSL_SET_RE.match(dsl.strip())
    if m:
        return {"kind": "SET", "object": m.group(1), "property": m.group(2),
                "factor": m.group(3), "to_value": m.group(4)}
    m = DSL_DEL_RE.match(dsl.strip())
    if m:
        return {"kind": "DELETE", "object": m.group(1), "property": None,
                "factor": None, "to_value": None}
    m = DSL_ADD_RE.match(dsl.strip())
    if m:
        # ADD carries no property: like DELETE it moves an object along the
        # presence axis, absent -> present. Where it lands is relative
        # geometry and lives in physics_diff, not in a from/to pair.
        midpoint, num, den, anchor = m.group(4), m.group(5), m.group(6), m.group(7)
        return {"kind": "ADD", "object": m.group(1), "property": None,
                "factor": None, "to_value": None,
                "endpoints": [m.group(2), m.group(3)],
                "at": "midpoint" if midpoint else f"{num}/{den}",
                "anchor": anchor}
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
    if kind == "ADD":
        return "object_added"
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


def load_scene_vocab(scene: str):
    """Import one scene's edit_vocab, which every scene names identically."""
    sys.path.insert(0, str(SCRIPTS))
    sys.path.insert(0, str(SCRIPTS / scene))
    try:
        sys.modules.pop("edit_vocab", None)
        return importlib.import_module("edit_vocab")
    finally:
        sys.path.remove(str(SCRIPTS / scene))
        sys.path.remove(str(SCRIPTS))


def write_source_prompts(scene: str, case_dir: Path, source: dict) -> dict:
    """Describe the source video at the two precisions the edits use.

    The description names every editable object as part of saying what it does,
    which is what a viewer of the clip would be told; `vague` says it without a
    number in it and `quantitative` carries the measured figures. A consumer
    asking for one flavour therefore gets a matched pair: this description of
    the source, and the edit instruction written to the same standard.

    `editable_objects` carries the same set in machine-readable form, with each
    object's editable properties. Objects only an ADD edit can bring in are not
    in this video and are left out of both.
    """
    import pcve_edit_dsl as dsl
    vocab = load_scene_vocab(scene).VOCAB
    motion = source.get("description", {})
    payload = {
        "schema_version": 1,
        "case_id": source["case_id"],
        "kind": "source",
        "editable_objects": [
            {k: v for k, v in obj.items() if k not in ("in_video", "can_add")}
            for obj in dsl.editable_objects(vocab) if obj["in_video"]
        ],
        "prompts": {
            flavour: {lang: motion.get(flavour, {}).get(lang, "")
                      for lang in ("zh", "en")}
            for flavour in ("vague", "quantitative")
        },
    }
    (case_dir / "prompts.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> None:
    suite_dirs = sorted(RENDERS.glob("pcve_*_suite"))
    if not suite_dirs:
        raise SystemExit(f"No suites found in {RENDERS}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scenes").mkdir(exist_ok=True)
    (OUT / "thumbnails").mkdir(exist_ok=True)
    (OUT / "splits" / "by_property").mkdir(parents=True, exist_ok=True)
    (OUT / "splits" / "by_timing").mkdir(parents=True, exist_ok=True)

    scenes: list[str] = []
    sources: list[dict] = []
    edits: list[dict] = []
    by_property: dict[str, list[str]] = defaultdict(list)
    by_timing: dict[str, list[str]] = defaultdict(list)

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
        source_prompts = write_source_prompts(scene, src_video_abs.parent, source)
        sources.append({
            "global_id": f"{scene}/{src_case_id}",
            "scene": scene,
            "case_id": src_case_id,
            "video": src_video_rel,
            "duration_sec": video_duration(src_video_abs),
            "prompts": source_prompts["prompts"],
            "editable_objects": source_prompts["editable_objects"],
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
                # 1 for an edit that holds for the whole clip; for a timed edit
                # the frame it lands on, before which the prediction is
                # supposed to match the source video.
                "applies_from_frame": parsed["at_frame"] or 1,
                "timing": "partway" if parsed["at_frame"] else "whole_clip",
                "physics_diff": diff,
                "edit_summary": e.get("edit_summary", ""),
                "ground_truth": f"scenes/{scene}/cases/{case_id}/ground_truth_transforms.json",
                "thumbnail": f"thumbnails/{scene}/{case_id}.jpg",
            })
            axis_key = parsed["property"] or "presence"
            by_property[axis_key].append(global_id)
            by_timing["partway" if parsed["at_frame"] else "whole_clip"].append(global_id)
            make_thumbnail(edited_abs, OUT / "thumbnails" / scene / f"{case_id}.jpg")

        print(f"[ok] {scene}: 1 source + {len(manifest['edits'])} edits")

    # Top-level manifest.
    manifest_out = {
        "schema_version": 1,
        "benchmark_name": BENCHMARK_NAME,
        "prompts_schema": {
            "flavors": ["vague.zh", "vague.en",
                        "quantitative.zh", "quantitative.en"],
            "notes": "quantitative names the number the edit turns on: for "
                     "SET the multiple of the baseline the property moves to "
                     "(5x, 1/4), for ADD the division point of the line "
                     "between two named objects (the midpoint, or the quarter "
                     "point nearer one of them). vague is direction-only "
                     "(increase/decrease/activate/deactivate/change), and for "
                     "ADD a relative position (closer to one endpoint, or "
                     "midway) with no number. Both flavours also state when "
                     "the edit takes effect: quantitative names the frame, "
                     "vague places it in the clip (from the start / early / "
                     "about halfway / late).",
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
                kind: sum(1 for e in edits if e["edit_kind"] == kind)
                for kind in ("SET", "DELETE", "ADD")
            },
            "by_timing": {k: len(v) for k, v in sorted(by_timing.items())},
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
    for bucket, ids in by_timing.items():
        (OUT / "splits" / "by_timing" / f"{bucket}.txt").write_text(
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
  friction / restitution / initial_velocity / presence for DELETE and ADD
  edits, which both move an object along the present/absent axis).
- `splits/by_timing/`: same, bucketed by when the edit takes effect --
  `whole_clip.txt` for edits that hold from the first frame, and
  `partway.txt` for edits that land at `applies_from_frame`, where the
  frames before it are supposed to match the source video exactly.

## Prompts
Each edit has 4 flavors under `prompts`:
- `vague.zh` / `vague.en`: direction-only ("调大一点" / "Decrease X").
- `quantitative.zh` / `quantitative.en`: includes `from` and `to` numbers.

Every prompt opens by saying when the edit takes effect, because that is
part of the instruction: "From frame 1 onwards" / "From the start of the
clip" for the whole-clip edits, and "From frame 22 onwards" / "Early in
the clip" for one that lands partway through. `applies_from_frame` in the
manifest is the same fact in machine-readable form.

## Task
Given (`source_video`, prompt), generate a video that matches
`edited_video`. Compare against `ground_truth_transforms.json` for
per-frame object trajectories, or against `edited_video` for pixel /
perceptual metrics.

## Counts
- Edits by kind: {manifest_out['counts']['by_kind']}
- Edits by timing: {manifest_out['counts']['by_timing']}
- Edits by property: {manifest_out['counts']['by_property']}
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    print(f"\n[bench] {len(scenes)} scenes, {len(sources)} sources, {len(edits)} edits")
    print(f"[bench] wrote {OUT/'benchmark_manifest.json'}")
    print(f"[bench] by_property: {manifest_out['counts']['by_property']}")


if __name__ == "__main__":
    main()
