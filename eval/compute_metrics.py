"""Score a baseline's predictions against the benchmark's ground truth.

Reads:
    {benchmark_root}/benchmark_manifest.json      (edits + GT paths)
    {predictions_root}/{baseline}/videos/{scene}/{case_id}.mp4  (predictions)

Writes:
    {predictions_root}/{baseline}/metrics.json    per-case + aggregate
    {predictions_root}/{baseline}/metrics.csv     flat table for pandas

Usage:
    python compute_metrics.py --benchmark-root /path/to/pcve_benchmark_v1 \\
                              --baseline wan_vace_14b \\
                              --no-lpips --no-clip     # optional switches
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
from statistics import fmean
from typing import Any

from metrics.fvd import I3DFeatures, frechet_distance
from metrics.perceptual import PerceptualMetrics
from metrics.physics import PhysicsScorer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--benchmark-root", type=Path, required=True)
    p.add_argument("--predictions-root", type=Path, default=None,
                   help="Defaults to {benchmark_root}/predictions/.")
    p.add_argument("--baseline", type=str, default=None,
                   help="Required when scoring predictions; unused for --precompute-tracks.")
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--no-psnr",  action="store_true")
    p.add_argument("--no-ssim",  action="store_true")
    p.add_argument("--no-lpips", action="store_true")
    p.add_argument("--no-clip",  action="store_true")
    p.add_argument("--no-physics", action="store_true")
    p.add_argument("--no-fvd", action="store_true",
                   help="Skip the Frechet Video Distance pass.")
    p.add_argument("--physics-only", action="store_true",
                   help="Skip the perceptual metrics; trajectory error only.")
    p.add_argument("--margin-radii", type=float, default=1.0,
                   help="Border margin, in object radii, for the in-frame gate.")

    p.add_argument("--precompute-tracks", action="store_true",
                   help="Track and cache the edited + source videos for every "
                        "case, then exit. Needs no predictions.")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Overall + per-property + per-scene means for each metric."""
    def mean(field: str, subset: list[dict]) -> float | None:
        vs = [r[field] for r in subset if r.get(field) is not None]
        return fmean(vs) if vs else None

    metric_keys = ["psnr", "ssim", "lpips", "clip_sim",
                   "disp_edited_px", "disp_affected_px", "disp_all_px",
                   "disp_all_radii", "abs_all_px", "null_disp_px", "gap_closed",
                   "removal_gap", "onset_err_frames", "mask_iou"]
    out: dict[str, Any] = {"overall": {k: mean(k, rows) for k in metric_keys}}
    for group_key in ("property", "scene", "edit_kind"):
        buckets: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            key = r.get(group_key) or "presence"
            buckets[key].append(r)
        out[f"by_{group_key}"] = {
            k: {m: mean(m, sub) for m in metric_keys}
            for k, sub in sorted(buckets.items())
        }
    out["objects"] = {
        "measured":      sum(r.get("n_measured") or 0 for r in rows),
        "total":         sum(r.get("n_objects") or 0 for r in rows),
        "per_case_mean": (round(fmean([r["n_measured"] for r in rows
                                       if r.get("n_measured") is not None]), 2)
                          if any(r.get("n_measured") is not None for r in rows)
                          else None),
    }
    out["counts"] = {
        "total_cases":   len(rows),
        "completed":     sum(1 for r in rows if r.get("status") == "completed"),
        "missing_pred":  sum(1 for r in rows if r.get("status") == "missing_prediction"),
        "failed":        sum(1 for r in rows if r.get("status") == "failed"),
    }
    return out


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.benchmark_root / "benchmark_manifest.json").read_text())
    predictions_root = args.predictions_root or (args.benchmark_root / "predictions")
    if args.precompute_tracks:
        # precompute only tracks GT / source videos; no baseline needed.
        baseline_dir = videos_dir = None
    else:
        if not args.baseline:
            raise SystemExit(
                "--baseline is required unless --precompute-tracks is set.")
        baseline_dir = predictions_root / args.baseline
        videos_dir = baseline_dir / "videos"
        if not baseline_dir.exists():
            raise SystemExit(f"No predictions found at {baseline_dir}")

    pm = PerceptualMetrics(
        device=args.device,
        enable_psnr=not args.no_psnr and not args.physics_only,
        enable_ssim=not args.no_ssim and not args.physics_only,
        enable_lpips=not args.no_lpips and not args.physics_only,
        enable_clip=not args.no_clip and not args.physics_only,
    )
    scorer = None
    if not args.no_physics:
        scorer = PhysicsScorer(args.benchmark_root, device=args.device,
                               margin_radii=args.margin_radii)

    edits = manifest["edits"]
    by_id = {e["global_id"]: e for e in edits}
    if args.limit is not None:
        edits = edits[:args.limit]

    if args.precompute_tracks:
        t0 = time.perf_counter()
        done = 0
        for i, edit in enumerate(edits, 1):
            r = scorer.precompute(edit)
            done += r["status"] == "cached"
            print(f"[{i}/{len(edits)}] {edit['global_id']:52s} {r['status']:16s} "
                  f"{r.get('n_tracked', 0)} objects")
        print(f"\n[precompute] {done}/{len(edits)} cases cached in "
              f"{time.perf_counter() - t0:.0f}s -> {scorer.cache_dir}")
        return

    rows: list[dict[str, Any]] = []
    t_start = time.perf_counter()
    for i, edit in enumerate(edits, 1):
        pred_path = videos_dir / f"{edit['scene']}/{edit['case_id']}.mp4"
        gt_video  = args.benchmark_root / edit["edited_video"]
        gt_json   = args.benchmark_root / edit["ground_truth"]

        row: dict[str, Any] = {
            "global_id":  edit["global_id"],
            "scene":      edit["scene"],
            "case_id":    edit["case_id"],
            "edit_kind":  edit["edit_kind"],
            "property":   edit["property"],
            "outcome_type": edit["outcome_type"],
            "pred_video": str(pred_path),
        }
        if not pred_path.exists():
            row["status"] = "missing_prediction"
            rows.append(row)
            print(f"[{i}/{len(edits)}] MISSING  {edit['global_id']}")
            continue

        try:
            row["status"] = "completed"
            if not args.physics_only:
                s = pm.score(pred_path, gt_video)
                row.update({
                    "psnr":       s.psnr,
                    "ssim":       s.ssim,
                    "lpips":      s.lpips,
                    "clip_sim":   s.clip_sim,
                    "num_frames": s.num_frames,
                })
            if scorer is not None:
                phy = scorer.score(pred_path, edit)
                row.update({
                    "physics_status":   phy.status,
                    "disp_edited_px":   phy.disp_edited_px,
                    "disp_affected_px": phy.disp_affected_px,
                    "disp_all_px":      phy.disp_all_px,
                    "disp_all_radii":   phy.disp_all_radii,
                    "abs_all_px":       phy.abs_all_px,
                    "null_disp_px":     phy.null_disp_px,
                    "gap_closed":       phy.gap_closed,
                    "removal_gap":      phy.removal_gap,
                    "onset_err_frames": phy.onset_err_frames,
                    "mask_iou":         phy.mask_iou,
                    "n_removals":       phy.n_removals,
                    "n_measured":       phy.n_measured,
                    "n_objects":        phy.n_objects,
                    "physics_objects":  phy.objects,
                    "physics_note":     phy.note,
                })
            per = "".join(
                f"  {k}={row[k]:6.2f}" for k in
                ("psnr", "ssim", "lpips", "clip_sim") if row.get(k) is not None)
            tr = ""
            if scorer is not None:
                d = row.get("disp_all_px")
                g = row.get("gap_closed")
                tr = (f"  disp={d:7.2f}px" if d is not None
                      else f"  disp={'n/a':>7s}  ") + \
                     (f" gap_closed={g:6.2f}" if g is not None else " gap_closed=   n/a") + \
                     f" ({row.get('n_measured')}/{row.get('n_objects')} obj)"
                rg = row.get("removal_gap")
                if rg is not None:
                    oe = row.get("onset_err_frames")
                    tr += (f"  removal={rg:5.2f}" +
                           (f" onset={oe:+.0f}f" if oe is not None else " onset=n/a"))
            print(f"[{i}/{len(edits)}] {edit['global_id']:52s}{per}{tr}")
        except Exception as exc:                           # noqa: BLE001
            traceback.print_exc()
            row["status"] = "failed"
            row["error"] = str(exc)
        rows.append(row)

    # ---- FVD ------------------------------------------------------------
    # A distribution metric: it compares the SET of predictions against the set
    # of references, so it has no per-case value and is computed once, here,
    # over every case that produced a video. Same I3D features every FVD since
    # StyleGAN-V uses, so the number is comparable with published ones.
    fvd = None
    if not args.no_fvd:
        done = [r for r in rows if r.get("status") == "completed"]
        try:
            import imageio.v3 as iio
            i3d = I3DFeatures(device=args.device)
            feats_pred, feats_ref = [], []
            for i, r in enumerate(done, 1):
                edit = by_id[r["global_id"]]
                ref_path = args.benchmark_root / edit["edited_video"]
                pred = iio.imread(r["pred_video"], plugin="pyav")
                ref = iio.imread(str(ref_path), plugin="pyav")
                feats_pred.append(i3d(pred))
                feats_ref.append(i3d(ref))
                if i % 25 == 0:
                    print(f"[fvd] {i}/{len(done)}")
            if len(feats_pred) >= 2:
                fvd = frechet_distance(np.stack(feats_ref),
                                       np.stack(feats_pred))
                print(f"[fvd] over {len(feats_pred)} clips: {fvd:.2f}")
            else:
                print("[fvd] skipped: needs at least 2 scored cases")
        except Exception as exc:                       # noqa: BLE001
            traceback.print_exc()
            print(f"[fvd] failed: {exc}")

    elapsed = time.perf_counter() - t_start
    summary = aggregate([r for r in rows if r.get("status") == "completed"])
    summary["fvd"] = fvd
    out = {
        "baseline": args.baseline,
        "benchmark_root": str(args.benchmark_root.resolve()),
        "num_cases_evaluated": len(rows),
        "elapsed_sec": round(elapsed, 1),
        "aggregate": summary,
        "per_case": rows,
    }
    (baseline_dir / "metrics.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    # Flat CSV -- easier to load into pandas for the paper's tables.
    csv_fields = ["global_id", "scene", "case_id", "edit_kind", "property",
                  "outcome_type", "status", "psnr", "ssim", "lpips",
                  "clip_sim", "num_frames", "physics_status", "disp_edited_px",
                  "disp_affected_px", "disp_all_px", "disp_all_radii", "abs_all_px", "null_disp_px", "gap_closed",
                  "removal_gap", "onset_err_frames", "mask_iou",
                  "n_removals", "n_measured", "n_objects"]
    with (baseline_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # Per-object table. The case-level means hide exactly what the multi-object
    # metric exists to show -- whether the model moved only the named object --
    # so the objects get their own rows.
    obj_fields = ["global_id", "scene", "case_id", "edit_kind", "property",
                  "object", "role", "status", "deleted", "disp_mean_px",
                  "disp_p95_px", "disp_final_px", "disp_mean_radii",
                  "abs_mean_px", "abs_mean_px_vs_projection",
                  "disp_mean_px_vs_projection", "radius_px", "n_scored", "n_tracked",
                  "frames_lost", "ref_frames_lost",
                  "removal_gap", "onset_err_frames", "mask_iou",
                  "mask_iou_frames", "removal_window_frames",
                  "wrong_present_frames",
                  "gt_gone_frame", "pred_gone_frame",
                  "n_offscreen", "n_clipped", "seed", "anchor", "prompt",
                  "ref_disp_px", "ref_disp_radii", "ref_verdict",
                  "null_disp_px", "gap_closed",
                  "gt_offscreen_frame", "pred_lost_frame", "present_fraction",
                  "mask_area_ratio", "divergence_px", "text", "anchor_frame",
                  # ADD cases: whether anything was placed at all, which
                  # division point of the line the prompt asked for, and how
                  # far off the placement was in apparent radii.
                  "added", "add_detected", "add_area_ratio", "add_at_asked",
                  "add_measured_from", "add_fraction_asked",
                  "add_err_radii", "add_err_radii_along"]
    with (baseline_dir / "metrics_objects.csv").open("w", encoding="utf-8",
                                                     newline="") as f:
        w = csv.DictWriter(f, fieldnames=obj_fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            for name, o in (r.get("physics_objects") or {}).items():
                w.writerow({**{k: r.get(k) for k in
                               ("global_id", "scene", "case_id", "edit_kind",
                                "property")},
                            "object": name, **o})

    print(f"\n[metrics] wrote {baseline_dir/'metrics.json'}, "
          f"{baseline_dir/'metrics.csv'} and "
          f"{baseline_dir/'metrics_objects.csv'}")
    print(f"[metrics] overall = {summary['overall']}")


if __name__ == "__main__":
    main()
