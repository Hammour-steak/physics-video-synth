"""End-to-end .score() test: cached refs + real prediction video."""
import os, json
from pathlib import Path
from pprint import pprint
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

from metrics.physics import PhysicsScorer

BENCH = Path('/remote-home/chenyuanjie/physics-video-synth/pcve_benchmark_v1')
m = json.load(open(BENCH / 'benchmark_manifest.json'))
edit = next(e for e in m['edits']
            if e['global_id'] == 'air_hockey_chain/edit_heavy_middle_mallet')
pred = BENCH / 'predictions' / 'wan_vace_14b' / 'videos' / edit['scene'] / f"{edit['case_id']}.mp4"
print(f"pred exists: {pred.exists()}  size: {pred.stat().st_size if pred.exists() else 0}")

scorer = PhysicsScorer(BENCH)
print("scorer built")

result = scorer.score(pred, edit)
print(f"\n== RESULT for {edit['global_id']} ==")
print(f"status:            {result.status}")
print(f"n_measured:        {result.n_measured}/{result.n_objects}")
print(f"disp_edited_px:    {result.disp_edited_px}")
print(f"disp_affected_px:  {result.disp_affected_px}")
print(f"disp_all_px:       {result.disp_all_px}")
print(f"disp_all_radii:    {result.disp_all_radii}")
print(f"abs_all_px:        {result.abs_all_px}")
print(f"null_disp_px:      {result.null_disp_px}")
print(f"gap_closed:        {result.gap_closed}")
print(f"note:              {result.note}")
print(f"\n-- per object --")
for n, r in result.objects.items():
    print(f"  {n:12s} role={r.get('role'):8s} status={r.get('status')}")
    for k in ['disp_mean_px', 'disp_mean_radii', 'abs_mean_px_vs_projection',
             'null_disp_px', 'gap_closed', 'ref_disp_radii', 'ref_verdict',
             'radius_px', 'anchor_frame']:
        if k in r and r[k] is not None:
            print(f"       {k:32s} = {r[k]}")
