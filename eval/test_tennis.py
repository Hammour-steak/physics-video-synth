"""Test tennis_flight edits (previously unmeasurable): try precompute + inspect."""
import os, json
from pathlib import Path
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
import numpy as np
from metrics.physics import PhysicsScorer

BENCH = Path('/remote-home/chenyuanjie/physics-video-synth/pcve_benchmark_v1')
m = json.load(open(BENCH / 'benchmark_manifest.json'))
tennis = [e for e in m['edits'] if e['scene'] == 'tennis_flight']
print(f'tennis_flight has {len(tennis)} edits')

scorer = PhysicsScorer(BENCH)
for edit in tennis:
    print(f"\n== {edit['global_id']} ==")
    got = scorer._plan(edit)
    if got is None:
        print("  _plan returned None"); continue
    *_, prompts, plan = got
    for n, r in plan.items():
        anchor = r.get('anchor_frame')
        print(f"  {n:14s} status={r['status']} anchor={anchor} text={r.get('text')!r}")
    r = scorer.precompute(edit)
    print(f"  precompute: {r}")
    # inspect tracked positions
    src_files = list((scorer.cache_dir).glob(f"tennis_flight__SRC__*.npz"))
    edit_files = list((scorer.cache_dir).glob(f"tennis_flight__{edit['case_id']}__*.npz"))
    for tag, files in (('SRC', src_files), ('EDIT', edit_files)):
        if not files: continue
        z = np.load(files[0])
        for k in sorted(z.files):
            if k.endswith('__uv'):
                uv = z[k]
                vis = np.isfinite(uv[:, 0])
                if vis.any():
                    idx = np.where(vis)[0]
                    print(f"    [{tag}] {k}: {vis.sum()}/{len(uv)} vis, "
                          f"first@{idx[0]}=({uv[idx[0]][0]:.0f},{uv[idx[0]][1]:.0f}) "
                          f"last@{idx[-1]}=({uv[idx[-1]][0]:.0f},{uv[idx[-1]][1]:.0f})")
                else:
                    print(f"    [{tag}] {k}: no visible frames")
