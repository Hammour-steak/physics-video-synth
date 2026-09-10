"""Run precompute on one edit per scene, then report per-object anchor error.

Anchor error = distance from the SAM2 centroid at the anchor frame to the
projected hint. If it's within ~1 radius the text prompt landed on the right
object; larger means GDINO detected the wrong instance (or nothing) and the
text needs tuning.
"""
import os, json, sys, time, glob
from pathlib import Path
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import numpy as np
from metrics.physics import PhysicsScorer

BENCH = Path('/remote-home/chenyuanjie/physics-video-synth/pcve_benchmark_v1')
m = json.load(open(BENCH / 'benchmark_manifest.json'))

# first edit per scene
by_scene = {}
for e in m['edits']:
    by_scene.setdefault(e['scene'], e)   # first wins

scorer = PhysicsScorer(BENCH)
cache_dir = scorer.cache_dir

print(f"\n== scenes: {len(by_scene)}, cache: {cache_dir}", flush=True)
results = []
for scene, edit in sorted(by_scene.items()):
    t0 = time.perf_counter()
    r = scorer.precompute(edit)
    dt = time.perf_counter() - t0
    print(f"\n[{scene}] {r['status']} n={r.get('n_tracked', 0)}/{r.get('n_objects', 0)} "
          f"({dt:.0f}s)", flush=True)
    # inspect the freshly cached source track
    got = scorer._plan(edit)
    if got is None:
        continue
    *_, prompts, plan = got
    src_cache = list(glob.glob(str(cache_dir / f"{scene}__SRC__*.npz")))
    if not src_cache:
        print(f"    no source cache", flush=True); continue
    z = np.load(src_cache[0])
    for n, p in prompts.items():
        uv = z[f"{n}__uv"]
        area = z[f"{n}__area"]
        f = int(p['anchor_frame'])
        hu, hv = p['hint_uv']
        r_px = p.get('hint_radius_px') or 16.0
        if not np.isfinite(uv[f, 0]):
            print(f"    ✗ {n:30s} text={p['text']!r:32s} NO MASK at anchor")
            results.append((scene, n, p['text'], None, hu, hv, None))
            continue
        du = uv[f, 0] - hu; dv = uv[f, 1] - hv
        dist = float(np.hypot(du, dv))
        n_vis = int(np.isfinite(uv[:, 0]).sum())
        tag = 'OK' if dist < r_px else ('OK?' if dist < 2*r_px else 'BAD')
        print(f"    {tag:3s} {n:30s} text={p['text']!r:32s} "
              f"anchor=({uv[f,0]:.0f},{uv[f,1]:.0f}) hint=({hu:.0f},{hv:.0f}) "
              f"d={dist:.0f}px (r={r_px:.0f}) vis={n_vis}/{len(uv)}")
        results.append((scene, n, p['text'], dist, hu, hv, r_px))

# summary
print("\n\n== SUMMARY ==", flush=True)
bad = [x for x in results if x[3] is None or x[3] > 2 * (x[6] or 16)]
okq = [x for x in results if x[3] is not None and (x[6] or 16) <= x[3] <= 2 * (x[6] or 16)]
good = [x for x in results if x[3] is not None and x[3] < (x[6] or 16)]
print(f"GOOD (< 1 radius):  {len(good)}/{len(results)}")
print(f"OK?  (1-2 radii):   {len(okq)}/{len(results)}")
print(f"BAD  (> 2 radii or unassigned): {len(bad)}/{len(results)}")
if bad:
    print("\n-- BAD entries (need text prompt tuning) --")
    for s, n, t, d, hu, hv, r in bad:
        dstr = 'no-mask' if d is None else f'{d:.0f}px'
        print(f"  {s:30s} {n:22s} '{t}' -> {dstr}")
