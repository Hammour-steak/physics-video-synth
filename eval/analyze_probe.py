"""Read cached GroundedSAM2 tracks and compare anchor-frame centroid to
projected hint per object. Bad -> text prompt didn't hit the right object."""
import json, glob, os
from pathlib import Path
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import numpy as np
from metrics.traj_lib import normalise, base_seed
from metrics.physics import _load_text_prompts, load_gt

BENCH = Path('/remote-home/chenyuanjie/physics-video-synth/pcve_benchmark_v1')
CACHE = Path('/remote-home/chenyuanjie/physics-video-synth/eval/metrics/gt_tracks_gsam2')
PROMPTS = Path('/remote-home/chenyuanjie/physics-video-synth/eval/metrics/text_prompts')

m = json.load(open(BENCH / 'benchmark_manifest.json'))
by_scene = {}
for e in m['edits']:
    by_scene.setdefault(e['scene'], e)

def verdict(dist, r):
    r = max(r or 16.0, 12.0)
    if dist is None: return 'NO-MASK'
    if dist < r: return 'GOOD'
    if dist < 2*r: return 'OK?'
    return 'BAD'

results = []
for scene, edit in sorted(by_scene.items()):
    edit_files = [p for p in CACHE.glob(f'{scene}__*.npz') if '__SRC__' not in p.name]
    if not edit_files:
        print(f'== {scene:30s}  NO CACHE'); continue
    z = np.load(edit_files[0])
    base_gt_path = BENCH / 'scenes' / scene / 'cases' / edit['source_case_id'] / 'ground_truth_transforms.json'
    base_gt = load_gt(base_gt_path)
    bobjs = normalise(base_gt)
    texts = _load_text_prompts(scene, PROMPTS)

    print(f'\n== {scene}')
    for n, text in texts.items():
        if n not in bobjs:
            continue
        f_src, uv0, r_px = base_seed(base_gt, n, bobjs)
        if f_src is None: continue
        key = f'{n}__uv'
        if key not in z.files:
            print(f'   NO-TRACK   {n:28s} text={text!r}  (never assigned a box)')
            results.append((scene, n, text, None, r_px)); continue
        uv = z[key]
        if not np.isfinite(uv[f_src, 0]):
            print(f'   NO-MASK    {n:28s} text={text!r}  hint=({uv0[0]:.0f},{uv0[1]:.0f})')
            results.append((scene, n, text, None, r_px)); continue
        dist = float(np.hypot(uv[f_src, 0] - uv0[0], uv[f_src, 1] - uv0[1]))
        v = verdict(dist, r_px)
        print(f'   {v:9s}  {n:28s} text={text!r:38s} anchor=({uv[f_src,0]:.0f},{uv[f_src,1]:.0f}) hint=({uv0[0]:.0f},{uv0[1]:.0f}) d={dist:.0f}px (r={r_px:.0f})')
        results.append((scene, n, text, dist, r_px))

print('\n\n== SUMMARY ==')
good = [r for r in results if r[3] is not None and r[3] < max(r[4] or 16, 12)]
okq  = [r for r in results if r[3] is not None and max(r[4] or 16, 12) <= r[3] < 2*max(r[4] or 16, 12)]
bad  = [r for r in results if r[3] is None or r[3] >= 2*max(r[4] or 16, 12)]
print(f'GOOD (< 1 radius):            {len(good)}/{len(results)}')
print(f'OK?  (1-2 radii):             {len(okq)}/{len(results)}')
print(f'BAD  (>=2 radii or no-mask):  {len(bad)}/{len(results)}')
if bad:
    print('\n-- BAD (need text-prompt tuning) --')
    for s, n, t, d, r in bad:
        dstr = 'no-mask' if d is None else f'{d:.0f}px vs r={r:.0f}'
        print(f'   {s:28s} {n:22s} {t!r:40s} {dstr}')
if okq:
    print('\n-- OK? (borderline) --')
    for s, n, t, d, r in okq:
        print(f'   {s:28s} {n:22s} {t!r:40s} {d:.0f}px vs r={r:.0f}')
