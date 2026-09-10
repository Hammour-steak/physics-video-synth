"""Trajectory metric building blocks.

The ground truth does not use one schema. Three layouts appear across the 20
scenes, so everything goes through normalise() first:

  A  nested dict     frame["mallet_0"]        = {present, matrix_world, ...}
  B  prefix-flat     frame["ball_matrix_world"], frame["ball_location"], ...
  C  list            frame["stones"]          = [{matrix_world, ...}, ...]
                     names come from gt["objects"]["stones"][i]["object_name"]

Projection is exact -- the camera matrices are in the ground truth -- and is
what seeds the tracker, bounds the in-frame gate and sets the error scale.
Tracking itself is GroundedSAM2 (metrics/grounded_sam2_tracker.py).

Everything here is per-object and the scene's whole object set is available:
resolve_targets() returns all of them tagged edited / affected / static, and
traj_error() scores one tracked path, so a caller measures as many objects as
the scene has rather than only the one the edit names.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

META = {"frame_index", "time_sec", "camera_matrix_world",
        "camera_world_to_camera_matrix"}


# ------------------------------------------------------------------ normalise
def normalise(gt: dict) -> dict[str, dict]:
    """-> {object_name: {"mw": (T,4,4) with NaN when absent, "radius": float|None}}"""
    frames = gt["frames"]
    T = len(frames)
    top = gt.get("objects", {})
    out: dict[str, dict] = {}

    def put(name, t, mw, present=True):
        rec = out.setdefault(name, {"mw": np.full((T, 4, 4), np.nan), "radius": None})
        if present and mw is not None:
            rec["mw"][t] = np.asarray(mw, dtype=np.float64)

    for t, fr in enumerate(frames):
        for k, v in fr.items():
            if k in META:
                continue
            # B first: "<name>_matrix_world" -- its value is a 4x4 nested list,
            # so testing isinstance(list) before this would swallow it.
            if k.endswith("_matrix_world"):
                put(k[: -len("_matrix_world")], t, v, True)
                continue
            # C: list of per-object dicts
            if isinstance(v, list):
                meta = top.get(k) or []
                for i, item in enumerate(v):
                    if not isinstance(item, dict):
                        continue
                    nm = None
                    if i < len(meta) and isinstance(meta[i], dict):
                        nm = meta[i].get("object_name")
                    nm = nm or f"{k}_{i}"
                    alive = item.get("present", item.get("active", True))
                    put(nm, t, item.get("matrix_world"), alive)
            # A: nested dict carrying matrix_world
            elif isinstance(v, dict) and "matrix_world" in v:
                alive = v.get("present", v.get("active", True))
                put(k, t, v.get("matrix_world"), alive)

    # A DELETE case does not stop writing matrices: pool_collision's removed
    # cue ball still has a matrix_world on every frame, and only
    # objects.cue_ball.present / physics.objects.cue_ball.active say it was
    # never rendered. Left alone, the removed object looks present and gets
    # scored against a trajectory that is not on screen.
    phys_objs = (gt.get("physics", {}) or {}).get("objects", {}) or {}
    for k, meta in top.items():
        entries = meta if isinstance(meta, list) else [meta]
        for i, m in enumerate(entries):
            if not isinstance(m, dict):
                continue
            names = [m.get("object_name"),
                     f"{k}_{i}" if isinstance(meta, list) else k]
            pm = phys_objs.get(k) if isinstance(phys_objs.get(k), dict) else {}
            dead = (m.get("present") is False or m.get("active") in (0, False)
                    or pm.get("active") in (0, False))
            if not dead:
                continue
            for nm in names:
                if nm in out:
                    out[nm]["mw"][:] = np.nan

    # radii, where the ground truth bothers to record them
    geo = gt.get("physics", {}).get("geometry", {}) or {}
    generic_r = next((float(v) for kk, v in geo.items()
                      if "radius" in kk.lower() and isinstance(v, (int, float))), None)
    for k, meta in top.items():
        entries = meta if isinstance(meta, list) else [meta]
        for i, m in enumerate(entries):
            if not isinstance(m, dict):
                continue
            r = m.get("radius_m_scene_units")
            if not r:
                continue
            # The name a track is filed under is the frame key for layouts A/B
            # ("ball") but object_name for layout C ("bowling_pin_0"), and the
            # two disagree -- bowling's ball is objects["ball"] with
            # object_name "bowling_ball". Try every spelling, or the radius is
            # silently dropped and the error scale falls back to 16 px.
            for nm in (m.get("object_name"),
                       f"{k}_{i}" if isinstance(meta, list) else k):
                if nm in out:
                    out[nm]["radius"] = float(r)
    # physics.objects records sizes too, and for half the scenes it is the only
    # place they appear: objects.soccer_ball has no radius at all while
    # physics.objects.soccer_ball.radius is 0.11. Falling back to a default
    # 16 px is not harmless -- the error scale and the border margin are both
    # in radii, so a wrong size mislabels a good track as unusable.
    phys = (gt.get("physics", {}) or {}).get("objects", {}) or {}
    for key, tgt in _physics_map(gt, list(out)).items():
        v = phys.get(key)
        if not isinstance(v, dict):
            continue
        r = v.get("radius")
        if not isinstance(r, (int, float)) or not r:
            # No radius, but a box: dominoes give thickness/width/height, the
            # toy car gives half_extents. Half the largest side is the closest
            # thing to an apparent radius such a shape has.
            dims = [d for kk in ("dimensions_scene_units", "half_extents")
                    for d in (v.get(kk) or []) if isinstance(d, (int, float))]
            sides = [v[kk] for kk in ("width", "height", "thickness", "length")
                     if isinstance(v.get(kk), (int, float))]
            if v.get("half_extents"):
                r = max(dims) if dims else None
            elif dims or sides:
                r = max(dims + sides) / 2.0
        if not isinstance(r, (int, float)) or not r:
            continue
        for nm in tgt:
            if nm in out and out[nm]["radius"] is None:
                out[nm]["radius"] = float(r)

    for rec in out.values():
        if rec["radius"] is None:
            rec["radius"] = generic_r
    return out


# ------------------------------------------------------------------ projection
class Camera:
    def __init__(self, cam: dict):
        self.W, self.H = cam["resolution"]
        # dining_chain omits sensor_width_mm; 36 mm is the Blender default.
        self.fx = cam["lens_mm"] / cam.get("sensor_width_mm", 36.0) * self.W
        self.cx, self.cy = self.W / 2.0, self.H / 2.0

    def project(self, P, cam_mw):
        Rt = np.linalg.inv(np.asarray(cam_mw, dtype=np.float64))
        x, y, z = (Rt @ np.array([*P, 1.0]))[:3]
        if z >= 0:
            return None
        return self.fx * (x / -z) + self.cx, self.cy - self.fx * (y / -z), -z


def projected_track(gt: dict, obj: str, objs: dict | None = None,
                    camera_from: dict | None = None):
    """-> uv (T,2) NaN where absent, present (T,), radius_px (T,).

    camera_from views another clip's world trajectory through THIS clip's
    camera. ball_block re-randomises camera position and focal length for
    every edited render, so projecting an edited trajectory with its own
    camera and comparing it against a prediction -- which inherits the source
    clip's camera -- would measure the viewpoint change, not the physics.
    """
    objs = objs if objs is not None else normalise(gt)
    rec = objs[obj]
    cam_src = camera_from if camera_from is not None else gt
    cam = Camera(cam_src["camera"])
    r_world = rec["radius"]
    uv = np.full((len(gt["frames"]), 2), np.nan)
    pres = np.zeros(len(gt["frames"]), bool)
    rad = np.full(len(gt["frames"]), np.nan)
    cam_frames = cam_src["frames"]
    for t, fr in enumerate(gt["frames"]):
        mw = rec["mw"][t]
        if not np.isfinite(mw).all():
            continue
        cf = cam_frames[t] if t < len(cam_frames) else cam_frames[-1]
        got = cam.project(mw[:3, 3], cf["camera_matrix_world"])
        if got is None:
            continue
        u, v, depth = got
        uv[t] = [u, v]; pres[t] = True
        if r_world:
            rad[t] = r_world * cam.fx / depth
    return uv, pres, rad


def project_point(gt: dict, xyz, frame: int, camera_from: dict | None = None):
    """Project one world-space point onto a frame, in source-camera pixels.

    projected_track needs an object to follow; this takes a bare coordinate,
    which is what lets an ADD's line endpoints be located from the numbers the
    edit recorded rather than by matching DSL object ids against render mesh
    names. Returns None when the point is behind the camera or the frame does
    not exist.
    """
    cam_src = camera_from if camera_from is not None else gt
    cam = Camera(cam_src["camera"])
    cam_frames = cam_src["frames"]
    if not cam_frames:
        return None
    cf = cam_frames[frame] if frame < len(cam_frames) else cam_frames[-1]
    got = cam.project(np.asarray(xyz, float)[:3], cf["camera_matrix_world"])
    if got is None:
        return None
    u, v, _depth = got
    return np.array([u, v], float)


# ---------------------------------------------------------------------- error
def in_frame(uv: np.ndarray, resolution, radius_px: float = 0.0,
             margin_radii: float = 0.0):
    """-> (fully_inside, centre_inside). Both (T,) bool.

    An object straddling the border is the awkward case: its centre is still in
    the image, but the mask is cut off by the edge, so the centroid gets pulled
    inward while the ground-truth ORIGIN is not pulled anywhere. Comparing the
    two then measures the crop. `margin_radii` keeps only frames where the whole
    object clears the border.
    """
    W, H = resolution
    mg = float(margin_radii) * float(radius_px)
    centre = ((uv[:, 0] >= 0) & (uv[:, 0] < W)
              & (uv[:, 1] >= 0) & (uv[:, 1] < H))
    full = ((uv[:, 0] >= mg) & (uv[:, 0] < W - mg)
            & (uv[:, 1] >= mg) & (uv[:, 1] < H - mg))
    return full & centre, centre


def traj_error(tuv: np.ndarray, ref_uv: np.ndarray, ref_pres: np.ndarray,
               seed: int = 0, radius_px: float = 16.0,
               resolution=None, margin_radii: float = 1.0,
               gate_uv: np.ndarray | None = None) -> dict:
    """One object's tracked path vs a reference path, both in pixels.

    Two numbers, because they answer different questions:
      abs   raw distance. Carries a constant bias: the reference is the object
            ORIGIN, a mask centroid is the middle of the visible blob.
      disp  distance after subtracting each path's own position at the anchor
            frame. The bias cancels and displacement is exactly what an edit
            changes, so this is the one to report.

    Errors are also given in object radii (`*_radii`), which is the only way to
    compare a 16 px marble against a 40 px block.

    Pass `resolution` to drop frames the object is not fully inside the image
    on -- when the REFERENCE leaves the shot, there is nothing to be right or
    wrong about, so those frames go into `n_offscreen` / `n_clipped` instead
    of being scored.

    When the PREDICTION's tracker lost the object on a frame the reference is
    still visible on (`frames_lost`), the frame is charged with the reference's
    distance to the nearest image edge as its error: the model must have driven
    the object off-screen (or somewhere SAM2 could not follow it), and that is
    the tightest lower bound on the true positional error without knowing where
    the model actually put it. Without this, a prediction that flies the object
    off the border scores `disp_mean_px = 0` on the remaining frames and the
    failure only shows up in the raw `frames_lost` count.

    `gate_uv` is where the border test is applied, defaulting to ref_uv. When
    the reference is itself a tracked centroid rather than a projection, pass
    the projected path here: a centroid cannot say whether the object left the
    frame, because a tracker that has lost the object still reports one. The
    same path is used for the `frames_lost` edge-distance penalty.
    """
    k = min(len(tuv), len(ref_uv), len(ref_pres))
    scale = max(float(radius_px), 12.0)
    # Frames the object can be scored on at all: it exists, the reference
    # projects, we are at or past the seed, and it is fully in the image.
    ok = ref_pres[:k] & np.isfinite(ref_uv[:k, 0])
    ok[:seed] = False
    n_off = n_clip = 0
    guv_for_edge = None
    if resolution is not None:
        guv = ref_uv if gate_uv is None else gate_uv
        full, centre = in_frame(guv[:k], resolution, radius_px, margin_radii)
        n_off = int((ok & ~centre).sum())
        n_clip = int((ok & centre & ~full).sum())
        ok &= full
        guv_for_edge = guv[:k]
    tracked = ok & np.isfinite(tuv[:k, 0])
    lost = ok & ~np.isfinite(tuv[:k, 0])       # ref there, pred gave up
    n_lost = int(lost.sum())
    out = {"n_scored": int(tracked.sum()) + n_lost,
           "n_tracked": int(tracked.sum()),
           "frames_lost": n_lost,
           "n_offscreen": n_off, "n_clipped": n_clip,
           "seed": int(seed), "radius_px": round(float(radius_px), 1)}
    # Displacement is measured from the first frame that survives the gate, not
    # blindly from the seed: if the seed frame itself is clipped, every disp
    # value inherits the crop it introduces.
    tr_idx = np.flatnonzero(tracked)
    if tr_idx.size < 3:
        return {**out, "anchor": None, "abs_mean_px": None, "disp_mean_px": None,
                "disp_p95_px": None, "disp_final_px": None,
                "disp_mean_radii": None, "verdict": "UNMEASURABLE"}
    a = int(seed) if tracked[seed] else int(tr_idx[0])
    out["anchor"] = a

    # Per-frame errors filled by time index, then aggregated -- this keeps
    # `disp_final` correctly indexed to the last scored frame (tracked or
    # penalised) instead of the last element of a flat scored list.
    per_abs = np.full(k, np.nan)
    per_disp = np.full(k, np.nan)
    per_abs[tracked] = np.linalg.norm(tuv[:k][tracked] - ref_uv[:k][tracked],
                                      axis=1)
    per_disp[tracked] = np.linalg.norm(
        (tuv[:k][tracked] - tuv[a]) - (ref_uv[:k][tracked] - ref_uv[a]),
        axis=1)

    # Charge lost frames the reference-to-nearest-edge distance (lower bound
    # on how far off-screen the model must have driven the object).
    if n_lost and guv_for_edge is not None:
        W, H = resolution
        u = guv_for_edge[lost, 0]; v = guv_for_edge[lost, 1]
        edge_d = np.clip(np.minimum.reduce([u, W - u, v, H - v]), 0.0, None)
        per_disp[lost] = edge_d
        per_abs[lost] = edge_d

    scored = np.isfinite(per_disp)
    disp_e = per_disp[scored]
    abs_e = per_abs[scored]
    last_scored = int(np.flatnonzero(scored)[-1])
    dm = float(disp_e.mean())
    return {**out,
            "abs_mean_px": round(float(abs_e.mean()), 2),
            "disp_mean_px": round(dm, 2),
            "disp_p95_px": round(float(np.percentile(disp_e, 95)), 2),
            "disp_final_px": round(float(per_disp[last_scored]), 2),
            "disp_mean_radii": round(dm / scale, 3),
            "verdict": ("GOOD" if dm < scale * 0.5 else
                        "OK" if dm < scale else "UNUSABLE")}


# --------------------------------------------------------------------- target
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _numkey(s: str):
    m = re.findall(r"\d+", s)
    return (re.sub(r"\d+", "", _norm(s)), int(m[-1]) if m else None)


def removed_objects(objs: dict, eobjs: dict) -> list[str]:
    """Objects present in the source render and gone from the edited one."""
    gone = []
    for n, rec in objs.items():
        a = np.isfinite(rec["mw"]).all(axis=(1, 2))
        b = (np.isfinite(eobjs[n]["mw"]).all(axis=(1, 2))
             if n in eobjs else np.zeros(len(a), bool))
        if a.any() and not b.any():
            gone.append(n)
    return gone


def deletion_frame(objs: dict, eobjs: dict, name: str,
                   min_after: int = 3) -> int | None:
    """Frame from which `name` is gone in the edited render, or None.

    Scenes record a timed DELETE two different ways: some drop the body from
    the edited run entirely (`removed_objects` catches those, and this returns
    0 for them), others keep it and let its transforms go NaN from the edit
    frame on. The second kind reads as an ordinary object to a check that only
    asks "is it in the edited object list", so the removal goes unscored -- the
    frames it should be absent for are simply not scorable, and a model that
    deleted nothing is charged nothing.

    A deletion is: present somewhere in the edited run, absent from some frame
    onwards, and the SOURCE still showing it there -- the last condition is
    what separates a deletion from an object that merely leaves the scene in
    both runs.
    """
    if name not in objs:
        return None
    a = np.isfinite(objs[name]["mw"]).all(axis=(1, 2))
    if not a.any():
        return None
    if name not in eobjs:
        return 0
    b = np.isfinite(eobjs[name]["mw"]).all(axis=(1, 2))
    if not b.any():
        return 0
    d = int(np.flatnonzero(b)[-1]) + 1
    if d >= len(b) or b[d:].any():
        return None
    return d if int(a[d:min(len(a), len(b))].sum()) >= min_after else None


def added_objects(objs: dict, eobjs: dict) -> list[str]:
    """Objects the edited render has and the source does not.

    ADD's mirror of removed_objects. Unlike a removed object, an added one has
    a full trajectory to be scored against -- it is in the edited render for
    the tracker to follow -- so these are measurable in more ways than a
    DELETE, not fewer.
    """
    new = []
    for n, rec in eobjs.items():
        b = np.isfinite(rec["mw"]).all(axis=(1, 2))
        a = (np.isfinite(objs[n]["mw"]).all(axis=(1, 2))
             if n in objs else np.zeros(len(b), bool))
        if b.any() and not a.any():
            new.append(n)
    return new


def edit_seed(base_gt: dict, edit_gt: dict, obj: str, eobjs: dict | None = None):
    """First frame an ADDed object is on screen, read off the edited render.

    base_seed cannot be used for these: the object does not exist in the source
    at all. The camera still comes from the source clip (it is edit-invariant),
    so the seed is in the same pixel frame as every other target's.

    -> (frame index or None, uv at that frame, radius_px)
    """
    eobjs = eobjs if eobjs is not None else normalise(edit_gt)
    if obj not in eobjs:
        return None, None, np.nan
    vis, uv = visible_mask(edit_gt, obj, eobjs, camera_from=base_gt)
    _, _, erad = projected_track(edit_gt, obj, eobjs, camera_from=base_gt)
    r_px = float(np.nanmedian(erad)) if np.isfinite(erad).any() else 16.0
    idx = np.where(vis)[0]
    if idx.size == 0:
        return None, None, r_px
    f = int(idx[0])
    return f, uv[f], r_px


def placement_error(pred_uv, gt_uv, anchor_uv, other_uv, radius_px: float):
    """How far a predicted placement is from where the edit asked for it.

    An ADD prompt states a position as a distance in radii along the line
    between two objects, so the error is reported in radii too -- split into
    the component along that line ("too far down the line from the anchor")
    and the component across it, which the prompt never asked the model to
    vary.

    The radii here are the object's *apparent* radius in pixels on this frame,
    the same scale the rest of these metrics use, not the scene-space radius
    the DSL was written in. Perspective foreshortens the two by the same
    factor, so the along-line figure is close to the scene-space error for
    objects at similar depth, but it is not literally the DSL's number: the
    pool line measures 20.8 radii in the scene and 15.4 on screen. Use it to
    compare predictions against each other and against ground truth on one
    case, not as a reading of the instruction's own quantity.

    All four points are pixel coordinates on the same frame. Returns None when
    any of them is missing, which is the honest answer when the tracker did not
    find the object at all.
    """
    pts = (pred_uv, gt_uv, anchor_uv, other_uv)
    if any(p is None or not np.all(np.isfinite(p)) for p in pts):
        return None
    r = float(radius_px) if radius_px and np.isfinite(radius_px) and radius_px > 0 else 16.0
    axis = np.asarray(other_uv, float) - np.asarray(anchor_uv, float)
    n = float(np.linalg.norm(axis))
    if n < 1e-6:
        return None
    unit = axis / n
    d = np.asarray(pred_uv, float) - np.asarray(gt_uv, float)
    along = float(np.dot(d, unit))
    perp = float(d[0] * unit[1] - d[1] * unit[0])
    return {
        "err_px": round(float(np.linalg.norm(d)), 2),
        "err_radii": round(float(np.linalg.norm(d)) / r, 3),
        "along_radii": round(along / r, 3),
        "perp_radii": round(perp / r, 3),
        # The prompt is stated in radii along the line, so this is the number
        # that answers "did it put it where the instruction said".
        "line_len_radii": round(n / r, 2),
    }


def _group_names(meta: list, key: str) -> list[str]:
    return [(m.get("object_name") or f"{key}_{i}") if isinstance(m, dict)
            else f"{key}_{i}" for i, m in enumerate(meta)]


def _physics_map(gt: dict, names: list[str]) -> dict[str, list[str]]:
    """-> {physics.objects key: the render names it describes, in index order}.

    physics.objects and the rendered object list are written by different parts
    of the generator and do not have to agree. Four spellings occur:

      cue_ball:  {mass: ...}                 key is the render name
      dominoes:  {masses: [...], ...}        group of arrays, index -> name
      mallet_masses: [...]                   bare array, index -> name
      ball:      {mass: ...}                 ramp_collision's falling_marble,
                                             which physics calls something else
    """
    top = gt.get("objects", {}) or {}
    phys = (gt.get("physics", {}) or {}).get("objects", {}) or {}
    out: dict[str, list[str]] = {}
    used: set[str] = set()

    for k, v in phys.items():
        if k in names:
            out[k] = [k]
            used.add(k)
        elif isinstance(top.get(k), list):
            g = [n for n in _group_names(top[k], k) if n in names]
            if g:
                out[k] = g
                used.update(g)
        elif isinstance(top.get(k), dict) and top[k].get("object_name") in names:
            out[k] = [top[k]["object_name"]]
            used.add(out[k][0])
        elif isinstance(v, list) and len(v) == len(names):
            out[k] = list(names)          # bare parallel array, positional

    free = [n for n in names if n not in used]
    for k, v in phys.items():             # group of arrays under a different key
        if k in out or not isinstance(v, dict):
            continue
        n_arr = v.get("count") or max((len(x) for x in v.values()
                                       if isinstance(x, list)), default=0)
        cands = [kk for kk, meta in top.items()
                 if isinstance(meta, list) and len(meta) == n_arr
                 and all(n in free for n in _group_names(meta, kk))]
        if n_arr > 1 and len(cands) == 1:
            out[k] = _group_names(top[cands[0]], cands[0])
            free = [n for n in free if n not in out[k]]
    # Last resort, and only when it is unambiguous: one physics body with a
    # mass left over, one rendered object left over, so they are each other.
    rest = [k for k, v in phys.items()
            if k not in out and isinstance(v, dict) and "mass" in v]
    if len(rest) == 1 and len(free) == 1:
        out[rest[0]] = free
    return out


def physics_diff(base_gt: dict, edit_gt: dict, names: list[str]) -> dict[str, str]:
    """-> {object name: the sim parameter the edit changed}."""
    bo = (base_gt.get("physics", {}) or {}).get("objects", {}) or {}
    eo = (edit_gt.get("physics", {}) or {}).get("objects", {}) or {}
    pmap = _physics_map(base_gt, names)
    hits: dict[str, str] = {}
    for key, bv in bo.items():
        ev, tgt = eo.get(key), pmap.get(key)
        if ev is None or not tgt:
            continue
        if isinstance(bv, dict) and isinstance(ev, dict):
            if len(tgt) == 1:
                changed = [p for p in bv if bv[p] != ev.get(p, bv[p])]
                if changed:
                    hits[tgt[0]] = changed[0]
                continue
            for p, bl in bv.items():                  # arrays inside the group
                el = ev.get(p)
                if not (isinstance(bl, list) and isinstance(el, list)
                        and len(bl) == len(el) == len(tgt)):
                    continue
                for i, (x, y) in enumerate(zip(bl, el)):
                    if x != y:
                        hits[tgt[i]] = f"{key}.{p}"
        elif (isinstance(bv, list) and isinstance(ev, list)
              and len(bv) == len(ev) == len(tgt)):
            for i, (x, y) in enumerate(zip(bv, ev)):
                if x != y:
                    hits[tgt[i]] = key
    return hits


def resolve_target(base_gt: dict, edit_gt: dict, object_id: str | None = None):
    """-> (name, how). The object the edit acted on.

    What the ground truth says changed beats what the edit calls the object.
    The two disagree: domino_chain's edits address `domino_1 .. domino_4` while
    the render names them `domino_000 .. domino_003`, so matching on the number
    picks the domino next to the right one -- for every domino case, SET and
    DELETE alike. The presence and parameter diffs are read off the two GT
    files and cannot drift from the render, so they go first, and the name is
    consulted only when they leave more than one candidate.
    """
    objs = normalise(base_gt)
    eobjs = normalise(edit_gt)
    names = list(objs)

    gone = removed_objects(objs, eobjs)                    # DELETE
    if len(gone) == 1:
        return gone[0], "presence diff"
    hits = physics_diff(base_gt, edit_gt, names)           # SET
    if len(hits) == 1:
        n, key = next(iter(hits.items()))
        return n, f"physics diff ({key})"

    if object_id:
        for n in names:                                    # exact
            if _norm(n) == _norm(object_id):
                return n, "object_id exact"
        ob, oi = _numkey(object_id)
        for n in names:                                    # same stem + index
            nb, ni = _numkey(n)
            if oi is not None and ni == oi and (ob in nb or nb in ob):
                return n, "object_id numeric"
        cands = [n for n in names if _norm(object_id) in _norm(n)
                 or _norm(n) in _norm(object_id)]
        if len(cands) == 1:
            return cands[0], "object_id substring"

    best, bestd = None, -1.0                               # largest divergence
    for n in names:
        if n not in eobjs:
            continue
        a, pa, _ = projected_track(base_gt, n, objs)
        b, pb, _ = projected_track(edit_gt, n, eobjs, camera_from=base_gt)
        k = min(len(a), len(b))
        m = pa[:k] & pb[:k]
        if not m.any():
            continue
        d = float(np.max(np.linalg.norm(a[:k][m] - b[:k][m], axis=1)))
        if d > bestd:
            best, bestd = n, d
    return best, "max divergence"


def divergence(base_gt: dict, edit_gt: dict, obj: str,
               bobjs: dict | None = None, eobjs: dict | None = None) -> float:
    """Peak pixel gap between the source and edited trajectories of one object.

    Both are viewed through the SOURCE camera, so a scene that re-randomises
    the viewpoint per render (ball_block) reports the physics change, not the
    viewpoint change. NaN when the two never share a visible frame.
    """
    bobjs = bobjs if bobjs is not None else normalise(base_gt)
    eobjs = eobjs if eobjs is not None else normalise(edit_gt)
    if obj not in bobjs or obj not in eobjs:
        return float("nan")
    a, pa, _ = projected_track(base_gt, obj, bobjs)
    b, pb, _ = projected_track(edit_gt, obj, eobjs, camera_from=base_gt)
    k = min(len(a), len(b))
    m = pa[:k] & pb[:k]
    if not m.any():
        return float("nan")
    return float(np.max(np.linalg.norm(a[:k][m] - b[:k][m], axis=1)))


def resolve_targets(base_gt: dict, edit_gt: dict, object_id: str | None = None,
                    min_visible: int = 3, moved_tol_radii: float = 0.25):
    """Every object worth measuring in this case, not just the edited one.

    An edit names one object but the whole point of the benchmark is the chain
    it sets off: heavier mallet_1 changes where mallet_0 and mallet_2 end up,
    and a model that moves only the named object has not understood the edit.
    Scoring one object per case throws that signal away.

    -> list of dicts, edited object first, then the objects the edit actually
       displaces (largest first), then the ones it leaves alone:

       name          object key in normalise(base_gt)
       role          "edited" | "affected" | "static"
       how           why it got the "edited" role ("" for the others)
       divergence_px peak source-vs-edited gap, the size of the answer
       seed          frame to start tracking at, or None
       reason        "ok", or why it is unmeasurable
       sep_px        source-vs-edited gap at the seed frame
       radius_px     apparent radius, the natural error scale
       n_visible     frames the object projects inside the image
    """
    bobjs = normalise(base_gt)
    eobjs = normalise(edit_gt)
    primary, how = resolve_target(base_gt, edit_gt, object_id)
    gone = set(removed_objects(bobjs, eobjs))
    arrived = set(added_objects(bobjs, eobjs))
    # "Make the stones bouncy" edits every stone; there is no single edited
    # object, and calling the other one merely "affected" would be a lie.
    edited = gone | arrived | set(physics_diff(base_gt, edit_gt, list(bobjs)))
    edited.add(primary)

    rows = []
    # An ADDed object is not in the source, so iterating the source alone would
    # miss the very thing the edit is about.
    for name in list(bobjs) + [n for n in eobjs if n not in bobjs]:
        if name not in bobjs and name not in arrived:
            # In the edited render's object list but never actually on screen
            # there either: nothing to measure, and the source-side code below
            # would have no trajectory to read.
            continue
        if name in arrived:
            f, uv0, r_px = edit_seed(base_gt, edit_gt, name, eobjs)
            vis_e, _ = visible_mask(edit_gt, name, eobjs, camera_from=base_gt)
            n_vis = int(vis_e.sum())
            rows.append({
                "name": name, "role": "edited",
                "how": how if name == primary else "added by the edit",
                # There is no source trajectory to diverge from; the whole
                # object is the divergence.
                "divergence_px": None,
                "deleted": False, "added": True,
                "seed": f if n_vis >= min_visible else None,
                "reason": "ok (added by the edit)" if n_vis >= min_visible
                          else f"visible in only {n_vis} frames",
                "sep_px": None,
                "radius_px": round(float(r_px), 1) if np.isfinite(r_px) else 16.0,
                "n_visible": n_vis,
            })
            continue
        vis, _ = visible_mask(base_gt, name, bobjs)
        n_vis = int(vis.sum())
        div = divergence(base_gt, edit_gt, name, bobjs, eobjs)
        f, sep, r_px, reason = seed_frame(base_gt, edit_gt, name,
                                          bobjs=bobjs, eobjs=eobjs)
        if n_vis < min_visible:
            f, reason = None, f"visible in only {n_vis} frames"
        # A DELETE case removes the object outright: there is no edited
        # trajectory to compare, but "did it disappear" is still measurable.
        del_from = 0 if name in gone else deletion_frame(bobjs, eobjs, name)
        deleted = del_from is not None
        if name in edited:
            role = "edited"
        elif deleted or (np.isfinite(div) and div > moved_tol_radii * max(r_px, 12.0)):
            role = "affected"
        else:
            role = "static"
        rows.append({"name": name, "role": role, "added": False,
                     "how": how if name == primary else "",
                     "divergence_px": None if not np.isfinite(div) else round(div, 2),
                     "deleted": bool(deleted),
                     "delete_from_frame": del_from,
                     "seed": f, "reason": reason,
                     "sep_px": None if not np.isfinite(sep) else round(float(sep), 2),
                     "radius_px": round(float(r_px), 1) if np.isfinite(r_px) else 16.0,
                     "n_visible": n_vis})

    order = {"edited": 0, "affected": 1, "static": 2}
    rows.sort(key=lambda r: (order[r["role"]], -(r["divergence_px"] or 0.0), r["name"]))
    return rows


# ----------------------------------------------------------------- seed frame
def visible_mask(gt: dict, obj: str, objs: dict | None = None,
                 camera_from: dict | None = None):
    """Frames where the object projects inside the image."""
    uv, pres, _ = projected_track(gt, obj, objs, camera_from=camera_from)
    cam_src = camera_from if camera_from is not None else gt
    W, H = cam_src["camera"]["resolution"]
    return (pres & (uv[:, 0] >= 0) & (uv[:, 0] < W)
            & (uv[:, 1] >= 0) & (uv[:, 1] < H)), uv


def base_seed(base_gt: dict, obj: str, objs: dict | None = None):
    """First frame the object is on screen in the SOURCE clip, and its scale.

    -> (frame index or None, uv at that frame, radius_px)

    This is the seed the tracker gets. It comes from the source alone, so it is
    identical for every case in a scene and leaks nothing about the answer.
    """
    objs = objs if objs is not None else normalise(base_gt)
    if obj not in objs:
        return None, None, np.nan
    vis, uv = visible_mask(base_gt, obj, objs)
    _, _, brad = projected_track(base_gt, obj, objs)
    r_px = float(np.nanmedian(brad)) if np.isfinite(brad).any() else 16.0
    idx = np.where(vis)[0]
    if idx.size == 0:
        return None, None, r_px
    f = int(idx[0])
    return f, uv[f], r_px


def seed_frame(base_gt: dict, edit_gt: dict, obj: str,
               bobjs: dict | None = None, eobjs: dict | None = None,
               tol_radii: float = None):  # kept for signature compat; unused
    """Where to start tracking. The source's first-visible frame -- edit-
    invariant, identical for every case in a scene, so nothing about the
    edited answer reaches the tracker.

    Legacy versions rejected cases where the source and edited projections
    diverged at the seed frame (initial_velocity edits like tennis_flight's
    hard/soft serve). That was a hard requirement of the point-seed tracker:
    seeding at the source (u,v) on the edited video landed on background.
    GroundedSAM2 seeds via text, not (u,v), so the divergence check is
    obsolete -- the tracker finds the object wherever it actually is.

    -> (frame index or None, separation_px_at_seed, radius_px, reason)
    """
    bobjs = bobjs if bobjs is not None else normalise(base_gt)
    eobjs = eobjs if eobjs is not None else normalise(edit_gt)
    f, buv_f, r_px = base_seed(base_gt, obj, bobjs)
    if obj not in bobjs:
        return None, np.nan, r_px, "object missing from source"
    if f is None:
        return None, np.nan, r_px, "never visible in the source"
    if obj not in eobjs:
        # DELETE cases: the object is absent from the edited render. Still
        # measurable (presence check) so we return the source seed; the
        # separation is not meaningful here.
        return f, np.nan, r_px, "ok (deleted in edit)"
    ve, euv = visible_mask(edit_gt, obj, eobjs, camera_from=base_gt)
    sep = float(np.linalg.norm(buv_f - euv[f])) if f < len(euv) and ve[f] \
        else float("nan")
    return f, sep, r_px, "ok"


def load_gt(p) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))
