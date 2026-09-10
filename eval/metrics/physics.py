"""Physics-consistency metric: does the prediction's motion match the edit?

The reference is the EDITED render -- that is what the model was asked to
produce. Both sides go through the same tracker: GroundedSAM2 follows each
object in the edited ground-truth video and in the prediction, from the same
text prompt, and the two pixel trajectories are compared.

Tracking both sides rather than comparing against the projected ground truth
directly is what makes the comparison fair for objects that rotate. The ground
truth stores an object's ORIGIN while a mask gives its CENTROID, and for a
domino or a bowling pin that gap swings as the object topples -- on the clean
render that shows up as 8-18 px of "error" with the tracker working perfectly.
Centroid against centroid, it cancels.

Text prompts replace projected seed points as the tracker anchor. A short
English phrase per GT object (metrics/text_prompts/{scene}.json) is fed to
GroundingDINO on the object's first-visible frame; the returned box is what
SAM2 propagates from. For visually distinct objects the phrase disambiguates on
its own; for visually identical groups (bowling pins, dominoes) every member
gets the same phrase and the returned boxes are assigned to GT identities by
nearest-neighbour against the projected first-frame origin, which is
edit-invariant and identical for every case in a scene.

The projection is still needed, for the things text and centroids cannot:

  identity     match same-text boxes to GT names (edited answer never touches
               this because the source projection is what is used)
  border gate  whether the object is really in frame -- a tracker that has lost
               an object still returns a centroid
  scale        the object's apparent radius, so errors can be read in radii
  timing       the frame the reference leaves the shot

and it also gives a second, independent number: `abs_*_vs_projection`, the
absolute placement error. Centroid-against-centroid cancels a constant offset,
so it cannot see an object that moves correctly in the wrong place; the
projection can.

DELETE cases have no reference trajectory for the removed object; what is
measurable there is whether it is gone, so the prediction's mask is compared
against the edited video's response to the same seed.

ADD cases are the mirror, and richer: the object the edit calls for is in the
edited render, so it has a full reference trajectory like any other target and
is scored the same way. On top of that it gets a placement score, because an
ADD prompt does not ask for a trajectory -- it asks for a position, stated as
a distance in radii along the line between two objects. `add_*` on the target
row reports that error in the prompt's own unit, split into the component
along the line (the distance the instruction named) and the component across
it (which it did not).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from .grounded_sam2_tracker import (MASK_H, MASK_W,
                                    GroundedSam2Tracker)

MASK_PIXELS = MASK_W * MASK_H
from .traj_lib import (base_seed, edit_seed, in_frame, load_gt, normalise,
                       placement_error, project_point, projected_track,
                       resolve_targets, traj_error)


@dataclass
class PhysicsScores:
    """Per-case result. `objects` holds one entry per object in the scene."""
    status: str
    objects: dict[str, dict[str, Any]] = field(default_factory=dict)
    disp_edited_px: float | None = None
    disp_affected_px: float | None = None
    disp_all_px: float | None = None
    disp_all_radii: float | None = None
    abs_all_px: float | None = None
    null_disp_px: float | None = None
    gap_closed: float | None = None
    # Removal, the mirror of the trajectory metrics: `removal_gap` is scaled
    # exactly like `gap_closed` -- 0 is what a model scores by ignoring the
    # instruction and leaving the object in, 1 is a clean removal -- and
    # `onset_err_frames` is signed, negative meaning the model took the object
    # away before the edit asked it to.
    removal_gap: float | None = None
    onset_err_frames: float | None = None
    mask_iou: float | None = None
    n_removals: int = 0
    n_measured: int = 0
    n_objects: int = 0
    note: str = ""


def load_ground_truth(gt_path: Path) -> dict:
    """Return the parsed ground_truth_transforms.json."""
    return json.loads(gt_path.read_text(encoding="utf-8"))


def _looks_present(p_uv: np.ndarray, p_area: np.ndarray,
                   s_uv: np.ndarray, s_area: np.ndarray,
                   radius_px: float, *, margin_radii: float = 2.0,
                   area_lo: float = 0.3, area_hi: float = 3.0) -> np.ndarray:
    """Per-frame: does the prediction still show the object?

    The source clip is the reference for what "still there" looks like -- it is
    the same tracker on the same scene with the object present, so its centroid
    says where the object would be and its mask area says how big it should
    look. Both gates are needed. Area alone is not enough because a mask of the
    right size can sit anywhere, and -- the reason this is not simply
    `p_area > 0` -- when the object IS gone GDINO finds nothing and SAM2 falls
    back onto whatever is behind it: a removed mallet of ~3200 px came back
    with a 107815 px mask, which any "is there a mask?" test reads as a failed
    removal. The upper area bound is what rejects those.

    Frames where the source itself has no mask are left False: with nothing to
    compare against, "still present" cannot be decided, and the removal window
    is built from the source's own presence so those frames are excluded there
    too.
    """
    k = min(len(p_uv), len(p_area), len(s_uv), len(s_area))
    out = np.zeros(k, bool)
    scale = max(float(radius_px), 12.0)
    for t in range(k):
        if not (s_area[t] > 0 and p_area[t] > 0):
            continue
        if not (np.isfinite(p_uv[t]).all() and np.isfinite(s_uv[t]).all()):
            continue
        ratio = float(p_area[t]) / float(s_area[t])
        if not (area_lo <= ratio <= area_hi):
            continue
        if float(np.linalg.norm(p_uv[t] - s_uv[t])) > margin_radii * scale:
            continue
        out[t] = True
    return out


def _plausible(area: np.ndarray, scale: float,
               lo: float = 0.3, hi: float = 3.0) -> np.ndarray:
    """Per-frame: is this mask plausibly the object rather than background?

    Area only, deliberately. A position test would need somewhere to measure
    from, and the only reference available for every object is the source
    clip -- but an edited object is SUPPOSED to be somewhere else, so gating on
    distance-from-source would throw away exactly the masks the metric exists
    to compare. Size travels better: an object keeps its apparent area whatever
    the edit does to its path.

    `scale` is the object's own median area in the source clip. When the object
    is gone GDINO finds nothing and SAM2 falls back onto the background -- a
    removed mallet of ~3200 px came back as 107815 px, 33x its size, and the
    upper bound is what rejects that.
    """
    if not scale:
        return area > 0
    return (area >= lo * scale) & (area <= hi * scale)


def mask_iou(pred_packed: np.ndarray, ref_packed: np.ndarray, n_pixels: int,
             pred_ok: np.ndarray | None = None,
             ref_ok: np.ndarray | None = None) -> tuple[float | None, int]:
    """Mean per-frame spatial IoU between two packed mask sequences.

    A side counts as showing nothing on a frame when its mask is empty OR when
    `_plausible` rejects it. The gate has to be applied to BOTH sides or the
    comparison is not symmetric: on a DELETE the reference video has no object
    either, so SAM2 grabs background there too, and the two background blobs
    sit in the same place and overlap almost perfectly. Ungated, a correctly
    removed object scored IoU 0.98 -- the tracker agreeing with itself about a
    failure, read as agreement about the edit.

    Frames where BOTH sides show nothing are not scored: for a whole-clip
    DELETE done right that is every frame, and averaging a 1.0 in there would
    reward a frame on which nothing was compared. Whether the removal happened
    is `removal_gap`'s question. Exactly one side showing something scores 0 --
    total disagreement, not an undefined comparison.

    -> (mean IoU over scored frames or None, number of scored frames)
    """
    T = min(len(pred_packed), len(ref_packed))
    vals = []
    for t in range(T):
        pa = True if pred_ok is None else bool(pred_ok[t])
        rb = True if ref_ok is None else bool(ref_ok[t])
        a = (np.unpackbits(pred_packed[t], count=n_pixels).astype(bool)
             if pa else None)
        b = (np.unpackbits(ref_packed[t], count=n_pixels).astype(bool)
             if rb else None)
        sa = int(a.sum()) if a is not None else 0
        sb = int(b.sum()) if b is not None else 0
        if not sa and not sb:
            continue
        if not sa or not sb:
            vals.append(0.0)
            continue
        inter = int(np.count_nonzero(a & b))
        union = int(np.count_nonzero(a | b))
        vals.append(inter / union)
    return (float(np.mean(vals)) if vals else None), len(vals)


def _sustained_false(mask: np.ndarray, start: int, run: int = 3) -> int | None:
    """First index at or after `start` where `mask` stays False for `run`.

    A single dropped frame is a tracker hiccup, not a removal, and the onset is
    meant to be the frame the object goes away and stays away.
    """
    n = len(mask)
    for t in range(max(int(start), 0), n):
        if not mask[t] and not mask[t:t + run].any():
            return t
    return None


def _first_offscreen(uv: np.ndarray, pres: np.ndarray, resolution) -> int | None:
    """Frame the reference leaves the image for good, or None."""
    W, H = resolution
    inside = pres & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    if inside.all() or not inside.any():
        return None
    last = int(np.flatnonzero(inside)[-1])
    return last + 1 if last + 1 < len(inside) else None


def _prompt_key(prompts: dict) -> str:
    """Cache key for a case's reference track. Covers the text, anchor frame,
    and (when set) the positional hint used to disambiguate identical-text
    groups, so a change to any of them misses the cache instead of silently
    reusing the wrong track."""
    blob = json.dumps({k: [v["text"], int(v["anchor_frame"]),
                           [round(float(x), 2) for x in (v.get("hint_uv") or [])]]
                       for k, v in sorted(prompts.items())}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def _load_text_prompts(scene: str, prompts_dir: Path) -> dict[str, str]:
    """Return {gt_object_name: english_phrase} for a scene, or {} if missing."""
    p = prompts_dir / f"{scene}.json"
    if not p.exists():
        return {}
    doc = json.loads(p.read_text(encoding="utf-8"))
    return doc.get("objects", {})


class PhysicsScorer:
    """Holds GroundedSAM2 (loaded once), the per-scene text prompts, and the
    GT track cache."""

    def __init__(self, benchmark_root: Path,
                 text_prompts_dir: Path | None = None,
                 device: str | None = None, margin_radii: float = 1.0,
                 min_frames: int = 3,
                 cache_dir: Path | None = None):
        self.root = Path(benchmark_root)
        self.tracker = GroundedSam2Tracker(device=device)
        self.margin_radii = margin_radii
        self.min_frames = min_frames
        self.prompts_dir = Path(text_prompts_dir
                                or (Path(__file__).parent / "text_prompts"))
        self.cache_dir = Path(cache_dir
                              or (Path(__file__).parent / "gt_tracks_gsam2"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._gt_cache: dict[str, dict] = {}
        self._prompt_cache: dict[str, dict[str, str]] = {}

    # ------------------------------------------------------------------ utils
    def _gt(self, rel: Path | str) -> dict:
        key = str(rel)
        if key not in self._gt_cache:
            self._gt_cache[key] = load_gt(rel)
        return self._gt_cache[key]

    def _scene_texts(self, scene: str) -> dict[str, str]:
        if scene not in self._prompt_cache:
            self._prompt_cache[scene] = _load_text_prompts(scene, self.prompts_dir)
        return self._prompt_cache[scene]

    @staticmethod
    def _read(path: Path) -> np.ndarray:
        import imageio.v3 as iio
        return iio.imread(str(path), plugin="pyav")

    def _cached_track(self, video: Path, tag: str,
                      prompts: dict) -> dict[str, dict]:
        """GroundedSAM2 on a benchmark video, cached on disk.

        Neither the edited render nor the source depends on the prediction, so
        every baseline after the first reuses both.
        """
        # v2: the cache now carries the packed masks the spatial IoU needs, so
        # entries written before that are not reusable -- a new suffix retires
        # them instead of silently returning tracks with no mask.
        cache = self.cache_dir / f"{tag}__{_prompt_key(prompts)}.v2.npz"
        if cache.exists():
            z = np.load(cache)
            return {n: {"uv": z[f"{n}__uv"], "area": z[f"{n}__area"],
                        "mask": z[f"{n}__mask"]}
                    for n in prompts}
        tr = self.tracker.track(self._read(video), prompts)
        # Atomic: the baselines are scored in parallel and share this cache, so
        # several processes can rebuild the same entry at once. Writing the
        # target path directly lets one process read a half-written npz.
        tmp = cache.with_suffix(f".{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, **{f"{n}__{k}": v
                                    for n, d in tr.items()
                                    for k, v in d.items()
                                    if k in ("uv", "area", "mask")})
        os.replace(tmp, cache)
        return tr

    def _reference_track(self, edit: dict, prompts: dict) -> dict[str, dict]:
        return self._cached_track(self.root / edit["edited_video"],
                                  f"{edit['scene']}__{edit['case_id']}", prompts)

    def _source_track(self, edit: dict, prompts: dict) -> dict[str, dict]:
        """GroundedSAM2 on the SOURCE clip -- the "model did nothing" baseline.

        It has to be tracked, not projected. The baseline is compared against
        the model's score, and the model's score is centroid against centroid;
        a projected baseline would carry the origin-vs-centroid swing that a
        toppling pin or domino produces and the model's number does not, making
        the two incomparable. One track per source clip, shared by every case
        in the scene.
        """
        return self._cached_track(
            self.root / "scenes" / edit["scene"] / "cases" /
            edit["source_case_id"] / "video.mp4",
            f"{edit['scene']}__SRC__{edit['source_case_id']}", prompts)

    # ------------------------------------------------------------------- plan
    def _plan(self, edit: dict):
        """What to track for a case, and where to start it.

        Depends on the two ground-truth files, the scene's text_prompts.json,
        and the source-clip camera only -- never on a prediction -- which is
        what lets the reference and source tracks be precomputed once and
        shared by every baseline.

        -> (base_gt, edit_gt, bobjs, eobjs, resolution, prompts, plan) or None.
        """
        gt_path = self.root / edit["ground_truth"]
        base_path = (self.root / "scenes" / edit["scene"] / "cases" /
                     edit["source_case_id"] / "ground_truth_transforms.json")
        edited_video = self.root / edit["edited_video"]
        if not (gt_path.exists() and base_path.exists() and edited_video.exists()):
            return None
        base_gt, edit_gt = self._gt(base_path), self._gt(gt_path)
        bobjs, eobjs = normalise(base_gt), normalise(edit_gt)
        resolution = base_gt["camera"]["resolution"]
        targets = resolve_targets(base_gt, edit_gt, edit.get("object_id"))
        texts = self._scene_texts(edit["scene"])

        prompts: dict[str, dict] = {}
        plan: dict[str, dict[str, Any]] = {}
        for t in targets:
            n = t["name"]
            row: dict[str, Any] = {
                "role": t["role"], "deleted": t["deleted"],
                "delete_from_frame": t.get("delete_from_frame"),
                "added": bool(t.get("added")),
                "divergence_px": t["divergence_px"],
            }
            if t.get("added"):
                # No source trajectory to seed from -- the object does not
                # exist there. The camera is the source's either way, so the
                # seed lands in the same pixel frame as every other target's.
                f_src, uv0, r_px = edit_seed(base_gt, edit_gt, n, eobjs)
            else:
                f_src, uv0, r_px = base_seed(base_gt, n, bobjs)
            row["radius_px"] = round(float(r_px), 1) if np.isfinite(r_px) else None
            text = texts.get(n)
            if f_src is None or t["n_visible"] < self.min_frames:
                row["status"] = "not_visible_in_source"
            elif not text:
                row["status"] = "no_text_prompt"
                row["reason"] = f"metrics/text_prompts/{edit['scene']}.json " \
                                f"has no entry for '{n}'"
            elif t["deleted"]:
                row["status"] = "deleted"           # presence check, below
            elif t.get("added"):
                row["status"] = "added"             # trajectory + placement
            elif t["seed"] is None:
                row["status"] = "unmeasurable"
                row["reason"] = t["reason"]
            else:
                row["status"] = "pending"
            if row["status"] in ("pending", "deleted", "added"):
                prompts[n] = {
                    "text": text,
                    "anchor_frame": int(f_src),
                    "hint_uv": (float(uv0[0]), float(uv0[1])),
                    "hint_radius_px": float(r_px) if np.isfinite(r_px) else None,
                }
                row["text"] = text
                row["anchor_frame"] = int(f_src)
            plan[n] = row
        return base_gt, edit_gt, bobjs, eobjs, resolution, prompts, plan

    @staticmethod
    def _add_position(edit: dict, obj: str) -> dict | None:
        """The relative position an ADD edit asked for, off the manifest.

        build_pcve_* records both halves in physics_diff: the instruction as
        written (which two objects name the line, which one the distance is
        measured from, how many radii) plus the centre it resolves to and the
        endpoints' coordinates.

        Not looked up by `obj`: physics_diff is keyed by the DSL's object id
        (`blue_stone`) while the render calls the same thing something else
        (`stone_2`), and the two vocabularies do not map onto each other by
        name. An ADD edit adds exactly one object, so the single entry
        carrying a `position` block is the one, whatever it is called.
        """
        for value in (edit.get("physics_diff") or {}).values():
            if isinstance(value, dict) and isinstance(value.get("position"), dict):
                return value["position"]
        return None

    @staticmethod
    def _gt_uv(edit_gt: dict, base_gt: dict, eobjs: dict,
               name: str | None, frame: int):
        """Where an object really is on one frame, in source-camera pixels.

        None when the object is not in the edited render or is off screen on
        that frame, which the caller treats as "not measurable" rather than
        as an error.
        """
        if not name or name not in eobjs:
            return None
        uv, pres, _ = projected_track(edit_gt, name, eobjs, camera_from=base_gt)
        if frame >= len(uv) or not pres[frame]:
            return None
        return uv[frame]

    def precompute(self, edit: dict) -> dict[str, Any]:
        """Track the edited render and the source clip, and cache both.

        Neither depends on a prediction, so this can run over the whole
        benchmark once -- including cases no baseline has produced a video for
        -- and every evaluation afterwards is one GroundedSAM2 pass per case
        instead of three. The cache is also shippable: someone scoring their
        own model never has to re-track the ground truth at all.
        """
        got = self._plan(edit)
        if got is None:
            return {"status": "no_ground_truth", "n_objects": 0}
        *_, prompts, plan = got
        if not prompts:
            return {"status": "nothing_to_track", "n_objects": len(plan)}
        self._reference_track(edit, prompts)
        self._source_track(edit, prompts)
        return {"status": "cached", "n_objects": len(plan),
                "n_tracked": len(prompts),
                "objects": sorted(prompts)}

    # ------------------------------------------------------------------ score
    def score(self, pred_video: Path, edit: dict,
              frames: np.ndarray | None = None) -> PhysicsScores:
        got = self._plan(edit)
        if got is None:
            return PhysicsScores(status="no_ground_truth",
                                 note=f"missing ground truth for {edit['global_id']}")
        base_gt, edit_gt, bobjs, eobjs, resolution, prompts, plan = got

        if not prompts:
            return PhysicsScores(
                status="nothing_measurable", objects=plan, n_objects=len(plan),
                note="no object could be seeded (missing text prompts or "
                     "edit already displaced the object at anchor frame)")

        # ---- track both sides from the same prompts ------------------------
        try:
            # The reference side is not guessed, it is known. Ground truth says
            # exactly which frames hold the object, so a DELETEd one is not put
            # to GroundedSAM2 at all: asked for something that is not there, it
            # answers with the scene's biggest lookalike (a removed domino came
            # back as 655164 px, the whole table) and the prediction side --
            # which fails the same way, on the same table -- then "agrees" with
            # it. The prediction gets no such help: nothing is known about what
            # a model did, so everything is tracked there.
            T_ref = len(edit_gt["frames"])
            ref_prompts = {n: q for n, q in prompts.items()
                           if plan.get(n, {}).get("delete_from_frame") != 0}
            ref_tracks = self._reference_track(edit, ref_prompts)
            for n in prompts:
                d = plan.get(n, {}).get("delete_from_frame")
                if n not in ref_tracks:
                    ref_tracks[n] = {
                        "uv": np.full((T_ref, 2), np.nan),
                        "area": np.zeros(T_ref),
                        "mask": np.zeros((T_ref, MASK_PIXELS // 8), np.uint8)}
                elif d:
                    # Timed delete: present up to the edit frame, known absent
                    # after it. Only the tail is blanked.
                    ref_tracks[n] = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                                     for k, v in ref_tracks[n].items()}
                    ref_tracks[n]["uv"][d:] = np.nan
                    ref_tracks[n]["area"][d:] = 0
                    ref_tracks[n]["mask"][d:] = 0
            # The source clip is tracked to build the null -- "what a model
            # scores by ignoring the prompt and reproducing the source" -- so
            # an ADDed object has no business in that pass: it does not exist
            # there, the null is set to None for it anyway, and asking
            # GroundedSAM2 for it anyway costs a propagation and returns
            # whatever the scene's biggest lookalike is (a green mallet prompt
            # came back with the whole air hockey table, 414656 px).
            src_prompts = {n: q for n, q in prompts.items()
                           if not plan.get(n, {}).get("added")}
            src_tracks = self._source_track(edit, src_prompts)
            if frames is None:
                frames = self._read(pred_video)
            pred_tracks = self.tracker.track(frames, prompts)
            # The tracker reports centroids and mask areas in the pixel grid of
            # the video it was given, and baselines do not render at the
            # benchmark's resolution -- VOID is 672x384, ditto 832x480, the
            # source 1280x720. Every comparison below is against the source's
            # grid (the reference tracks, the projected paths, the radii), so
            # the prediction has to be expressed in the same units first. This
            # is a change of units, not a resize: re-tracking an upscaled copy
            # would measure a different video. perceptual.py already resizes
            # for the same reason.
            ph, pw = frames.shape[1:3]
            W, H = resolution
            if (pw, ph) != (int(W), int(H)):
                sx, sy = float(W) / pw, float(H) / ph
                for d in pred_tracks.values():
                    d["uv"] = d["uv"] * np.array([sx, sy])
                    d["area"] = d["area"] * (sx * sy)
        except Exception as exc:                            # noqa: BLE001
            return PhysicsScores(status="tracker_failed", objects=plan,
                                 n_objects=len(plan), note=str(exc))

        # ---- score ----------------------------------------------------------
        for n, row in plan.items():
            if row["status"] not in ("pending", "deleted", "added"):
                continue
            f = int(row["anchor_frame"])
            r_px = row["radius_px"] or 16.0
            p_uv, p_area = pred_tracks[n]["uv"], pred_tracks[n]["area"]
            g_uv, g_area = ref_tracks[n]["uv"], ref_tracks[n]["area"]
            p_mask, g_mask = pred_tracks[n]["mask"], ref_tracks[n]["mask"]
            k = min(len(p_uv), len(g_uv))
            row["mask_area_ratio"] = (round(float(p_area[f]) / float(g_area[f]), 3)
                                      if g_area[f] else None)

            if row["status"] == "deleted":
                # No reference trajectory: after the edit frame the object does
                # not exist, so there is no position to be wrong about and
                # traj_error has nothing to score. What can be wrong is whether
                # it is on screen at all, and from which frame -- so presence
                # over time is the reference signal here, in place of a path.
                #
                # traj_lib.traj_error already handles the opposite asymmetry
                # (the reference still shows an object the prediction lost, and
                # the frame is charged the reference's distance to the nearest
                # edge). This is the mirror image of that: the reference says
                # gone, the prediction still shows it.
                s_uv, s_area = src_tracks[n]["uv"], src_tracks[n]["area"]
                kk = min(k, len(s_uv))
                pred_present = _looks_present(p_uv[:kk], p_area[:kk],
                                              s_uv[:kk], s_area[:kk], r_px,
                                              margin_radii=self.margin_radii * 2.0)
                # Where the reference says the object should be visible. A
                # whole-clip DELETE is not in the edited sim at all -- there is
                # no record to read a presence flag off -- and that absence is
                # itself the answer: gone on every frame. A timed DELETE is
                # still in the edited sim, present up to its edit frame, so its
                # own flags carry the schedule the metric has to check.
                if n in eobjs:
                    del_uv, del_pres, _ = projected_track(edit_gt, n, eobjs,
                                                          camera_from=base_gt)
                    kk = min(kk, len(del_pres))
                    pred_present = pred_present[:kk]
                    gt_present = del_pres[:kk] & np.isfinite(del_uv[:kk, 0])
                else:
                    gt_present = np.zeros(kk, bool)
                # Only frames the removal actually asks about: the object is
                # gone in the reference and the source would still be showing
                # it. For a whole-clip DELETE that is the whole clip; for
                # `AT FRAME n` it starts at n, which is what makes the timed
                # cases scoreable at all.
                window = (~gt_present) & (s_area[:kk] > 0)
                n_window = int(window.sum())
                wrong = int((pred_present & window).sum())
                row["removal_window_frames"] = n_window
                row["wrong_present_frames"] = wrong
                # Same shape as gap_closed: the null is "ignore the prompt and
                # leave it in", which is wrong on every frame of the window.
                row["removal_gap"] = (round(1.0 - wrong / n_window, 3)
                                      if n_window else None)
                scale = float(np.median(s_area[s_area > 0])) if (s_area > 0).any() else 0.0
                iou, n_iou = mask_iou(
                    p_mask, g_mask, MASK_PIXELS,
                    _plausible(p_area, scale), _plausible(g_area, scale))
                row["mask_iou"] = round(iou, 3) if iou is not None else None
                row["mask_iou_frames"] = n_iou
                row["mask_area_scale"] = round(scale, 1)

                # --- positional cost of an object that is still there -------
                # traj_lib charges the opposite asymmetry (reference present,
                # prediction lost) with the reference's distance to the nearest
                # image edge, on the reasoning that the model must have driven
                # the object at least that far. This is the mirror of it: the
                # reference says gone, the prediction still shows it at p, and
                # the smallest change that would have made it disappear is
                # pushing it off the frame -- so dist(p, edge) is the tightest
                # lower bound on the positional error without knowing what the
                # model should have painted there instead.
                #
                # It goes on BOTH sides. The null baseline ("ignore the prompt
                # and reproduce the source") keeps the object too, so it is
                # charged the same way, and null_disp stops collapsing to the
                # tracker noise of the bystanders -- which is what made
                # gap_closed divide by 1.46 px on mahjong_dice.
                W, H = float(resolution[0]), float(resolution[1])

                def _edge_dist(uv: np.ndarray) -> np.ndarray:
                    x, y = uv[:, 0], uv[:, 1]
                    d = np.minimum(np.minimum(x, W - 1.0 - x),
                                   np.minimum(y, H - 1.0 - y))
                    return np.clip(d, 0.0, None)

                if n_window:
                    pred_cost = np.where(pred_present[:kk],
                                         np.nan_to_num(_edge_dist(p_uv[:kk])), 0.0)
                    src_seen = (s_area[:kk] > 0) & np.isfinite(s_uv[:kk, 0])
                    null_cost = np.where(src_seen,
                                         np.nan_to_num(_edge_dist(s_uv[:kk])), 0.0)
                    err_px = float(pred_cost[window].mean())
                    nul_px = float(null_cost[window].mean())
                    n_used = n_window

                    # A timed DELETE has two halves and they ask different
                    # questions. Before the edit frame the object is supposed to
                    # be there, following the reference exactly, and that is an
                    # ordinary trajectory comparison -- scored the same way, and
                    # with the same gates, as any other object. From the edit
                    # frame on it is supposed to be gone, which is the presence
                    # cost above. Averaging the two by frame count keeps one
                    # number per object without pretending either half answers
                    # the other's question.
                    pre = gt_present.copy()
                    pre[np.flatnonzero(window)[0]:] = False
                    if int(pre.sum()) >= self.min_frames:
                        e_pre = traj_error(p_uv, g_uv, pre, seed=f,
                                           radius_px=r_px, resolution=resolution,
                                           margin_radii=self.margin_radii)
                        n_pre = int(e_pre.get("n_scored") or 0)
                        if e_pre.get("disp_mean_px") is not None and n_pre:
                            n_null = traj_error(s_uv, g_uv, pre, seed=f,
                                                radius_px=r_px,
                                                resolution=resolution,
                                                margin_radii=self.margin_radii)
                            err_px = (err_px * n_window
                                      + e_pre["disp_mean_px"] * n_pre) / (n_window + n_pre)
                            nul_px = (nul_px * n_window
                                      + (n_null.get("disp_mean_px") or 0.0) * n_pre) \
                                     / (n_window + n_pre)
                            n_used = n_window + n_pre
                            row["pre_delete_disp_px"] = round(e_pre["disp_mean_px"], 2)
                            row["pre_delete_frames"] = n_pre

                    row["disp_mean_px"] = round(err_px, 2)
                    row["abs_mean_px"] = row["disp_mean_px"]
                    row["abs_mean_px_vs_projection"] = row["disp_mean_px"]
                    row["null_disp_px"] = round(nul_px, 2)
                    row["gap_closed"] = (round(1.0 - err_px / nul_px, 3)
                                         if nul_px else None)
                    row["n_scored"] = n_used
                    if row["radius_px"]:
                        row["disp_mean_radii"] = round(
                            err_px / max(row["radius_px"], 12.0), 3)

                gt_gone = int(np.flatnonzero(window)[0]) if n_window else None
                pred_gone = (_sustained_false(pred_present, 0)
                             if n_window else None)
                row["gt_gone_frame"] = gt_gone
                row["pred_gone_frame"] = pred_gone
                row["onset_err_frames"] = (pred_gone - gt_gone
                                           if (gt_gone is not None
                                               and pred_gone is not None)
                                           else None)
                row["status"] = "deleted_presence_only"
                continue

            if row["status"] == "added":
                # Did anything get put there at all? An empty mask over the
                # whole clip is a model that ignored the instruction, and it
                # has to be distinguishable from one that placed the object
                # badly -- those are different failures.
                row["gt_area_median"] = int(np.median(g_area[f:k]))
                row["pred_area_median"] = int(np.median(p_area[f:k]))
                row["add_detected"] = bool(np.median(p_area[f:k]) > 0)
                row["add_area_ratio"] = (
                    round(float(np.median(p_area[f:k]))
                          / float(np.median(g_area[f:k])), 3)
                    if np.median(g_area[f:k]) else None)
                # Where it was put, in the unit the prompt asked in. Measured
                # at the seed frame: the placement is an initial condition, and
                # by later frames the object has been hit and moved.
                pos = (self._add_position(edit, n) or {})
                # Measure on the first frame from the seed where the tracker
                # actually produced a centroid AND the object is on screen in
                # the reference. Pinning to the seed alone throws the case away
                # whenever the tracker needs a frame or two to lock on, which
                # is a tracker artefact, not a wrong placement.
                gt_uv_all, gt_pres, _ = projected_track(edit_gt, n, eobjs,
                                                        camera_from=base_gt)
                fp = None
                for t in range(f, min(len(p_uv), len(gt_uv_all))):
                    if gt_pres[t] and np.all(np.isfinite(p_uv[t])):
                        fp = t
                        break
                ends = pos.get("endpoint_locations")
                if fp is not None and isinstance(ends, list) and len(ends) == 2:
                    anchor_uv = project_point(edit_gt, ends[0], fp,
                                              camera_from=base_gt)
                    other_uv = project_point(edit_gt, ends[1], fp,
                                             camera_from=base_gt)
                    place = placement_error(p_uv[fp], gt_uv_all[fp],
                                            anchor_uv, other_uv, r_px)
                    if place is not None:
                        place["measured_at_frame"] = int(fp)
                else:
                    place = None
                    if fp is None:
                        row["add_placement_reason"] = (
                            "the tracker never produced a centroid for the "
                            "added object on a frame where it is on screen")
                    else:
                        row["add_placement_reason"] = (
                            "this case's physics_diff has no "
                            "endpoint_locations; rebuild the suite to record "
                            "them")
                if place is not None:
                    row["add_placement"] = place
                    # What the edit asked for, in its own terms: which
                    # division point of the line, measured from which end,
                    # and that point as a fraction of the way from the first
                    # endpoint to the second.
                    row["add_at_asked"] = pos.get("at")
                    row["add_measured_from"] = pos.get("measured_from")
                    row["add_fraction_asked"] = pos.get("fraction_from_endpoint_a")
                    # The headline number: how far off along the line, in
                    # apparent radii (see placement_error on what that scale
                    # is and is not).
                    row["add_err_radii_along"] = place["along_radii"]
                    row["add_err_radii"] = place["err_radii"]
                elif not row["add_detected"]:
                    row["add_placement_reason"] = "nothing detected to measure"
                # Fall through: an added object also has a real trajectory in
                # the edited render, so it is scored like any other target.

            # Is the reference itself trustworthy IN THIS CASE? The scene's
            # text prompt may not resolve to the right object in the edited
            # render even when it works on the source (e.g. same edited item
            # changes appearance), so gate on the per-case reference verdict.
            proj_uv, proj_pres, _ = projected_track(edit_gt, n, eobjs,
                                                    camera_from=base_gt)
            ref_q = traj_error(g_uv, proj_uv, proj_pres, seed=f,
                               radius_px=r_px, resolution=resolution,
                               margin_radii=self.margin_radii)
            row["ref_disp_px"] = ref_q["disp_mean_px"]
            row["ref_disp_radii"] = ref_q["disp_mean_radii"]
            row["ref_verdict"] = ref_q["verdict"]
            # The reference is the tracked centroid; the gate and the timing
            # come from the projection, which is the only side that knows where
            # the object actually is when the tracker cannot see it.
            ref_pres = proj_pres[:k] & np.isfinite(g_uv[:k, 0])
            err = traj_error(p_uv, g_uv, ref_pres, seed=f, radius_px=r_px,
                             resolution=resolution,
                             margin_radii=self.margin_radii,
                             gate_uv=proj_uv)
            row.update(err)
            row["ref_frames_lost"] = int((proj_pres[:k]
                                          & ~np.isfinite(g_uv[:k, 0])).sum())

            # Second, independent number: absolute placement against the
            # projection. Centroid-vs-centroid cannot see a correct motion
            # carried out in the wrong place.
            vs_proj = traj_error(p_uv, proj_uv, proj_pres, seed=f,
                                 radius_px=r_px, resolution=resolution,
                                 margin_radii=self.margin_radii)
            row["abs_mean_px_vs_projection"] = vs_proj["abs_mean_px"]
            row["disp_mean_px_vs_projection"] = vs_proj["disp_mean_px"]

            # What a model scores by ignoring the prompt and reproducing the
            # source clip. Without it "67 px of error" is uninterpretable: the
            # edits themselves are worth anywhere from 15 to 5900 px, so the
            # same pixel count is a near miss in one scene and a total failure
            # in another. Measured exactly as the model is -- source track
            # against edited track -- so the two numbers can be divided.
            if row.get("added"):
                # An ADDed object has no source trajectory to be the null:
                # "ignore the prompt and reproduce the source" means the object
                # is simply not there, so there is nothing to divide by and no
                # source projection to read an offscreen frame from. Leaving
                # these None is the honest answer -- add_detected already says
                # whether the model produced the object at all.
                row["null_disp_px"] = None
                row["gap_closed"] = None
                row["src_offscreen_frame"] = None
            else:
                null = traj_error(src_tracks[n]["uv"], g_uv, ref_pres, seed=f,
                                  radius_px=r_px, resolution=resolution,
                                  margin_radii=self.margin_radii, gate_uv=proj_uv)
                row["null_disp_px"] = null["disp_mean_px"]
                if null["disp_mean_px"] and err["disp_mean_px"] is not None:
                    row["gap_closed"] = round(
                        1.0 - err["disp_mean_px"] / null["disp_mean_px"], 3)
                else:
                    row["gap_closed"] = None
                src_proj_uv, src_proj_pres, _ = projected_track(base_gt, n, bobjs)
                row["src_offscreen_frame"] = _first_offscreen(src_proj_uv,
                                                              src_proj_pres,
                                                              resolution)
            row["gt_offscreen_frame"] = _first_offscreen(proj_uv, proj_pres, resolution)
            seen = p_area[f:k] > 0
            row["pred_lost_frame"] = (int(f + np.flatnonzero(~seen)[0])
                                      if (~seen).any() else None)
            # Scale from wherever the object actually exists: the source for
            # everything the edit did not invent, the edited render for an ADD.
            s_area_o = (g_area if row.get("added")
                        else src_tracks[n]["area"])
            scale = (float(np.median(s_area_o[s_area_o > 0]))
                     if (s_area_o > 0).any() else 0.0)
            iou, n_iou = mask_iou(
                p_mask, g_mask, MASK_PIXELS,
                _plausible(p_area, scale), _plausible(g_area, scale))
            row["mask_iou"] = round(iou, 3) if iou is not None else None
            row["mask_iou_frames"] = n_iou
            row["mask_area_scale"] = round(scale, 1)
            row["status"] = ("scored" if err["disp_mean_px"] is not None
                             else "unmeasurable")

        removed = {n: r for n, r in plan.items()
                   if r.get("status") == "deleted_presence_only"}
        # Deleted objects that produced a positional cost join the trajectory
        # aggregation: "the object is still on screen" is a physical error of
        # the edit, and leaving it out is what let a model that deletes nothing
        # score disp = 0 on the very cases it failed.
        scored = {n: r for n, r in plan.items()
                  if r.get("status") == "scored"
                  or (r.get("status") == "deleted_presence_only"
                      and r.get("disp_mean_px") is not None)}

        def removal_mean(fieldname: str) -> float | None:
            vs = [r[fieldname] for r in removed.values()
                  if r.get(fieldname) is not None]
            return round(fmean(vs), 3) if vs else None

        def mean_of(fieldname: str, role: str | None = None) -> float | None:
            vs = [r[fieldname] for r in scored.values()
                  if r.get(fieldname) is not None and (role is None or r["role"] == role)]
            return round(fmean(vs), 3 if "radii" in fieldname else 2) if vs else None

        # Aggregate the two sides and then divide, rather than averaging each
        # object's ratio. An object whose edit barely moves it has a null of a
        # fraction of a pixel, and its ratio swings to -3 on tracking noise --
        # averaging those lets the least informative object in the scene decide
        # the case's score.
        num = [r["disp_mean_px"] for r in scored.values()
               if r.get("null_disp_px")]
        den = [r["null_disp_px"] for r in scored.values()
               if r.get("null_disp_px")]
        gap = round(1.0 - sum(num) / sum(den), 3) if den and sum(den) else None

        return PhysicsScores(
            status="completed" if (scored or removed) else "nothing_measurable",
            objects=plan,
            disp_edited_px=mean_of("disp_mean_px", "edited"),
            disp_affected_px=mean_of("disp_mean_px", "affected"),
            disp_all_px=mean_of("disp_mean_px"),
            disp_all_radii=mean_of("disp_mean_radii"),
            abs_all_px=mean_of("abs_mean_px_vs_projection"),
            null_disp_px=mean_of("null_disp_px"),
            gap_closed=gap,
            removal_gap=removal_mean("removal_gap"),
            onset_err_frames=removal_mean("onset_err_frames"),
            # Every object with a comparable frame, deleted or not.
            mask_iou=(lambda vs: round(fmean(vs), 3) if vs else None)(
                [r["mask_iou"] for r in plan.values()
                 if r.get("mask_iou") is not None]),
            n_removals=len(removed),
            n_measured=len(scored),
            n_objects=len(plan),
        )


def score_physics(pred_video: Path, gt_json: Path) -> PhysicsScores:
    """Kept for callers that only have the two paths.

    The real metric also needs the SOURCE ground truth (for identity matching
    and for the camera) and the edited video, which a single gt_json cannot
    supply, so this only reports whether the ground truth loads. Use
    PhysicsScorer.
    """
    if not Path(gt_json).exists():
        return PhysicsScores(status="no_ground_truth",
                             note=f"missing ground-truth file: {gt_json}")
    gt = load_ground_truth(Path(gt_json))
    return PhysicsScores(status="not_scored",
                         note=f"ground truth loads ({len(gt.get('frames') or [])} "
                              f"frames); use PhysicsScorer for the real metric")
