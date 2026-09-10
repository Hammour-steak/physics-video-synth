"""GroundedSAM2 as a multi-object tracker for the trajectory metric.

Same one-propagation-per-video shape as a plain SAM2 video tracker, but the anchor prompt is a
text description instead of a projected point. Grounding DINO runs on each
object's anchor frame, produces a box, that box is handed to SAM2's video
predictor, and the rest of the pipeline is unchanged.

For visually distinct objects a unique text is enough ("red mallet",
"blue mallet"). For visually identical groups (bowling's three pins,
domino_chain's four tiles), the same text describes every member; one
detection call returns N boxes and each is assigned to a GT identity by
nearest-neighbour against a positional hint (the projected first-frame origin,
which is edit-invariant and identical for every case in a scene). The hint is
NOT fed to SAM2 -- it only decides which detected box gets which name.
"""
from __future__ import annotations

import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# Canonical grid every mask is resampled onto before it is compared or stored.
MASK_W, MASK_H = 640, 360
import torch
from PIL import Image

SAM2_CKPT = "/remote-home/chenyuanjie/models/sam2.1-hiera-large/sam2.1_hiera_large.pt"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GDINO_ID = "IDEA-Research/grounding-dino-tiny"


def _hungarian(cost: np.ndarray) -> list[tuple[int, int]]:
    """Small assignment problem; the group is always <= 10, so scipy is overkill.

    Returns (row, col) pairs minimising total cost with each row/col used once.
    Uses a simple greedy on the sorted cost list, which is optimal here because
    the objects are far enough apart in every scene that the assignment is
    unambiguous when it exists.
    """
    R, C = cost.shape
    order = sorted(((cost[r, c], r, c) for r in range(R) for c in range(C)))
    ur, uc, out = set(), set(), []
    for _, r, c in order:
        if r in ur or c in uc:
            continue
        ur.add(r); uc.add(c); out.append((r, c))
        if len(out) == min(R, C):
            break
    return out


class GroundedSam2Tracker:
    """GroundingDINO (per anchor frame, per text) + SAM2 video predictor.

    Both models load once; call `track()` per video.
    """

    def __init__(self, sam2_ckpt: str = SAM2_CKPT, sam2_cfg: str = SAM2_CFG,
                 gdino_id: str = GDINO_ID, device: str | None = None,
                 box_threshold: float = 0.20, text_threshold: float = 0.20):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        from sam2.build_sam import build_sam2_video_predictor
        self.predictor = build_sam2_video_predictor(sam2_cfg, sam2_ckpt,
                                                    device=self.device)

        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self.gdino_proc = AutoProcessor.from_pretrained(gdino_id)
        self.gdino = AutoModelForZeroShotObjectDetection.from_pretrained(
            gdino_id).to(self.device).eval()

    # ------------------------------------------------------------------ gdino
    def _detect(self, frame: np.ndarray, text: str) -> np.ndarray:
        """One text prompt on one frame -> boxes, xyxy in pixels (N, 4).

        Grounding DINO expects the prompt in lowercase and terminated with a
        period; multiple phrases in one prompt are separated by periods. We
        take one phrase at a time -- the tracker groups objects that share a
        text, and mixing distinct phrases in one call makes it harder to say
        which box came from which phrase.
        """
        prompt = text.strip().lower()
        if not prompt.endswith("."):
            prompt = prompt + "."
        img = Image.fromarray(frame)
        with torch.inference_mode():
            inputs = self.gdino_proc(images=img, text=prompt,
                                     return_tensors="pt").to(self.device)
            out = self.gdino(**inputs)
        results = self.gdino_proc.post_process_grounded_object_detection(
            out, threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[img.size[::-1]])[0]
        boxes = results["boxes"].detach().cpu().numpy()  # (N, 4) xyxy
        scores = results["scores"].detach().cpu().numpy()
        # Sort high score first so single-object callers can just take [0].
        if len(scores):
            order = np.argsort(-scores)
            boxes = boxes[order]
        return boxes

    def _assign_boxes(self, boxes: np.ndarray, group: list[dict],
                      resolution) -> dict[str, np.ndarray]:
        """Give each group member a box.

        Unique-text objects (group of one): prefer the highest-scoring box
        within ~2 diameters of the positional hint (defense against GDINO
        mislabelling a same-shape neighbour). If nothing lands in the gate --
        the object has legitimately moved far from its source position, e.g.
        an initial_velocity edit -- fall back to the top-1 detection so
        `unmeasurable` is not the default answer whenever the physics diverges
        from the source at the anchor frame.

        Identical-appearance groups (bowling pins) still need the hint gate:
        the members share a text so the only signal for identity is position,
        and a missing pin should be reported unassigned rather than stolen
        from a neighbour.
        """
        W, H = resolution
        out: dict[str, np.ndarray] = {}
        if len(boxes) == 0:
            return out
        centres = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2,
                            (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)

        if len(group) == 1:
            g = group[0]
            hint = g.get("hint_uv")
            r = g.get("hint_radius_px")
            pick = 0                              # top-1 by default
            if hint is not None and r and len(boxes) > 1:
                # Multiple detections: prefer the one within the hint gate.
                d = np.linalg.norm(centres - np.asarray(hint), axis=1)
                mask = d <= 2.0 * max(float(r), 12.0)
                if mask.any():
                    pick = int(np.flatnonzero(mask)[0])
            out[g["name"]] = boxes[pick]
            return out

        # Multiple members share the text: every member must have a hint.
        hints = np.array([g["hint_uv"] for g in group])
        # cost[i, j] = distance from member i's hint to box j
        cost = np.linalg.norm(hints[:, None, :] - centres[None, :, :], axis=2)
        # Reject impossibly-far matches (> 2 diameters): a missing pin should
        # be reported unassigned, not stolen from a nearby pin.
        gate = np.array([2.0 * max(g.get("hint_radius_px") or 16.0, 12.0)
                         for g in group])
        for i, j in _hungarian(cost):
            if cost[i, j] <= gate[i]:
                out[group[i]["name"]] = boxes[j]
        return out

    # ------------------------------------------------------------------ track
    def track(self, video: np.ndarray,
              prompts: dict[str, dict],
              ) -> dict[str, dict[str, np.ndarray]]:
        """video (T,H,W,3) uint8.

        prompts: {name: {"text": str, "anchor_frame": int,
                         "hint_uv": (u, v) | None,
                         "hint_radius_px": float | None}}

        Names sharing the same (text, anchor_frame) go through one GDINO call
        together and are assigned to the returned boxes by nearest hint.

        -> {name: {"uv": (T,2) NaN where no mask, "area": (T,) pixels,
                   "box": (4,) xyxy at anchor, or None if detection failed}}
        """
        T = len(video)
        names = list(prompts)
        out = {n: {"uv": np.full((T, 2), np.nan), "area": np.zeros(T),
                   "mask": np.zeros((T, MASK_W * MASK_H // 8), np.uint8),
                   "box": None}
               for n in names}
        if not names:
            return out
        H, W = video.shape[1:3]

        # Group by (text, anchor_frame): one detection call per group.
        groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
        for n, p in prompts.items():
            groups[(p["text"], int(p["anchor_frame"]))].append(
                {"name": n,
                 "hint_uv": p.get("hint_uv"),
                 "hint_radius_px": p.get("hint_radius_px")})

        seeds: dict[str, tuple[int, np.ndarray]] = {}  # name -> (frame, box)
        for (text, f), group in groups.items():
            f = int(np.clip(f, 0, T - 1))
            boxes = self._detect(video[f], text)
            assigned = self._assign_boxes(boxes, group, (W, H))
            for name, box in assigned.items():
                seeds[name] = (f, box.astype(np.float32))
                out[name]["box"] = box

        if not seeds:
            return out

        # Same SAM2 video-propagator wiring as before.
        ids = {i + 1: n for i, n in enumerate(seeds)}
        tmp = Path(tempfile.mkdtemp(prefix="gsam2_frames_"))
        try:
            for i, fr in enumerate(video):
                Image.fromarray(fr).save(tmp / f"{i:05d}.jpg", quality=95)
            with torch.inference_mode(), torch.autocast(self.device,
                                                        dtype=torch.bfloat16):
                st = self.predictor.init_state(video_path=str(tmp),
                                               offload_video_to_cpu=True,
                                               offload_state_to_cpu=True)
                for oid, n in ids.items():
                    f, box = seeds[n]
                    self.predictor.add_new_points_or_box(
                        inference_state=st, frame_idx=int(f), obj_id=oid,
                        box=box)
                for fidx, out_ids, logits in self.predictor.propagate_in_video(st):
                    for j, oid in enumerate(out_ids):
                        n = ids.get(int(oid))
                        if n is None or fidx < seeds[n][0]:
                            continue
                        m = (logits[j] > 0).cpu().numpy().squeeze()
                        if m.sum():
                            ys, xs = np.nonzero(m)
                            out[n]["uv"][fidx] = [xs.mean(), ys.mean()]
                            out[n]["area"][fidx] = m.sum()
                        # Kept for the spatial IoU, on a fixed grid: the
                        # prediction and the reference are different renders at
                        # different resolutions (VOID 672x384, the reference
                        # 1280x720) and two masks can only be intersected on a
                        # common one. Half the benchmark's resolution keeps a
                        # 16 px object 8 px across -- still several hundred
                        # pixels of mask -- while making the cache 4x smaller,
                        # and it is stored packed, one bit per pixel.
                        small = cv2.resize(m.astype(np.uint8),
                                           (MASK_W, MASK_H),
                                           interpolation=cv2.INTER_AREA)
                        out[n]["mask"][fidx] = np.packbits(small > 0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return out
