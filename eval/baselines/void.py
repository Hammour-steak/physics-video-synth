"""Netflix VOID baseline -- https://github.com/Netflix/void-model

VOID ("Video Object and Interaction Deletion") removes an object from a video
along with the traces of it having been there. It only does removal, so it is
run on this benchmark's DELETE edits and nothing else:

    python run_baseline.py --baseline void --filter-kind DELETE ...

This adapter drives VOID's *own* pipeline end to end and generates nothing
itself. That is deliberate: the point of a benchmark number is that nobody can
argue with how it was produced, and a mask built by some substitute
segmenter -- or worse, projected out of this benchmark's ground truth -- would
be a different experiment from the one the VOID authors describe. So the two
official entry points are called as-is:

    VLM-MASK-REASONER/run_pipeline.sh   points -> quadmask_0.mp4
        stage 1  SAM2 segmentation of the primary object from click points
        stage 2  Gemini VLM analysis of interaction-affected regions
        stage 3  grey masks for those regions       (needs SAM3)
        stage 4  combine into the quadmask
    inference/cogvideox_fun/predict_v2v.py          quadmask -> edited video

Two inputs come from outside this file and cannot be synthesised here:

  * GEMINI_API_KEY in the environment. Stage 2 raises without it, and stage 2
    is what produces the 127 "affected region" layer that separates VOID from
    an ordinary inpainter. It is read from the environment and never written
    to disk by this adapter.
  * Click points identifying the object, one entry per case in
    metrics/void_points.json. In VOID's flow a person places these with
    point_selector_gui.py; they are the instruction, the same role the text
    prompt plays for the other baselines. Deriving them from ground truth
    would hand VOID the answer, so a missing entry is an error, not something
    to guess around.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .base import BaselineModel, BaselineResult

REPO_DEFAULT = "/remote-home/chenyuanjie/void-model"
BASE_MODEL_DEFAULT = "/remote-home/chenyuanjie/models/CogVideoX-Fun-V1.5-5b-InP"
PASS1_DEFAULT = "/remote-home/chenyuanjie/models/void/void_pass1.safetensors"
# SAM 2.0, not 2.1. stage1_sam2_segmentation.py builds the predictor with
# model_cfg="sam2_hiera_l.yaml" and its own help text points at the 072824
# release; a 2.1 checkpoint carries extra keys (no_obj_embed_spatial,
# obj_ptr_tpos_proj.*) and load_state_dict rejects it outright.
SAM2_CKPT_DEFAULT = "/remote-home/chenyuanjie/models/sam2_hiera_large.pt"
CONFIG_DEFAULT = "config/quadmask_cogvideox.py"
# VOID runs in its own conda env, not the one the rest of the eval uses.
# requirements.txt pins torch 2.7.1 / numpy 1.26 / transformers 4.57, while
# ditto and wan_vace need torch 2.9 / numpy 2 / transformers 5.14 -- and
# albumentations, which VOID imports, genuinely breaks on numpy 2. Sharing one
# env would mean either downgrading the stack the other two baselines and the
# scoring were validated on, or running VOID on a stack its authors did not
# specify. A second env avoids both.
VOID_ENV_DEFAULT = "/remote-home/chenyuanjie/miniconda/envs/void"
POINTS_DEFAULT = "metrics/void_points.json"
BG_PROMPTS_DEFAULT = "metrics/void_bg_prompts.json"


class VoidBaseline(BaselineModel):
    name = "void"

    def __init__(
        self,
        repo_dir: str = REPO_DEFAULT,
        base_model: str = BASE_MODEL_DEFAULT,
        pass1_path: str = PASS1_DEFAULT,
        pass2_path: str | None = None,      # None = pass 1 only
        sam2_checkpoint: str = SAM2_CKPT_DEFAULT,
        config_path: str = CONFIG_DEFAULT,
        points_file: str | None = None,
        vlm_model: str = "gemini-3.1-flash-lite",
        # VOID is fixed to 85 frames @ 12 fps ("Due to the temporal
        # compression, the model does not handle all output sizes. We fixed
        # this to 85 frames @ 12 fps" -- Netflix/void-model issue #17). Our
        # sources are 96 frames @ 24 fps, so every object moves half as far
        # between adjacent frames as the model was trained for. input_fps
        # resamples the clip before the mask pipeline sees it, so mask and
        # video stay on the same timebase.
        input_fps: int | None = None,
        # temporal_padding pads up to max(temporal_window_size, ...) by
        # mirroring the clip. At the default 85 a 48-frame input becomes 37
        # frames of played-backwards padding; issue #17's answer is that this
        # is the knob to change.
        temporal_window_size: int | None = None,
        out_fps: int | None = None,
        height: int = 384,
        width: int = 672,
        seed: int = 42,
        work_dir: str | None = None,
        void_env: str = VOID_ENV_DEFAULT,
        python_exe: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            repo_dir=repo_dir, base_model=base_model, pass1_path=pass1_path,
            pass2_path=pass2_path, sam2_checkpoint=sam2_checkpoint,
            config_path=config_path, points_file=points_file,
            vlm_model=vlm_model, input_fps=input_fps,
            temporal_window_size=temporal_window_size, out_fps=out_fps,
            height=height, width=width, seed=seed, work_dir=work_dir,
            void_env=void_env,
            python_exe=python_exe or str(Path(void_env) / "bin" / "python"),
            **kwargs,
        )
        self._points: dict[str, dict] = {}
        self._bg_prompts: dict[str, str] = {}

    # ---------------------------------------------------------------- setup
    def setup(self) -> None:
        repo = Path(self.config["repo_dir"])
        needed = {
            "VOID repo": repo,
            "inference script": repo / "inference/cogvideox_fun/predict_v2v.py",
            "mask pipeline": repo / "VLM-MASK-REASONER/run_pipeline.sh",
            "CogVideoX-Fun base model": Path(self.config["base_model"]),
            "VOID pass-1 weights": Path(self.config["pass1_path"]),
            "SAM2 checkpoint": Path(self.config["sam2_checkpoint"]),
            "VOID conda env": Path(self.config["void_env"]) / "bin" / "python",
        }
        missing = [f"{k} ({v})" for k, v in needed.items() if not v.exists()]
        if missing:
            raise FileNotFoundError("VOID is not fully installed:\n  " +
                                    "\n  ".join(missing))
        if not os.environ.get("GEMINI_API_KEY"):
            raise RuntimeError(
                "GEMINI_API_KEY is not set. VOID's stage 2 (VLM analysis) "
                "requires it and cannot be skipped -- it produces the "
                "affected-region layer of the quadmask. Export it in the "
                "shell that runs this."
            )
        # Checked in VOID's env, not this process's: that is where the mask
        # pipeline will actually import it.
        probe = subprocess.run(
            [self.config["python_exe"], "-c", "import sam3"],
            capture_output=True, text=True)
        if probe.returncode != 0:
            raise RuntimeError(
                "SAM3 is not importable in the VOID env "
                f"({self.config['void_env']}); "
                "VLM-MASK-REASONER/run_pipeline.sh calls "
                "stage3a_generate_grey_masks_v2.py, which imports it. Install "
                "from https://github.com/facebookresearch/sam3."
            )

        eval_dir = Path(__file__).resolve().parents[1]
        pts = Path(self.config["points_file"] or (eval_dir / POINTS_DEFAULT))
        if not pts.exists():
            raise FileNotFoundError(
                f"{pts} not found. It holds the click points identifying the "
                "object to remove in each DELETE case -- VOID's instruction "
                "input, authored with point_selector_gui.py."
            )
        self._points = json.loads(pts.read_text(encoding="utf-8"))
        bg = eval_dir / BG_PROMPTS_DEFAULT
        if not bg.exists():
            raise FileNotFoundError(
                f"{bg} not found. It holds the hand-written prompt.json `bg` "
                "text for each DELETE case."
            )
        self._bg_prompts = json.loads(bg.read_text(encoding="utf-8"))

    # ------------------------------------------------------------- prompting
    def _bg_prompt(self, global_id: str) -> str:
        """The `bg` field of VOID's prompt.json, hand-written per case.

        Not generated. In VOID's flow this is a sentence a person writes --
        the repo has no code that produces it -- and their own samples read
        like scene descriptions ("A ball rolls off the table.", "Two pillows
        placed on the table."), not object inventories. An earlier version of
        this adapter composed it by listing every object in the scene's
        text_prompts.json minus the removed one, which both drifted from that
        style and put objects in the prompt that are not in the video at all
        (the yellow ball and green mallet exist only for the ADD edits).
        A missing entry is an error rather than something to synthesise.
        """
        bg = self._bg_prompts.get(global_id)
        if not bg:
            raise KeyError(
                f"no bg prompt for {global_id} in metrics/void_bg_prompts.json. "
                "VOID's prompt.json is written by hand; add an entry describing "
                "what remains after the removal."
            )
        return bg

    # ---------------------------------------------------------------- edit
    def edit_video(self, source_video: Path, prompt: str, output_path: Path,
                   *, edit_info: dict[str, Any]) -> BaselineResult:
        if edit_info.get("edit_kind") != "DELETE":
            raise ValueError(
                f"VOID only removes objects; {edit_info.get('global_id')} is a "
                f"{edit_info.get('edit_kind')} edit. Use --filter-kind DELETE."
            )
        gid = edit_info["global_id"]
        spec = self._points.get(gid)
        if not spec:
            raise KeyError(
                f"no click points for {gid} in the points file. Add an entry "
                "(authored with VOID's point_selector_gui.py) rather than "
                "letting the mask be inferred some other way."
            )

        t0 = time.perf_counter()
        seq = f"{edit_info['scene']}__{edit_info['case_id']}"
        # Absolute. predict_v2v.py runs with cwd=repo_dir, so a work dir
        # spelled relative to the eval directory (which is what a relative
        # --benchmark-root produces) does not exist from where it looks.
        work = Path(self.config["work_dir"]
                    or (output_path.parent.parent.parent / "void_work")).resolve()
        seq_dir = work / "data" / seq
        seq_dir.mkdir(parents=True, exist_ok=True)
        # stage1_sam2_segmentation.py seeds SAM2 at the frame the object is
        # clicked and propagates forward only, so an object that enters the
        # clip late gets a mask shorter than the video (the apple: 87 against
        # 96) and predict_v2v.py dies rearranging the two together. Hand VOID
        # the clip from that frame instead: the frames before it hold nothing
        # to remove, and they go back on -- untouched -- when the result is
        # assembled. None of VOID's own processing changes.
        start = int(spec.get("first_appears_frame", 0))
        dst = seq_dir / "input_video.mp4"
        ff = shutil.which("ffmpeg") or str(
            Path(self.config["void_env"]) / "bin" / "ffmpeg")
        filters = []
        if start:
            filters.append("select=gte(n\\," + str(start) + ")")
        if self.config["input_fps"]:
            # Resample before anything else: stage1 segments this file, so the
            # quadmask is built on the same timebase and cannot desync.
            filters.append("fps=" + str(int(self.config["input_fps"])))
        if filters:
            r = subprocess.run(
                [ff, "-y", "-loglevel", "error", "-i", str(source_video),
                 "-vf", ",".join(filters), "-fps_mode", "passthrough",
                 "-an", str(dst)],
                capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(
                    f"preparing {source_video} (trim={start}) failed:\n"
                    f"{r.stderr[-400:]}")
        else:
            shutil.copy2(source_video, dst)

        # --- stage 1-4: VOID's own mask pipeline --------------------------
        entry: dict[str, Any] = {
            "video_path": str((seq_dir / "input_video.mp4").resolve()),
            "output_dir": str(seq_dir.resolve()),
            "instruction": spec.get("instruction", prompt),
            # 0, not spec's value: input_video.mp4 was cut to start there.
            "first_appears_frame": 0,
        }
        if spec.get("primary_points_by_frame"):
            entry["primary_points_by_frame"] = {
                str(int(f) - start): pts
                for f, pts in spec["primary_points_by_frame"].items()}
        else:
            entry["primary_points"] = spec["primary_points"]
        cfg_path = seq_dir / "config_points.json"
        cfg_path.write_text(json.dumps({"videos": [entry]}, indent=2),
                            encoding="utf-8")

        # The same four stages run_pipeline.sh runs, in its order, with its
        # flags -- called individually only so stage 2 can be given --model.
        # VOID hardcodes DEFAULT_MODEL = "gemini-3-pro-preview", which Google
        # has retired ("no longer available. Please update your code to use
        # models/gemini-3.1-pro-preview"), so the shipped default cannot run at
        # all. --model is VOID's own supported flag; run_pipeline.sh simply
        # does not forward it.
        mr = "VLM-MASK-REASONER"
        cfg = str(cfg_path.resolve())
        py = self.config["python_exe"]
        self._run([py, f"{mr}/stage1_sam2_segmentation.py", "--config", cfg,
                   "--sam2-checkpoint", str(Path(self.config["sam2_checkpoint"]).resolve()),
                   "--device", "cuda"], what=f"stage 1 (SAM2) for {seq}")
        self._run([py, f"{mr}/stage2_vlm_analysis.py", "--config", cfg,
                   "--model", self.config["vlm_model"]],
                  what=f"stage 2 (VLM) for {seq}")
        self._run([py, f"{mr}/stage3a_generate_grey_masks_v2.py", "--config", cfg],
                  what=f"stage 3a (grey masks) for {seq}")
        self._run([py, f"{mr}/stage4_combine_masks.py", "--config", cfg],
                  what=f"stage 4 (combine) for {seq}")

        # Check each stage's own artifact, not just the final one. stage 2
        # catches its own exceptions, prints "Video 1 skipped" and exits 0 --
        # so a VLM that could not be reached does not fail run_pipeline.sh
        # (set -e never fires) and the first sign of trouble is a missing
        # quadmask three stages later. Naming the stage that actually stopped
        # is the difference between a five-minute diagnosis and an hour's.
        for artifact, stage in ((seq_dir / "black_mask.mp4", "1 (SAM2 segmentation)"),
                                (seq_dir / "vlm_analysis.json", "2 (VLM analysis)")):
            if not artifact.exists():
                raise RuntimeError(
                    f"stage {stage} produced no {artifact.name} for {seq}. "
                    "It may have failed and still exited 0 -- re-run that "
                    "stage alone to see its output. A common cause for stage "
                    "2 is that generativelanguage.googleapis.com is "
                    "unreachable from this host; set https_proxy."
                )
        quadmask = seq_dir / "quadmask_0.mp4"
        if not quadmask.exists():
            raise RuntimeError(
                f"stages 1 and 2 produced their artifacts but no "
                f"quadmask_0.mp4 appeared in {seq_dir}; check stages 3-4."
            )

        # --- prompt.json --------------------------------------------------
        (seq_dir / "prompt.json").write_text(
            json.dumps({"bg": self._bg_prompt(gid)}, ensure_ascii=False, indent=1),
            encoding="utf-8")

        # --- inference ----------------------------------------------------
        # Write at the source clip's own rate. config.data.fps is only the fps
        # stamped on the saved mp4 (save_videos_grid), so this changes how the
        # result is timed, not how it is generated: without it every prediction
        # comes back labelled 12 fps against a 24 fps source and plays at half
        # speed, which the scoring reads as the object moving half as far.
        n_src, fps_src = self._video_meta(source_video)
        out_fps = self.config["out_fps"] or int(round(fps_src))
        save_path = work / "out" / seq
        save_path.mkdir(parents=True, exist_ok=True)
        self._predict(seq, work / "data", save_path, self.config["pass1_path"],
                      out_fps=out_fps)
        if self.config["pass2_path"]:
            self._predict(seq, work / "data", save_path,
                          self.config["pass2_path"], out_fps=out_fps)

        # predict_v2v.py writes two files per sequence, e.g.
        #   moving_ball-fg=-1-0001.mp4        (85, 384,  672, 3)  the result
        #   moving_ball-fg=-1-0001_tuple.mp4  (85, 384, 2688, 3)  a 4-panel
        #                                      input|mask|output|diff contact
        #                                      sheet, written *second*
        # Taking the newest mp4 therefore picks the contact sheet, which would
        # be scored as if it were the edited video. Select on the name.
        produced = sorted((p for p in save_path.rglob("*.mp4")
                           if not p.stem.endswith("_tuple")),
                          key=lambda p: p.stat().st_mtime)
        if not produced:
            raise RuntimeError(
                f"predict_v2v.py wrote no non-tuple mp4 under {save_path}; "
                f"found {[p.name for p in save_path.rglob('*.mp4')]}"
            )
        # temporal_padding() in videox_fun/utils/utils.py rounds the clip up to
        # a length the VAE can take (96 -> 101 for these sources) by appending
        # the clip played backwards, so the tail of every result is padding the
        # source has no counterpart for. Drop it: the metrics line prediction
        # frame i up against ground-truth frame i and score whatever is there.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        n_out = self._video_meta(produced[-1])[0]
        n_edited = n_src - start          # what VOID was actually given
        if start == 0 and n_out <= n_src:
            shutil.copy2(produced[-1], output_path)
        elif start == 0:
            r = subprocess.run(
                [ff, "-y", "-loglevel", "error", "-i", str(produced[-1]),
                 "-frames:v", str(n_src), "-an", str(output_path)],
                capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(
                    f"trimming {produced[-1]} to {n_src} frames failed:\n"
                    f"{r.stderr[-400:]}")
        else:
            # Source frames [0, start) -- which VOID never saw, and which hold
            # nothing to remove -- then its result, minus the padding.
            w, h = self.config["width"], self.config["height"]
            chain = (
                "[0:v]select=lt(n\\," + str(start) + "),"
                "scale=" + str(w) + ":" + str(h) + ",setsar=1,"
                "setpts=N/FRAME_RATE/TB[lead];"
                "[1:v]select=lt(n\\," + str(n_edited) + "),setsar=1,"
                "setpts=N/FRAME_RATE/TB[body];"
                "[lead][body]concat=n=2:v=1[v]"
            )
            r = subprocess.run(
                [ff, "-y", "-loglevel", "error",
                 "-i", str(source_video), "-i", str(produced[-1]),
                 # -r alone: the concat filter already emits one uniform
                 # stream (both inputs run at out_fps), and pairing -r with
                 # -fps_mode is rejected as contradictory.
                 "-filter_complex", chain, "-map", "[v]", "-r", str(out_fps),
                 "-an", str(output_path)],
                capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(
                    f"stitching {start} lead-in frames onto {produced[-1]} "
                    f"failed:\n{r.stderr[-400:]}")
            n_final = self._video_meta(output_path)[0]
            if n_final != n_src:
                raise RuntimeError(
                    f"assembled {output_path.name} has {n_final} frames, "
                    f"expected {n_src} (source {n_src}, lead-in {start}, "
                    f"VOID {n_out})")

        return BaselineResult(
            output_video=output_path,
            elapsed_sec=time.perf_counter() - t0,
            extra={
                "mask_source": "void_official_pipeline",
                "vlm_model": self.config["vlm_model"],
                "input_fps": self.config["input_fps"],
                "temporal_window_size": self.config["temporal_window_size"],
                "points": entry.get("primary_points")
                          or entry.get("primary_points_by_frame"),
                "quadmask": str(quadmask.resolve()),
                "bg_prompt": json.loads((seq_dir / "prompt.json").read_text())["bg"],
                "two_pass": bool(self.config["pass2_path"]),
                "height": self.config["height"], "width": self.config["width"],
                "seed": self.config["seed"],
                "out_fps": out_fps,
                "source_frames": n_src,
                "void_frames_before_trim": n_out,
                "lead_in_frames": start,
                "void_output": str(produced[-1]),
            },
        )

    # ---------------------------------------------------------------- utils
    def _run(self, cmd: list[str], *, what: str) -> None:
        env = dict(os.environ)          # carries GEMINI_API_KEY through
        # run_pipeline.sh invokes a bare `python`, so VOID's env has to be
        # first on PATH or the stages would run against the eval's stack.
        env["PATH"] = (str(Path(self.config["void_env"]) / "bin") +
                       os.pathsep + env.get("PATH", ""))
        p = subprocess.run(cmd, cwd=Path(self.config["repo_dir"]),
                           capture_output=True, text=True, env=env)
        if p.returncode != 0:
            tail = "\n".join((p.stderr or p.stdout or "").splitlines()[-30:])
            raise RuntimeError(f"{what} failed:\n{tail}")

    @staticmethod
    def _video_meta(path: Path) -> tuple[int, float]:
        import cv2
        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if n <= 0 or not fps:
            raise RuntimeError(f"could not read frame count / fps from {path}")
        return n, fps

    def _predict(self, seq: str, data_root: Path, save_path: Path,
                 transformer_path: str, *, out_fps: int | None = None) -> None:
        self._run([
            self.config["python_exe"],
            "inference/cogvideox_fun/predict_v2v.py",
            "--config", self.config["config_path"],
            f"--config.data.data_rootdir={data_root}",
            f"--config.experiment.run_seqs={seq}",
            f"--config.experiment.save_path={save_path}",
            f"--config.video_model.model_name={self.config['base_model']}",
            f"--config.video_model.transformer_path={transformer_path}",
        ] + ([f"--config.video_model.temporal_window_size={int(self.config['temporal_window_size'])}"]
             if self.config["temporal_window_size"] else [])
          + ([f"--config.data.fps={int(out_fps)}"] if out_fps else []),
          what=f"predict_v2v.py for {seq}")
