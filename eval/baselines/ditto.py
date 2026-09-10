"""Ditto (Editto) instruction-based video editing baseline.

Ditto -- https://github.com/EzioBy/Ditto (CVPR'26 Highlight). A LoRA
distillation of Wan2.1-VACE-14B trained on the Ditto-1M synthetic dataset.
Uses DiffSynth-Studio's `WanVideoPipeline` as the runner and applies the
Ditto LoRA on top of the VACE branch. On an A6000 the reference config
(832x480x73) sits at ~11 GB and runs in ~4 min per clip.

Two LoRA flavours ship on HuggingFace under `QingyanBai/Ditto_models`:
  - ditto_local.safetensors   -- local edits (object attribute, addition,
                                 removal, replacement). Right choice for the
                                 PCVE benchmark since every edit here is
                                 local (heavy/grippy/soft/remove).
  - ditto_global.safetensors  -- global style transfer.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .base import BaselineModel, BaselineResult

# The base checkpoint DiffSynth wants is the *original* Wan-AI layout
# (models_t5_umt5-xxl-enc-bf16.pth + Wan2.1_VAE.pth + diffusion_pytorch_model*.
# safetensors), NOT the diffusers-converted directory we point wan_vace.py at.
# ModelConfig(model_id=...) will pull from HF if not cached. Override with
# --model-id if you have the original snapshot on disk.
MODEL_ID_DEFAULT = "Wan-AI/Wan2.1-VACE-14B"

# Downloaded by tools/download_ditto_lora.sh into this repo-local path so
# multiple shards share one copy.
LORA_PATH_DEFAULT = (
    "/remote-home/chenyuanjie/models/ditto/ditto_local.safetensors"
)

# Ditto's paper resolution / frame count. The LoRA was trained here, so
# deviating too far degrades quality; keep 832x480 unless you know why.
HEIGHT_DEFAULT = 480
WIDTH_DEFAULT = 832
NUM_FRAMES_DEFAULT = 73

# Verbatim from Ditto's `inference/infer_ditto.py` -- the model card's
# negative prompt (in Chinese). Do not translate; the LoRA was trained
# with the T5 encoder seeing these exact tokens.
NEGATIVE_PROMPT_DEFAULT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，"
    "静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
    "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，"
    "背景人很多，倒着走"
)


def _round_to_vace_len(n: int) -> int:
    """Wan VACE requires num_frames of the form 4k+1."""
    n_gen = ((n - 1) // 4) * 4 + 1
    return n_gen if n_gen >= n else n_gen + 4


class DittoBaseline(BaselineModel):
    name = "ditto"

    def __init__(
        self,
        model_id: str = MODEL_ID_DEFAULT,
        lora_path: str = LORA_PATH_DEFAULT,
        lora_alpha: float = 1.0,
        negative_prompt: str = NEGATIVE_PROMPT_DEFAULT,
        num_inference_steps: int | None = None,  # LoRA is distilled; pipe picks its own default
        height: int = HEIGHT_DEFAULT,
        width: int = WIDTH_DEFAULT,
        num_frames: int = NUM_FRAMES_DEFAULT,
        seed: int = 42,
        dtype: str = "bfloat16",
        enable_vram_management: bool = True,
        tiled_vae: bool = True,
    ) -> None:
        super().__init__(
            model_id=model_id,
            lora_path=lora_path,
            lora_alpha=lora_alpha,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            height=height, width=width,
            num_frames=num_frames,
            seed=seed,
            dtype=dtype,
            enable_vram_management=enable_vram_management,
            tiled_vae=tiled_vae,
        )
        self._pipe = None
        self._device = None
        self._dtype = None

    def setup(self) -> None:
        import os
        import torch
        # DiffSynth-Studio; installed either via `pip install diffsynth`
        # or `pip install -e Ditto/` (the vendored copy).
        from diffsynth.pipelines.wan_video_new import (  # type: ignore
            WanVideoPipeline, ModelConfig,
        )

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                       "float32": torch.float32}[self.config["dtype"]]

        model_id = self.config["model_id"]
        # If model_id looks like a local directory, resolve files directly
        # via ModelConfig(path=...) to avoid touching HF/ModelScope.
        if os.path.isdir(model_id):
            import glob
            dit_files = sorted(glob.glob(os.path.join(
                model_id, "diffusion_pytorch_model*.safetensors")))
            t5_path = os.path.join(model_id, "models_t5_umt5-xxl-enc-bf16.pth")
            vae_path = os.path.join(model_id, "Wan2.1_VAE.pth")
            if not dit_files:
                raise FileNotFoundError(
                    f"No diffusion_pytorch_model*.safetensors under {model_id}")
            model_configs = [
                ModelConfig(path=dit_files, offload_device="cpu"),
                ModelConfig(path=t5_path,   offload_device="cpu"),
                ModelConfig(path=vae_path,  offload_device="cpu"),
            ]
        else:
            model_configs = [
                ModelConfig(model_id=model_id,
                            origin_file_pattern="diffusion_pytorch_model*.safetensors",
                            offload_device="cpu"),
                ModelConfig(model_id=model_id,
                            origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                            offload_device="cpu"),
                ModelConfig(model_id=model_id,
                            origin_file_pattern="Wan2.1_VAE.pth",
                            offload_device="cpu"),
            ]
        from_kwargs = dict(
            torch_dtype=self._dtype,
            device=self._device,
            model_configs=model_configs,
        )
        if os.path.isdir(model_id):
            tok_dir = os.path.join(model_id, "google", "umt5-xxl")
            if os.path.isdir(tok_dir):
                from_kwargs["tokenizer_config"] = ModelConfig(path=tok_dir)
                from_kwargs["redirect_common_files"] = False
        pipe = WanVideoPipeline.from_pretrained(**from_kwargs)

        lora_path = self.config["lora_path"]
        if not os.path.exists(lora_path):
            raise FileNotFoundError(
                f"Ditto LoRA not found at {lora_path}. Download with:\n"
                f"  huggingface-cli download QingyanBai/Ditto_models "
                f"models/ditto_local.safetensors --local-dir "
                f"{os.path.dirname(lora_path) or '.'}"
            )
        # The Ditto LoRA is trained against the VACE branch specifically,
        # not the base Wan transformer.
        pipe.load_lora(pipe.vace, lora_path, alpha=self.config["lora_alpha"])

        if self.config["enable_vram_management"]:
            # DiffSynth's own submodule-swap offloader. Roughly analogous to
            # diffusers' enable_model_cpu_offload; peak ~11 GB at 832x480x73.
            pipe.enable_vram_management()

        self._pipe = pipe
        print(f"[ditto] loaded {model_id} + LoRA {os.path.basename(lora_path)} "
              f"(alpha={self.config['lora_alpha']}) on {self._device} "
              f"({self.config['dtype']}, {self.config['width']}x"
              f"{self.config['height']}x{self.config['num_frames']})")

    def teardown(self) -> None:
        import torch
        self._pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def edit_video(
        self,
        source_video: Path,
        prompt: str,
        output_path: Path,
        *,
        edit_info: dict[str, Any],
    ) -> BaselineResult:
        assert self._pipe is not None, "Call setup() first."
        from diffsynth import VideoData, save_video  # type: ignore

        output_path.parent.mkdir(parents=True, exist_ok=True)

        height = self.config["height"]
        width = self.config["width"]

        # VideoData resamples the file to (height, width) on read; we then
        # slice to the LoRA's trained frame budget. If the source is shorter,
        # DiffSynth silently loops -- we cap at what the file has and pad up
        # to the next VACE-legal 4k+1 by holding the last frame.
        video_obj = VideoData(str(source_video), height=height, width=width)
        n_src = min(int(self.config["num_frames"]), len(video_obj))
        cond_frames = [video_obj[i] for i in range(n_src)]
        n_gen = _round_to_vace_len(n_src)
        if n_gen > n_src:
            cond_frames = cond_frames + [cond_frames[-1]] * (n_gen - n_src)

        # WanVideoPipeline exposes num_inference_steps but the distilled LoRA
        # is meant to run in far fewer steps than the base model; pass through
        # only if the caller set it explicitly.
        call_kwargs = dict(
            prompt=prompt,
            negative_prompt=self.config["negative_prompt"],
            vace_video=cond_frames,
            vace_reference_image=None,
            num_frames=n_gen,
            seed=int(self.config["seed"]),
            tiled=bool(self.config["tiled_vae"]),
        )
        if self.config["num_inference_steps"] is not None:
            call_kwargs["num_inference_steps"] = int(
                self.config["num_inference_steps"]
            )

        t0 = time.perf_counter()
        frames = self._pipe(**call_kwargs)
        elapsed = time.perf_counter() - t0

        # DiffSynth's pipe returns list[PIL.Image] directly (not .frames[0]).
        # Drop the padding so the prediction lines up with ground truth.
        frames = frames[:n_src]
        source_fps = _probe_fps(source_video) or 24
        save_video(frames, str(output_path), fps=source_fps, quality=5)

        return BaselineResult(
            output_video=output_path,
            elapsed_sec=elapsed,
            extra={
                "model_id": self.config["model_id"],
                "lora_path": self.config["lora_path"],
                "lora_alpha": self.config["lora_alpha"],
                "height": height,
                "width": width,
                "num_frames": len(frames),
                "num_frames_generated": n_gen,
                "num_frames_source": n_src,
                "seed": self.config["seed"],
                "num_inference_steps": self.config["num_inference_steps"],
            },
        )


def _probe_fps(video_path: Path) -> int | None:
    """Read fps from a video file via imageio-ffmpeg's bundled ffprobe."""
    import re
    import subprocess
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None
    r = subprocess.run([ffmpeg, "-i", str(video_path)],
                       capture_output=True, text=True)
    m = re.search(r"(\d+(?:\.\d+)?)\s*fps", r.stderr)
    return int(round(float(m.group(1)))) if m else None
