"""Wan 2.1 VACE-14B baseline via diffusers.

VACE = Video-based All-in-one Creation and Editing. The 14B checkpoint on
HuggingFace is `Wan-AI/Wan2.1-VACE-14B` (Apache 2.0). The diffusers
integration exposes it as WanVACEPipeline; the exact API surface has moved
a couple of times since Feb 2025 so treat the call below as a template and
adjust to whatever version of `diffusers` you actually have.

Verified path (diffusers >= 0.33 approximately):
    from diffusers import WanVACEPipeline
    pipe = WanVACEPipeline.from_pretrained(
        "Wan-AI/Wan2.1-VACE-14B", torch_dtype=torch.bfloat16,
    ).to("cuda")
    frames = pipe(
        prompt=prompt,
        video=source_frames,          # list[PIL.Image]
        num_inference_steps=50,
        height=720, width=1280,
        num_frames=len(source_frames),
    ).frames[0]

On an A6000 (48 GB) at 720p / bf16, expect roughly 4-8 min per 3-second
clip. If you OOM, first try `pipe.enable_model_cpu_offload()`, then
--height 480 --width 720 (matches most eval papers).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .base import BaselineModel, BaselineResult

MODEL_ID_DEFAULT = "/remote-home/chenyuanjie/models/Wan2.1-VACE-14B-diffusers"

# The negative prompt every Wan 2.1 model-card example passes. Leaving it empty
# lets the CFG unconditional branch drift towards exactly the artefacts listed
# here -- the first two terms ("Bright tones, overexposed") are the oversaturated
# wash we measured when it was omitted.
NEGATIVE_PROMPT_DEFAULT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, "
    "works, paintings, images, static, overall gray, worst quality, low quality, "
    "JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn "
    "hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused "
    "fingers, still picture, messy background, three legs, many people in the "
    "background, walking backwards"
)


def _check_resolution(src_h: int, src_w: int, height: int, width: int) -> None:
    """Fail loudly when height/width would desynchronise the spatial latents.

    WanVACEPipeline builds conditioning latents from the source video after
    rescaling it to fit height x width and flooring both sides to a multiple of
    16, but it allocates noise latents from the requested height/width. When the
    two disagree the conditioning is applied with a row/column offset and the
    output tiles or wraps -- and unlike the frame-count case, the pipeline emits
    no warning at all. 832x480 against a 16:9 source hits this (464 != 480), as
    does the 720x480 fallback suggested in eval/README.md.
    """
    base, vae = 16, 8
    vh, vw = src_h, src_w
    if vh * vw > height * width:
        scale = min(width / vw, height / vh)
        vh, vw = int(vh * scale), int(vw * scale)
    if vh % base or vw % base:
        vh, vw = (vh // base) * base, (vw // base) * base
    if (vh // vae, vw // vae) != (height // vae, width // vae):
        raise ValueError(
            f"Resolution {width}x{height} is incompatible with a {src_w}x{src_h} "
            f"source: conditioning latents would be {vw // vae}x{vh // vae} but "
            f"noise latents {width // vae}x{height // vae}. Pick a size with the "
            f"same aspect ratio whose sides are multiples of {base} -- for 16:9 "
            f"that is 512x288, 768x432, 1024x576 or 1280x720."
        )


def _vace_num_frames(n: int) -> int:
    """Round n up to the nearest VACE-legal length (4k+1).

    WanVACEPipeline rounds `num_frames` to 4k+1 internally, but builds the
    conditioning latents from the *video* it is handed. Feeding it a 72-frame
    clip therefore yields 18 conditioning latent frames against 19 noise latent
    frames -- the trailing 4 output frames get no conditioning at all and
    collapse into noise. Padding the clip to 4k+1 keeps both at 19.
    """
    n_gen = ((n - 1) // 4) * 4 + 1
    return n_gen if n_gen >= n else n_gen + 4


class WanVACEBaseline(BaselineModel):
    name = "wan_vace_14b"

    def __init__(
        self,
        model_id: str = MODEL_ID_DEFAULT,
        negative_prompt: str = NEGATIVE_PROMPT_DEFAULT,
        flow_shift: float | None = None,
        num_inference_steps: int = 50,
        height: int = 720,
        width: int = 1280,
        guidance_scale: float = 5.0,
        seed: int = 42,
        enable_cpu_offload: bool = False,
        offload_mode: str | None = None,
        num_blocks_per_group: int = 4,
        dtype: str = "bfloat16",
    ) -> None:
        # Legacy alias: `--enable-cpu-offload` used to be the only knob and
        # meant sequential. Keep it working when offload_mode is unset.
        if offload_mode is None:
            offload_mode = "sequential" if enable_cpu_offload else "none"
        if offload_mode not in ("sequential", "group", "model", "none"):
            raise ValueError(
                f"offload_mode must be one of sequential/group/model/none, "
                f"got {offload_mode!r}"
            )
        # The checkpoint ships flow_shift=3.0, which is the 480P value; the Wan
        # model card calls for 5.0 at 720P. Pick by resolution unless told.
        if flow_shift is None:
            flow_shift = 5.0 if height * width >= 720 * 1280 else 3.0
        super().__init__(
            model_id=model_id,
            negative_prompt=negative_prompt,
            flow_shift=flow_shift,
            num_inference_steps=num_inference_steps,
            height=height, width=width,
            guidance_scale=guidance_scale, seed=seed,
            enable_cpu_offload=enable_cpu_offload,
            offload_mode=offload_mode,
            num_blocks_per_group=num_blocks_per_group,
            dtype=dtype,
        )
        self._pipe = None
        self._device = None
        self._dtype = None
        self._generator = None

    def setup(self) -> None:
        import torch
        # Delayed import so `run_baseline.py --baseline stub` does not need
        # diffusers / torch installed at all.
        from diffusers import WanVACEPipeline  # type: ignore
        from diffusers.utils import export_to_video  # noqa: F401

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                       "float32": torch.float32}[self.config["dtype"]]
        pipe = WanVACEPipeline.from_pretrained(
            self.config["model_id"], torch_dtype=self._dtype,
        )
        mode = self.config["offload_mode"]
        # Offload strategies, in decreasing memory pressure / increasing speed:
        #   sequential  — leaf-level swap. ~25 GB peak on a 97-frame 720p clip
        #                 (fits our 20 GB measurement w/ attention_slicing +
        #                 VAE tiling). ~3x slower per step vs. resident.
        #   group       — block-level swap on transformer + text_encoder
        #                 (both have discoverable ModuleLists: 40+8 and 24).
        #                 With use_stream=True the H2D copy of the next group
        #                 overlaps compute on the current one. VAE is only
        #                 ~260 MB so we keep it resident. Peak ≈ 2·G/N of the
        #                 transformer weights (34 GB / 40 blocks · N=4 groups
        #                 = ~7 GB) + text_encoder resident during prompt
        #                 encode (~11 GB, then swapped out) + activations.
        #                 Fits well under 48 GB, ~2x faster than sequential.
        #   model       — whole-submodule swap. Fastest per step but the
        #                 transformer forward peaks near the A6000 ceiling
        #                 and OOMs on the heavy scenes (curling_collision,
        #                 picnic_apple_ball, tennis_flight).
        #   none        — everything resident. OOMs at 14B / 720p.
        if mode == "sequential":
            pipe.enable_sequential_cpu_offload()
        elif mode == "group":
            import torch as _torch
            from diffusers.hooks import apply_group_offloading  # type: ignore
            onload = _torch.device(self._device)
            offload = _torch.device("cpu")
            n = int(self.config["num_blocks_per_group"])
            # diffusers 0.39: use_stream=True only supports num_blocks_per_group=1
            # (async prefetch is leaf-by-leaf). Larger groups need synchronous
            # swaps but move more weight per transfer. Let n pick the mode:
            #   n == 1  ->  streamed prefetch, best speed at min memory
            #   n >  1  ->  sync block swaps, fewer transfers, more memory
            use_stream = (n == 1)
            common = dict(
                onload_device=onload, offload_device=offload,
                offload_type="block_level", num_blocks_per_group=n,
                use_stream=use_stream,
                # Without this the hook pre-pins every weight tensor before the
                # first forward -- 46 GB of cudaHostRegister per shard, which
                # blocks setup for tens of minutes when 5 shards contend on the
                # same box. With it, pinning happens on-the-fly at H2D time:
                # slightly higher per-transfer cost, but startup is prompt and
                # peak host RAM per shard drops from ~46 GB to a few GB.
                low_cpu_mem_usage=True,
            )
            apply_group_offloading(pipe.transformer, **common)
            apply_group_offloading(pipe.text_encoder, **common)
            pipe.vae.to(self._device)
        elif mode == "model":
            pipe.enable_model_cpu_offload()
        else:  # "none"
            pipe = pipe.to(self._device)
        # VAE decode of a 97-frame 720p clip would otherwise be the peak
        # memory step; tiling + slicing take it out of the picture entirely.
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        # Recompute attention in smaller chunks -- shaves more off the
        # transformer forward, cheap insurance on top of sequential offload.
        pipe.enable_attention_slicing("auto")
        # Rebuild the scheduler with the resolution-appropriate flow_shift.
        from diffusers.schedulers.scheduling_unipc_multistep import (  # type: ignore
            UniPCMultistepScheduler,
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=self.config["flow_shift"],
        )

        self._pipe = pipe
        self._generator = torch.Generator(device=self._device).manual_seed(
            int(self.config["seed"])
        )
        print(f"[wan_vace] loaded {self.config['model_id']} on {self._device} "
              f"({self.config['dtype']}, flow_shift={self.config['flow_shift']}, "
              f"offload={self.config['offload_mode']})")

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
        from diffusers.utils import export_to_video, load_video  # type: ignore

        output_path.parent.mkdir(parents=True, exist_ok=True)
        source_frames = load_video(str(source_video))  # list[PIL.Image]

        # Pad the conditioning clip to a VACE-legal 4k+1 length by holding the
        # last frame, so conditioning and noise latents cover the same span.
        n_src = len(source_frames)
        _check_resolution(source_frames[0].height, source_frames[0].width,
                          self.config["height"], self.config["width"])
        n_gen = _vace_num_frames(n_src)
        cond_frames = source_frames + [source_frames[-1]] * (n_gen - n_src)

        # Drop the previous case's fragmented allocator state before the
        # pipe call -- with cpu_offload the transformer forward comes right
        # up to the A6000 ceiling and even a few hundred MiB of freed-but-
        # cached blocks can turn a 512 MiB alloc into an OOM.
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        t0 = time.perf_counter()
        out = self._pipe(
            prompt=prompt,
            negative_prompt=self.config["negative_prompt"],
            video=cond_frames,
            height=self.config["height"],
            width=self.config["width"],
            num_frames=n_gen,
            num_inference_steps=self.config["num_inference_steps"],
            guidance_scale=self.config["guidance_scale"],
            generator=self._generator,
        )
        # WanVACEPipeline returns .frames as list[list[PIL.Image]] (one clip
        # per batch element); we always run batch=1. Drop the padding so the
        # prediction lines up 1-for-1 with the ground truth.
        frames = out.frames[0][:n_src]
        # Match the source's frame rate so the edit lines up 1-for-1 with
        # the ground truth.
        source_fps = _probe_fps(source_video) or 24
        export_to_video(frames, str(output_path), fps=source_fps)
        elapsed = time.perf_counter() - t0

        return BaselineResult(
            output_video=output_path,
            elapsed_sec=elapsed,
            extra={
                "model_id": self.config["model_id"],
                "num_inference_steps": self.config["num_inference_steps"],
                "guidance_scale": self.config["guidance_scale"],
                "height": self.config["height"],
                "width": self.config["width"],
                "num_frames": len(frames),
                "flow_shift": self.config["flow_shift"],
                "num_frames_generated": n_gen,
                "num_frames_source": n_src,
                "seed": self.config["seed"],
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
