from __future__ import annotations

import argparse
import copy
import json
import math
import random
import shutil
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from incline_geometry import GROUNDED_WEDGE_QUADS, grounded_wedge_vertices


WORKSPACE_DIR = Path(__file__).resolve().parents[1]
POLYHAVEN_DIR = WORKSPACE_DIR / "assets" / "polyhaven"
AMBIENTCG_DIR = WORKSPACE_DIR / "assets" / "ambientcg"

OUTPUT_STEM = "ball_block_impact"
DIRECT_MP4_NAME = f"{OUTPUT_STEM}_blender_direct.mp4"
FINAL_MP4_NAME = f"{OUTPUT_STEM}.mp4"
TEMP_MP4_NAME = f"{OUTPUT_STEM}_tmp.mp4"
BLEND_NAME = f"{OUTPUT_STEM}.blend"
GROUND_TRUTH_NAME = "ground_truth_transforms.json"
PHYSICS_TEMP_NAME = "physics_transforms.json"
SCENARIO_METADATA_NAME = "scenario_metadata.json"

BALL_RADIUS = 0.34
SIDE_IMPACT_BALL_INITIAL_XY = (-3.15, -0.12)
WOOD_BLOCK_LOCATION = (0.23, -0.02, 0.35)
WOOD_BLOCK_DIMENSIONS = (0.92, 0.58, 0.70)
REALISM_PROFILE = "enhanced"
MOTION_CHOICES = (
    "side_impact",
    "drop_onto_block",
    "incline_slide_falloff",
    "wood_incline_slide_falloff",
    "wall_bounce",
)
DEFAULT_BLOCK_TEXTURE_ASSET = "wood_table"
INCLINE_RAMP_LOCATION = (-0.35, 0.0, 0.77)
INCLINE_RAMP_DIMENSIONS = (2.5, 1.1, 0.08)
INCLINE_RAMP_PITCH_DEG = 18.0

CAMERA_LOCATION = (3.9, -5.7, 2.35)
CAMERA_TARGET = (0.05, -0.05, 0.45)
LIGHT_TARGET = (0.0, -0.05, 0.45)

BALL_SEAMS = (
    ("ball_equator_seam", (0.0, 0.0, 0.0)),
    ("ball_vertical_seam", (math.radians(90.0), 0.0, 0.0)),
    ("ball_diagonal_seam", (0.0, math.radians(63.0), 0.0)),
)

BALL_SCUFF_PATCHES = (
    ((0.56, -0.62, 0.54), 0.033, 0.010, 0.2),
    ((0.38, -0.82, 0.42), 0.020, 0.008, 1.1),
    ((0.70, -0.35, 0.62), 0.024, 0.009, -0.7),
    ((0.18, -0.93, 0.30), 0.028, 0.008, 0.6),
    ((-0.30, -0.76, 0.58), 0.027, 0.010, -1.0),
    ((0.82, -0.16, 0.55), 0.019, 0.007, 1.4),
    ((-0.18, 0.50, 0.85), 0.025, 0.009, 0.3),
    ((0.62, 0.25, 0.74), 0.020, 0.008, -0.4),
)


@dataclass(frozen=True)
class PBRTextureSet:
    asset_name: str
    material_name: str
    diffuse_name: str
    roughness_name: str
    normal_name: str
    ao_name: str
    height_name: str
    normal_strength: float
    height_bump_strength: float
    height_bump_distance: float


@dataclass(frozen=True)
class MotionSetup:
    ball_initial_location: tuple[float, float, float]
    ball_initial_velocity: tuple[float, float, float]


@dataclass(frozen=True)
class BlockMotionSetup:
    block_location: tuple[float, float, float]
    block_pitch_deg: float
    block_yaw_deg: float
    block_initial_velocity: tuple[float, float, float]


FLOOR_TEXTURE = PBRTextureSet(
    asset_name="wood_floor_worn",
    material_name="4k pbr worn pine floor",
    diffuse_name="wood_floor_worn_diff_4k.jpg",
    roughness_name="wood_floor_worn_rough_4k.jpg",
    normal_name="wood_floor_worn_nor_gl_4k.jpg",
    ao_name="wood_floor_worn_ao_4k.jpg",
    height_name="wood_floor_worn_disp_4k.png",
    normal_strength=0.36,
    height_bump_strength=0.045,
    height_bump_distance=0.006,
)

BLOCK_TEXTURES = {
    "wood_table": PBRTextureSet(
        asset_name="wood_table",
        material_name="4k pbr scanned tabletop wood block",
        diffuse_name="wood_table_diff_4k.jpg",
        roughness_name="wood_table_rough_4k.jpg",
        normal_name="wood_table_nor_gl_4k.jpg",
        ao_name="wood_table_ao_4k.jpg",
        height_name="wood_table_disp_4k.png",
        normal_strength=0.34,
        height_bump_strength=0.080,
        height_bump_distance=0.010,
    ),
    "stained_pine": PBRTextureSet(
        asset_name="stained_pine",
        material_name="4k pbr stained pine wood block",
        diffuse_name="stained_pine_diff_4k.jpg",
        roughness_name="stained_pine_rough_4k.jpg",
        normal_name="stained_pine_nor_gl_4k.jpg",
        ao_name="stained_pine_ao_4k.jpg",
        height_name="stained_pine_disp_4k.png",
        normal_strength=0.28,
        height_bump_strength=0.055,
        height_bump_distance=0.008,
    ),
}


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preview", "animation"), default="animation")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resolution", nargs=2, type=int, default=(960, 540))
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-sec", type=float, default=8.0)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--preview-frame", type=int, default=86)
    parser.add_argument("--device", choices=("auto", "cpu"), default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--motion", choices=MOTION_CHOICES, default="side_impact")
    parser.add_argument(
        "--block-texture-asset",
        choices=tuple(BLOCK_TEXTURES),
        default=DEFAULT_BLOCK_TEXTURE_ASSET,
    )
    parser.add_argument("--drop-x-velocity", type=float, default=None)
    parser.add_argument("--drop-y-velocity", type=float, default=None)
    parser.add_argument("--physics-jitter", type=float, default=1.0)
    parser.add_argument("--camera-jitter", type=float, default=0.0)
    parser.add_argument("--surface-marks", choices=("none", "subtle", "full"), default="none")
    parser.add_argument(
        "--scenario-json",
        type=Path,
        default=None,
        help="Optional complete scenario_metadata.json to render instead of sampling one.",
    )
    parser.add_argument(
        "--scenario-overrides-json",
        type=Path,
        default=None,
        help="Optional JSON object recursively merged onto the sampled scenario.",
    )
    parser.add_argument("--video-postprocess", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def frame_count(args: argparse.Namespace) -> int:
    return max(2, int(round(float(args.duration_sec) * int(args.fps))))


def clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def jittered_color(
    rng: random.Random,
    color: tuple[float, float, float, float],
    amount: float,
) -> tuple[float, float, float, float]:
    return (
        clamp(color[0] + rng.uniform(-amount, amount), 0.0, 1.0),
        clamp(color[1] + rng.uniform(-amount, amount), 0.0, 1.0),
        clamp(color[2] + rng.uniform(-amount, amount), 0.0, 1.0),
        color[3],
    )


def create_side_impact_motion(
    rng: random.Random,
    jitter: float,
) -> MotionSetup:
    return MotionSetup(
        ball_initial_location=(
            SIDE_IMPACT_BALL_INITIAL_XY[0] + rng.uniform(-0.08, 0.10) * jitter,
            SIDE_IMPACT_BALL_INITIAL_XY[1] + rng.uniform(-0.10, 0.10) * jitter,
            BALL_RADIUS + 0.001,
        ),
        ball_initial_velocity=(
            6.0 + rng.uniform(-0.45, 0.55) * jitter,
            rng.uniform(-0.24, 0.24) * jitter,
            0.0,
        ),
    )


def create_drop_onto_block_motion(
    args: argparse.Namespace,
    rng: random.Random,
    jitter: float,
    block_location: tuple[float, float, float],
) -> MotionSetup:
    block_top_z = block_location[2] + WOOD_BLOCK_DIMENSIONS[2] * 0.5
    drop_x_velocity = (
        0.18 + rng.uniform(-0.04, 0.06) * jitter
        if args.drop_x_velocity is None
        else float(args.drop_x_velocity) + rng.uniform(-0.03, 0.03) * jitter
    )
    drop_y_velocity = (
        rng.uniform(-0.04, 0.04) * jitter
        if args.drop_y_velocity is None
        else float(args.drop_y_velocity) + rng.uniform(-0.02, 0.02) * jitter
    )
    ball_initial_location = (
        block_location[0] - 0.16 + rng.uniform(-0.04, 0.05) * jitter,
        block_location[1] + rng.uniform(-0.05, 0.05) * jitter,
        block_top_z + BALL_RADIUS + 1.32 + rng.uniform(-0.10, 0.10) * jitter,
    )
    ball_initial_velocity = (
        drop_x_velocity,
        drop_y_velocity,
        -0.18 + rng.uniform(-0.04, 0.02) * jitter,
    )
    return MotionSetup(
        ball_initial_location=ball_initial_location,
        ball_initial_velocity=ball_initial_velocity,
    )


def ramp_top_ball_center(
    *,
    local_x: float,
    local_y: float,
    radius: float,
    location: tuple[float, float, float],
    dimensions: tuple[float, float, float],
    pitch_deg: float,
) -> tuple[float, float, float]:
    pitch = math.radians(float(pitch_deg))
    half_thickness = 0.5 * float(dimensions[2])
    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)
    surface_point = (
        location[0] + cos_pitch * local_x + sin_pitch * half_thickness,
        location[1] + local_y,
        location[2] - sin_pitch * local_x + cos_pitch * half_thickness,
    )
    normal = (sin_pitch, 0.0, cos_pitch)
    return (
        surface_point[0] + radius * normal[0],
        surface_point[1] + radius * normal[1],
        surface_point[2] + radius * normal[2],
    )


def ramp_top_box_center(
    *,
    local_x: float,
    local_y: float,
    half_height: float,
    location: tuple[float, float, float],
    dimensions: tuple[float, float, float],
    pitch_deg: float,
) -> tuple[float, float, float]:
    pitch = math.radians(float(pitch_deg))
    half_thickness = 0.5 * float(dimensions[2])
    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)
    surface_point = (
        location[0] + cos_pitch * local_x + sin_pitch * half_thickness,
        location[1] + local_y,
        location[2] - sin_pitch * local_x + cos_pitch * half_thickness,
    )
    normal = (sin_pitch, 0.0, cos_pitch)
    return (
        surface_point[0] + half_height * normal[0],
        surface_point[1] + half_height * normal[1],
        surface_point[2] + half_height * normal[2],
    )


def create_incline_slide_falloff_motion(
    rng: random.Random,
    jitter: float,
) -> MotionSetup:
    local_x = -0.92 + rng.uniform(-0.04, 0.04) * jitter
    local_y = rng.uniform(-0.05, 0.05) * jitter
    ball_initial_location = ramp_top_ball_center(
        local_x=local_x,
        local_y=local_y,
        radius=BALL_RADIUS,
        location=INCLINE_RAMP_LOCATION,
        dimensions=INCLINE_RAMP_DIMENSIONS,
        pitch_deg=INCLINE_RAMP_PITCH_DEG,
    )
    ball_initial_velocity = (
        0.46 + rng.uniform(-0.05, 0.05) * jitter,
        rng.uniform(-0.03, 0.03) * jitter,
        -0.02,
    )
    return MotionSetup(
        ball_initial_location=ball_initial_location,
        ball_initial_velocity=ball_initial_velocity,
    )


def create_wood_incline_slide_falloff_motion(
    rng: random.Random,
    jitter: float,
) -> BlockMotionSetup:
    local_x = -1.35 + rng.uniform(-0.03, 0.03) * jitter
    local_y = rng.uniform(-0.04, 0.04) * jitter
    pitch_deg = INCLINE_RAMP_PITCH_DEG
    pitch = math.radians(pitch_deg)
    downslope_speed = 0.15 + rng.uniform(-0.03, 0.03) * jitter
    axis_u = (math.cos(pitch), 0.0, -math.sin(pitch))
    return BlockMotionSetup(
        block_location=ramp_top_box_center(
            local_x=local_x,
            local_y=local_y,
            half_height=0.5 * WOOD_BLOCK_DIMENSIONS[2],
            location=INCLINE_RAMP_LOCATION,
            dimensions=INCLINE_RAMP_DIMENSIONS,
            pitch_deg=pitch_deg,
        ),
        block_pitch_deg=pitch_deg,
        block_yaw_deg=0.0,
        block_initial_velocity=(
            downslope_speed * axis_u[0],
            0.0,
            downslope_speed * axis_u[2],
        ),
    )


def create_wall_bounce_motion(
    rng: random.Random,
    jitter: float,
) -> MotionSetup:
    return MotionSetup(
        ball_initial_location=(
            -0.85 + rng.uniform(-0.04, 0.04) * jitter,
            -1.32 + rng.uniform(-0.05, 0.05) * jitter,
            BALL_RADIUS + 0.001,
        ),
        ball_initial_velocity=(
            0.60 + rng.uniform(-0.04, 0.04) * jitter,
            4.10 + rng.uniform(-0.12, 0.12) * jitter,
            0.0,
        ),
    )


def create_side_impact_camera(rng: random.Random) -> dict[str, object]:
    base_location = [
        CAMERA_LOCATION[0] + rng.uniform(-0.18, 0.16),
        CAMERA_LOCATION[1] + rng.uniform(-0.18, 0.20),
        CAMERA_LOCATION[2] + rng.uniform(-0.08, 0.10),
    ]
    target = [
        CAMERA_TARGET[0] + rng.uniform(-0.08, 0.08),
        CAMERA_TARGET[1] + rng.uniform(-0.06, 0.07),
        CAMERA_TARGET[2] + rng.uniform(-0.02, 0.05),
    ]
    return {
        "base_location": base_location,
        "target": target,
        "lens_mm": 46.0 + rng.uniform(-3.0, 3.0),
        "sensor_width_mm": 32.0,
        "focus_distance": math.dist(base_location, target) + rng.uniform(-0.08, 0.08),
        "aperture_fstop": 6.2 + rng.uniform(-0.6, 0.8),
    }


def create_drop_onto_block_camera(
    rng: random.Random,
    block_location: tuple[float, float, float],
) -> dict[str, object]:
    base_location = [
        CAMERA_LOCATION[0] + rng.uniform(-0.12, 0.14),
        CAMERA_LOCATION[1] - 0.22 + rng.uniform(-0.14, 0.14),
        CAMERA_LOCATION[2] + 0.42 + rng.uniform(-0.06, 0.08),
    ]
    target = [
        block_location[0] - 0.05 + rng.uniform(-0.04, 0.04),
        block_location[1] + rng.uniform(-0.04, 0.04),
        0.88 + rng.uniform(-0.03, 0.04),
    ]
    return {
        "base_location": base_location,
        "target": target,
        "lens_mm": 41.0 + rng.uniform(-1.5, 1.8),
        "sensor_width_mm": 32.0,
        "focus_distance": math.dist(base_location, target) + rng.uniform(-0.05, 0.05),
        "aperture_fstop": 7.0 + rng.uniform(-0.4, 0.6),
    }


def create_incline_slide_falloff_camera(rng: random.Random) -> dict[str, object]:
    base_location = [
        3.10 + rng.uniform(-0.08, 0.10),
        -4.80 + rng.uniform(-0.12, 0.10),
        2.15 + rng.uniform(-0.05, 0.07),
    ]
    target = [
        -0.25 + rng.uniform(-0.04, 0.04),
        0.00 + rng.uniform(-0.04, 0.04),
        0.82 + rng.uniform(-0.03, 0.03),
    ]
    return {
        "base_location": base_location,
        "target": target,
        "lens_mm": 43.0 + rng.uniform(-1.5, 1.5),
        "sensor_width_mm": 32.0,
        "focus_distance": math.dist(base_location, target) + rng.uniform(-0.05, 0.05),
        "aperture_fstop": 7.4 + rng.uniform(-0.3, 0.5),
    }


def create_wall_bounce_camera(rng: random.Random) -> dict[str, object]:
    base_location = [
        3.20 + rng.uniform(-0.08, 0.10),
        -4.65 + rng.uniform(-0.10, 0.10),
        2.05 + rng.uniform(-0.04, 0.06),
    ]
    target = [
        -0.25 + rng.uniform(-0.04, 0.04),
        1.00 + rng.uniform(-0.05, 0.05),
        0.58 + rng.uniform(-0.03, 0.03),
    ]
    return {
        "base_location": base_location,
        "target": target,
        "lens_mm": 42.0 + rng.uniform(-1.2, 1.4),
        "sensor_width_mm": 32.0,
        "focus_distance": math.dist(base_location, target) + rng.uniform(-0.04, 0.05),
        "aperture_fstop": 7.2 + rng.uniform(-0.3, 0.5),
    }


def create_scenario(args: argparse.Namespace) -> dict[str, object]:
    seed = int(args.seed)
    rng = random.Random(seed)
    jitter = max(0.0, float(args.physics_jitter))
    motion = str(args.motion)

    block_location = (
        WOOD_BLOCK_LOCATION[0] + rng.uniform(-0.08, 0.10) * jitter,
        WOOD_BLOCK_LOCATION[1] + rng.uniform(-0.07, 0.08) * jitter,
        WOOD_BLOCK_LOCATION[2],
    )
    if motion == "side_impact":
        motion_settings = create_side_impact_motion(rng, jitter)
        camera_settings = create_side_impact_camera(rng)
        ball_enabled = True
        block_enabled = True
        block_pitch_deg = 0.0
        block_yaw_deg = rng.uniform(-3.5, 3.5) * jitter
        block_initial_velocity = (0.0, 0.0, 0.0)
    elif motion == "drop_onto_block":
        motion_settings = create_drop_onto_block_motion(args, rng, jitter, block_location)
        camera_settings = create_drop_onto_block_camera(rng, block_location)
        ball_enabled = True
        block_enabled = True
        block_pitch_deg = 0.0
        block_yaw_deg = rng.uniform(-3.5, 3.5) * jitter
        block_initial_velocity = (0.0, 0.0, 0.0)
    elif motion == "incline_slide_falloff":
        block_location = (0.20, 1.05, WOOD_BLOCK_LOCATION[2])
        motion_settings = create_incline_slide_falloff_motion(rng, jitter)
        camera_settings = create_incline_slide_falloff_camera(rng)
        ball_enabled = True
        block_enabled = True
        block_pitch_deg = 0.0
        block_yaw_deg = rng.uniform(-3.5, 3.5) * jitter
        block_initial_velocity = (0.0, 0.0, 0.0)
    elif motion == "wood_incline_slide_falloff":
        wood_motion = create_wood_incline_slide_falloff_motion(rng, jitter)
        block_location = wood_motion.block_location
        motion_settings = MotionSetup(
            ball_initial_location=(0.0, -10.0, BALL_RADIUS + 0.001),
            ball_initial_velocity=(0.0, 0.0, 0.0),
        )
        camera_settings = create_incline_slide_falloff_camera(rng)
        ball_enabled = False
        block_enabled = True
        block_pitch_deg = wood_motion.block_pitch_deg
        block_yaw_deg = wood_motion.block_yaw_deg
        block_initial_velocity = wood_motion.block_initial_velocity
    elif motion == "wall_bounce":
        block_location = (0.20, 1.05, WOOD_BLOCK_LOCATION[2])
        motion_settings = create_wall_bounce_motion(rng, jitter)
        camera_settings = create_wall_bounce_camera(rng)
        ball_enabled = True
        block_enabled = False
        block_pitch_deg = 0.0
        block_yaw_deg = 0.0
        block_initial_velocity = (0.0, 0.0, 0.0)
    else:
        raise ValueError(f"Unsupported motion: {motion}")

    extra_scuffs = []
    for _ in range(8):
        z = rng.uniform(-0.25, 0.92)
        theta = rng.uniform(0.0, 2.0 * math.pi)
        r = math.sqrt(max(0.0, 1.0 - z * z))
        extra_scuffs.append(
            {
                "normal": [r * math.cos(theta), r * math.sin(theta), z],
                "width": rng.uniform(0.010, 0.034),
                "height": rng.uniform(0.004, 0.012),
                "rotation": rng.uniform(-math.pi, math.pi),
            }
        )

    scenario = {
        "schema_version": 1,
        "seed": seed,
        "realism_profile": REALISM_PROFILE,
        "motion": motion,
        "active_objects": [
            *(['ball'] if ball_enabled else []),
            *(['wood_block'] if block_enabled else []),
        ],
        "surface_marks": args.surface_marks,
        "physics_jitter": float(args.physics_jitter),
        "camera_jitter": float(args.camera_jitter),
        "video_postprocess": bool(args.video_postprocess),
        "physics": {
            "motion": motion,
            "ball_enabled": ball_enabled,
            "block_enabled": block_enabled,
            "ball_radius": BALL_RADIUS,
            "ball_initial_location": list(motion_settings.ball_initial_location),
            "ball_initial_velocity": list(motion_settings.ball_initial_velocity),
            "block_location": list(block_location),
            "block_yaw_deg": block_yaw_deg,
            "block_pitch_deg": block_pitch_deg,
            "block_initial_velocity": list(block_initial_velocity),
            "ball_mass": 0.58 + rng.uniform(-0.05, 0.05) * jitter,
            "block_mass": 0.65 + rng.uniform(-0.08, 0.09) * jitter,
            "floor_friction": 0.82 + rng.uniform(-0.08, 0.05) * jitter,
            "ball_friction": 0.38 + rng.uniform(-0.06, 0.06) * jitter,
            "ball_restitution": 0.78 + rng.uniform(-0.06, 0.04) * jitter,
            "block_friction": 0.32 + rng.uniform(-0.05, 0.07) * jitter,
            "block_restitution": 0.55 + rng.uniform(-0.06, 0.05) * jitter,
            "ramp_enabled": motion in {"incline_slide_falloff", "wood_incline_slide_falloff"},
            "ramp_location": list(INCLINE_RAMP_LOCATION),
            "ramp_dimensions": list(INCLINE_RAMP_DIMENSIONS),
            "ramp_pitch_deg": INCLINE_RAMP_PITCH_DEG,
            "ramp_friction": 0.58,
            "ramp_restitution": 0.05,
            "ramp_profile": "tilted_slab",
            "wall_enabled": motion == "wall_bounce",
            "wall_location": [0.0, 3.05, 1.45],
            "wall_dimensions": [8.6, 0.08, 2.90],
            "wall_friction": 0.34,
            "wall_restitution": 0.82,
        },
        "render": {
            "exposure": rng.uniform(-0.08, 0.05),
            "gamma": 1.0,
            "motion_blur_shutter": 0.30 + rng.uniform(-0.035, 0.035),
            "postprocess_noise_strength": rng.randint(2, 3),
            "postprocess_contrast": 1.012 + rng.uniform(0.0, 0.012),
            "postprocess_saturation": 1.012 + rng.uniform(0.0, 0.018),
            "postprocess_gamma": 0.998 + rng.uniform(-0.004, 0.006),
            "lens_correction_k1": rng.uniform(-0.018, -0.010),
            "lens_correction_k2": rng.uniform(0.003, 0.008),
            "chromatic_shift_px": 0,
            "vignette_angle": rng.uniform(0.34, 0.40),
        },
        "camera": camera_settings,
        "lighting": {
            "hdri_rotation_deg": 128.0 + rng.uniform(-18.0, 20.0),
            "hdri_strength": 0.30 + rng.uniform(-0.04, 0.05),
            "key_power": 285.0 + rng.uniform(-35.0, 45.0),
            "key_size": 5.1 + rng.uniform(-0.45, 0.45),
            "bounce_power": 34.0 + rng.uniform(-7.0, 8.0),
            "wall_fill_power": 12.0 + rng.uniform(-3.5, 4.0),
            "key_color": list(jittered_color(rng, (0.95, 0.98, 1.0, 1.0), 0.018)),
            "bounce_color": list(jittered_color(rng, (1.0, 0.78, 0.55, 1.0), 0.025)),
            "fill_color": list(jittered_color(rng, (0.82, 0.86, 0.90, 1.0), 0.020)),
        },
        "materials": {
            "floor_texture_asset": "wood_floor_worn",
            "block_texture_asset": str(args.block_texture_asset),
            "ball_texture_asset": "Rubber002",
            "floor_texture_scale": 48.0 + rng.uniform(-3.0, 3.0),
            "block_texture_scale": 0.95 + rng.uniform(-0.08, 0.08),
            "ball_texture_scale": 3.8 + rng.uniform(-0.30, 0.35),
            "floor_tint": list(jittered_color(rng, (0.52, 0.34, 0.20, 1.0), 0.035)),
            "block_tint": list(jittered_color(rng, (0.64, 0.42, 0.24, 1.0), 0.045)),
            "wall_color": list(jittered_color(rng, (0.70, 0.66, 0.60, 1.0), 0.035)),
            "baseboard_color": list(jittered_color(rng, (0.78, 0.74, 0.68, 1.0), 0.025)),
            "ball_base_color": list(jittered_color(rng, (0.72, 0.075, 0.025, 1.0), 0.035)),
            "ramp_color": list(jittered_color(rng, (0.54, 0.42, 0.30, 1.0), 0.030)),
        },
        "ball_scuffs": extra_scuffs,
    }
    return scenario


def read_json(path: Path) -> dict[str, object]:
    data = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def recursive_update(
    base: dict[str, object],
    updates: dict[str, object],
) -> dict[str, object]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = recursive_update(
                merged[key],  # type: ignore[arg-type]
                value,
            )
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def scenario_from_args(args: argparse.Namespace) -> dict[str, object]:
    if args.scenario_json is not None:
        scenario = read_json(args.scenario_json)
        scenario.setdefault(
            "scenario_source",
            str(args.scenario_json.expanduser().resolve()),
        )
    else:
        scenario = create_scenario(args)

    if args.scenario_overrides_json is not None:
        overrides = read_json(args.scenario_overrides_json)
        scenario = recursive_update(scenario, overrides)
        scenario["scenario_overrides_path"] = str(
            args.scenario_overrides_json.expanduser().resolve()
        )
    return scenario


def ball_initial_location(scenario: dict[str, object]) -> tuple[float, float, float]:
    physics = scenario["physics"]
    assert isinstance(physics, dict)
    return tuple(float(value) for value in physics["ball_initial_location"])


def ball_enabled(scenario: dict[str, object]) -> bool:
    physics = scenario.get("physics")
    return not isinstance(physics, dict) or bool(physics.get("ball_enabled", True))


def block_enabled(scenario: dict[str, object]) -> bool:
    physics = scenario.get("physics")
    return not isinstance(physics, dict) or bool(physics.get("block_enabled", True))


def write_scenario_metadata(out_dir: Path, scenario: dict[str, object]) -> None:
    (out_dir / SCENARIO_METADATA_NAME).write_text(
        json.dumps(scenario, indent=2),
        encoding="utf-8",
    )


def output_path(out_dir: Path, filename: str) -> Path:
    return (out_dir / filename).resolve()


def require_polyhaven_path(asset_name: str, filename: str) -> Path:
    path = POLYHAVEN_DIR / asset_name / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing render asset: {path}. Run scripts/download_render_assets.py first."
        )
    return path


def require_ambientcg_path(asset_name: str, filename: str) -> Path:
    path = AMBIENTCG_DIR / asset_name / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing render asset: {path}. Run scripts/download_render_assets.py first."
        )
    return path


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for collection in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.textures,
        bpy.data.images,
        bpy.data.lights,
        bpy.data.cameras,
    ):
        for item in list(collection):
            collection.remove(item)


def configure_cycles_device(scene: bpy.types.Scene, mode: str) -> None:
    if mode != "auto":
        scene.cycles.device = "CPU"
        return

    scene.cycles.device = "CPU"
    try:
        preferences = bpy.context.preferences.addons["cycles"].preferences
        for device_type in ("OPTIX", "CUDA"):
            try:
                preferences.compute_device_type = device_type
                preferences.get_devices()
                for device in preferences.devices:
                    device.use = "CPU" not in device.name.upper()
                scene.cycles.device = "GPU"
                return
            except Exception:
                continue
    except Exception:
        return


def configure_ffmpeg(scene: bpy.types.Scene, out_dir: Path) -> None:
    scene.render.filepath = str(output_path(out_dir, DIRECT_MP4_NAME))
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    scene.render.ffmpeg.gopsize = 12


def setup_render(args: argparse.Namespace, scenario: dict[str, object]) -> None:
    scene = bpy.context.scene
    render = scenario["render"]
    assert isinstance(render, dict)
    scene.frame_start = 1
    scene.frame_end = frame_count(args)
    scene.frame_set(1)
    scene.render.fps = int(args.fps)
    scene.render.resolution_x = int(args.resolution[0])
    scene.render.resolution_y = int(args.resolution[1])
    scene.render.resolution_percentage = 100
    scene.render.engine = "CYCLES"
    scene.cycles.samples = int(args.samples)
    scene.cycles.preview_samples = min(32, int(args.samples))
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 7
    scene.cycles.diffuse_bounces = 3
    scene.cycles.glossy_bounces = 3
    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = float(render["motion_blur_shutter"])
    try:
        scene.view_settings.view_transform = "Filmic"
        scene.view_settings.look = "Medium High Contrast"
    except TypeError:
        scene.view_settings.view_transform = "Standard"
    scene.view_settings.exposure = float(render["exposure"])
    scene.view_settings.gamma = float(render["gamma"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    configure_cycles_device(scene, args.device)
    configure_ffmpeg(scene, args.out_dir)


def write_compatible_mp4(out_dir: Path, fps: int, scenario: dict[str, object]) -> None:
    direct_path = out_dir / DIRECT_MP4_NAME
    final_path = out_dir / FINAL_MP4_NAME
    temp_path = out_dir / TEMP_MP4_NAME
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        direct_path.replace(final_path)
        return

    render = scenario["render"]
    assert isinstance(render, dict)
    video_filter = None
    if bool(scenario.get("video_postprocess", True)):
        noise_strength = int(render.get("postprocess_noise_strength", 3))
        filters = [
            (
                "lenscorrection="
                f"cx=0.5:cy=0.5:k1={float(render.get('lens_correction_k1', 0.0)):.5f}:"
                f"k2={float(render.get('lens_correction_k2', 0.0)):.5f}:i=1"
            )
        ]
        chromatic_shift_px = int(render.get("chromatic_shift_px", 0))
        if chromatic_shift_px:
            filters.append(f"chromashift=cbh={chromatic_shift_px}:crh={-chromatic_shift_px}:edge=smear")
        filters.extend(
            [
                f"noise=alls={noise_strength}:allf=t+u",
                f"vignette={float(render.get('vignette_angle', math.pi / 8.0)):.5f}:eval=frame",
                (
                    "eq="
                    f"contrast={float(render.get('postprocess_contrast', 1.015)):.4f}:"
                    f"saturation={float(render.get('postprocess_saturation', 1.025)):.4f}:"
                    f"gamma={float(render.get('postprocess_gamma', 1.0)):.4f}"
                ),
                "unsharp=3:3:0.18:3:3:0.02",
            ]
        )
        video_filter = ",".join(filters)

    command = [
        ffmpeg,
        "-y",
        "-i",
        str(direct_path),
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
    ]
    if video_filter:
        command.extend(["-vf", video_filter])
    command.extend(
        [
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "18",
        "-profile:v",
        "baseline",
        "-level",
        "3.1",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(int(fps)),
        "-vsync",
        "cfr",
        "-bf",
        "0",
        "-refs",
        "1",
        "-x264-params",
        f"keyint={int(fps)}:min-keyint={int(fps)}:scenecut=0",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-shortest",
        "-movflags",
        "+faststart",
        "-brand",
        "mp42",
        "-video_track_timescale",
        str(int(fps) * 1000),
        str(temp_path),
        ]
    )
    try:
        subprocess.run(command, check=True)
        temp_path.replace(final_path)
        direct_path.unlink(missing_ok=True)
    except Exception:
        temp_path.unlink(missing_ok=True)
        direct_path.replace(final_path)


def look_at(obj: bpy.types.Object, target: tuple[float, float, float]) -> None:
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_camera(scenario: dict[str, object]) -> bpy.types.Object:
    camera_settings = scenario["camera"]
    assert isinstance(camera_settings, dict)
    location = tuple(float(value) for value in camera_settings["base_location"])
    target = tuple(float(value) for value in camera_settings["target"])
    bpy.ops.object.camera_add(location=location)
    camera = bpy.context.object
    look_at(camera, target)
    camera.data.lens = float(camera_settings["lens_mm"])
    camera.data.sensor_width = float(camera_settings["sensor_width_mm"])
    camera.data.dof.use_dof = True
    camera.data.dof.focus_distance = float(camera_settings["focus_distance"])
    camera.data.dof.aperture_fstop = float(camera_settings["aperture_fstop"])
    try:
        camera.data.dof.aperture_blades = 7
    except Exception:
        pass
    bpy.context.scene.camera = camera
    apply_camera_jitter(camera, scenario)
    return camera


def apply_camera_jitter(camera: bpy.types.Object, scenario: dict[str, object]) -> None:
    scene = bpy.context.scene
    strength = max(0.0, float(scenario.get("camera_jitter", 0.0)))
    if strength <= 0.0 or scene.frame_end <= scene.frame_start:
        return

    camera_settings = scenario["camera"]
    assert isinstance(camera_settings, dict)
    base_location = Vector(camera_settings["base_location"])
    base_target = Vector(camera_settings["target"])
    rng = random.Random(int(scenario["seed"]) + 303)
    location_offset = Vector((0.0, 0.0, 0.0))
    target_offset = Vector((0.0, 0.0, 0.0))
    for frame in range(scene.frame_start, scene.frame_end + 1):
        location_offset = location_offset * 0.82 + Vector(
            (
                rng.uniform(-strength, strength),
                rng.uniform(-strength, strength),
                rng.uniform(-strength * 0.45, strength * 0.45),
            )
        )
        target_offset = target_offset * 0.86 + Vector(
            (
                rng.uniform(-strength * 0.45, strength * 0.45),
                rng.uniform(-strength * 0.45, strength * 0.45),
                rng.uniform(-strength * 0.22, strength * 0.22),
            )
        )
        camera.location = base_location + location_offset
        look_at(camera, tuple(base_target + target_offset))
        camera.keyframe_insert(data_path="location", frame=frame)
        camera.keyframe_insert(data_path="rotation_euler", frame=frame)

    scene.frame_set(scene.frame_start)
    camera.location = base_location
    look_at(camera, tuple(base_target))


def add_area_light(
    name: str,
    location: tuple[float, float, float],
    power: float,
    size: float,
    target: tuple[float, float, float],
    color: tuple[float, float, float, float] | None = None,
) -> bpy.types.Object:
    bpy.ops.object.light_add(type="AREA", location=location)
    light = bpy.context.object
    light.name = name
    if hasattr(light.data, "power"):
        light.data.power = float(power)
    else:
        light.data.energy = float(power)
    light.data.size = float(size)
    if color is not None and hasattr(light.data, "color"):
        light.data.color = tuple(float(value) for value in color[:3])
    look_at(light, target)
    return light


def set_input_default(node: bpy.types.Node, input_name: str, value: object) -> None:
    if input_name in node.inputs:
        node.inputs[input_name].default_value = value


def add_noise_node(
    nodes: bpy.types.Nodes,
    scale: float,
    detail: float,
    roughness: float,
) -> bpy.types.Node:
    noise = nodes.new(type="ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = float(scale)
    noise.inputs["Detail"].default_value = float(detail)
    noise.inputs["Roughness"].default_value = float(roughness)
    return noise


def add_color_ramp(
    nodes: bpy.types.Nodes,
    left_position: float,
    left_color: tuple[float, float, float, float],
    right_position: float,
    right_color: tuple[float, float, float, float],
) -> bpy.types.Node:
    ramp = nodes.new(type="ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = float(left_position)
    ramp.color_ramp.elements[0].color = left_color
    ramp.color_ramp.elements[1].position = float(right_position)
    ramp.color_ramp.elements[1].color = right_color
    return ramp


def add_noise_bump(
    material: bpy.types.Material,
    bsdf: bpy.types.Node,
    strength: float,
    distance: float,
    scale: float,
    detail: float,
    roughness: float,
) -> None:
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    noise = add_noise_node(nodes, scale, detail, roughness)
    bump = nodes.new(type="ShaderNodeBump")
    bump.inputs["Strength"].default_value = float(strength)
    bump.inputs["Distance"].default_value = float(distance)
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])


def create_principled_material(
    name: str,
    base_color: tuple[float, float, float, float],
    roughness: float,
    metallic: float = 0.0,
    noise_bump: float = 0.0,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf is not None:
        set_input_default(bsdf, "Base Color", base_color)
        set_input_default(bsdf, "Roughness", float(roughness))
        set_input_default(bsdf, "Metallic", float(metallic))
        if noise_bump > 0.0:
            add_noise_bump(mat, bsdf, noise_bump, 0.055, 55.0, 9.0, 0.62)
    return mat


def load_image(path: Path, colorspace: str) -> bpy.types.Image:
    image = bpy.data.images.load(str(path.resolve()))
    try:
        image.colorspace_settings.name = colorspace
    except Exception:
        pass
    return image


def add_texture_mapping(
    nodes: bpy.types.Nodes,
    links: bpy.types.NodeLinks,
    scale: tuple[float, float, float],
    coordinate_output: str = "Generated",
) -> bpy.types.Node:
    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    mapping = nodes.new(type="ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = scale
    if coordinate_output not in tex_coord.outputs:
        coordinate_output = "Generated"
    links.new(tex_coord.outputs[coordinate_output], mapping.inputs["Vector"])
    return mapping


def add_mapped_image_texture(
    nodes: bpy.types.Nodes,
    links: bpy.types.NodeLinks,
    mapping: bpy.types.Node,
    image_path: Path,
    colorspace: str,
) -> bpy.types.Node:
    texture = nodes.new(type="ShaderNodeTexImage")
    texture.image = load_image(image_path, colorspace)
    texture.extension = "REPEAT"
    links.new(mapping.outputs["Vector"], texture.inputs["Vector"])
    return texture


def create_pbr_material(
    name: str,
    base_color: tuple[float, float, float, float],
    roughness: float,
    diffuse_path: Path,
    roughness_path: Path,
    normal_path: Path,
    texture_scale: tuple[float, float, float],
    normal_strength: float,
    ao_path: Path | None = None,
    height_path: Path | None = None,
    ao_strength: float = 0.0,
    height_bump_strength: float = 0.0,
    height_bump_distance: float = 0.006,
    coordinate_output: str = "Generated",
    tint_strength: float = 0.0,
    detail_bump_strength: float = 0.0,
    detail_bump_distance: float = 0.004,
    detail_bump_scale: float = 90.0,
) -> bpy.types.Material:
    mat = create_principled_material(name, base_color, roughness)
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        return mat

    mapping = add_texture_mapping(nodes, links, texture_scale, coordinate_output)

    diffuse = add_mapped_image_texture(nodes, links, mapping, diffuse_path, "sRGB")
    if tint_strength > 0.0:
        tint = nodes.new(type="ShaderNodeMixRGB")
        tint.blend_type = "MULTIPLY"
        tint.inputs["Fac"].default_value = float(tint_strength)
        tint.inputs["Color2"].default_value = base_color
        links.new(diffuse.outputs["Color"], tint.inputs["Color1"])
        base_color_output = tint.outputs["Color"]
    else:
        base_color_output = diffuse.outputs["Color"]

    if ao_path is not None and ao_strength > 0.0:
        ao = add_mapped_image_texture(nodes, links, mapping, ao_path, "Non-Color")
        ao_mix = nodes.new(type="ShaderNodeMixRGB")
        ao_mix.blend_type = "MULTIPLY"
        ao_mix.inputs["Fac"].default_value = float(ao_strength)
        links.new(base_color_output, ao_mix.inputs["Color1"])
        links.new(ao.outputs["Color"], ao_mix.inputs["Color2"])
        base_color_output = ao_mix.outputs["Color"]
    links.new(base_color_output, bsdf.inputs["Base Color"])

    rough = add_mapped_image_texture(nodes, links, mapping, roughness_path, "Non-Color")
    links.new(rough.outputs["Color"], bsdf.inputs["Roughness"])

    normal = add_mapped_image_texture(nodes, links, mapping, normal_path, "Non-Color")
    normal_map = nodes.new(type="ShaderNodeNormalMap")
    normal_map.inputs["Strength"].default_value = float(normal_strength)
    links.new(normal.outputs["Color"], normal_map.inputs["Color"])

    normal_output = normal_map.outputs["Normal"]
    if height_path is not None and height_bump_strength > 0.0:
        height = add_mapped_image_texture(nodes, links, mapping, height_path, "Non-Color")
        height_bump = nodes.new(type="ShaderNodeBump")
        height_bump.inputs["Strength"].default_value = float(height_bump_strength)
        height_bump.inputs["Distance"].default_value = float(height_bump_distance)
        links.new(height.outputs["Color"], height_bump.inputs["Height"])
        links.new(normal_output, height_bump.inputs["Normal"])
        normal_output = height_bump.outputs["Normal"]

    if detail_bump_strength > 0.0:
        detail_noise = add_noise_node(nodes, detail_bump_scale, 11.0, 0.62)
        detail_bump = nodes.new(type="ShaderNodeBump")
        detail_bump.inputs["Strength"].default_value = float(detail_bump_strength)
        detail_bump.inputs["Distance"].default_value = float(detail_bump_distance)
        links.new(detail_noise.outputs["Fac"], detail_bump.inputs["Height"])
        links.new(normal_output, detail_bump.inputs["Normal"])
        links.new(detail_bump.outputs["Normal"], bsdf.inputs["Normal"])
    else:
        links.new(normal_output, bsdf.inputs["Normal"])
    return mat


def scenario_color(
    scenario: dict[str, object],
    name: str,
    default_color: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    materials = scenario.get("materials", {})
    if not isinstance(materials, dict) or name not in materials:
        return default_color
    values = materials[name]
    return tuple(float(value) for value in values)


def scenario_material_float(
    scenario: dict[str, object],
    name: str,
    default_value: float,
) -> float:
    materials = scenario.get("materials", {})
    if not isinstance(materials, dict):
        return float(default_value)
    return float(materials.get(name, default_value))


def create_wood_material(scenario: dict[str, object]) -> bpy.types.Material:
    texture_scale = scenario_material_float(scenario, "block_texture_scale", 1.10)
    materials = scenario.get("materials", {})
    if not isinstance(materials, dict):
        raise ValueError("Scenario materials must be a dictionary.")
    block_texture_name = str(materials["block_texture_asset"])
    texture = BLOCK_TEXTURES[block_texture_name]
    return create_pbr_material(
        texture.material_name,
        scenario_color(scenario, "block_tint", (0.64, 0.42, 0.24, 1.0)),
        0.54,
        require_polyhaven_path(texture.asset_name, texture.diffuse_name),
        require_polyhaven_path(texture.asset_name, texture.roughness_name),
        require_polyhaven_path(texture.asset_name, texture.normal_name),
        (texture_scale * 1.16, texture_scale, 1.0),
        texture.normal_strength,
        ao_path=require_polyhaven_path(texture.asset_name, texture.ao_name),
        height_path=require_polyhaven_path(texture.asset_name, texture.height_name),
        ao_strength=0.24,
        height_bump_strength=texture.height_bump_strength,
        height_bump_distance=texture.height_bump_distance,
        coordinate_output="UV",
        tint_strength=0.08,
        detail_bump_strength=0.014,
        detail_bump_distance=0.0032,
        detail_bump_scale=150.0,
    )


def create_hardwood_floor_material(scenario: dict[str, object]) -> bpy.types.Material:
    scale = scenario_material_float(scenario, "floor_texture_scale", 48.0)
    return create_pbr_material(
        FLOOR_TEXTURE.material_name,
        scenario_color(scenario, "floor_tint", (0.52, 0.34, 0.20, 1.0)),
        0.55,
        require_polyhaven_path(FLOOR_TEXTURE.asset_name, FLOOR_TEXTURE.diffuse_name),
        require_polyhaven_path(FLOOR_TEXTURE.asset_name, FLOOR_TEXTURE.roughness_name),
        require_polyhaven_path(FLOOR_TEXTURE.asset_name, FLOOR_TEXTURE.normal_name),
        (scale, scale, 1.0),
        FLOOR_TEXTURE.normal_strength,
        ao_path=require_polyhaven_path(FLOOR_TEXTURE.asset_name, FLOOR_TEXTURE.ao_name),
        height_path=require_polyhaven_path(FLOOR_TEXTURE.asset_name, FLOOR_TEXTURE.height_name),
        ao_strength=0.18,
        height_bump_strength=FLOOR_TEXTURE.height_bump_strength,
        height_bump_distance=FLOOR_TEXTURE.height_bump_distance,
        tint_strength=0.045,
        detail_bump_strength=0.008,
        detail_bump_distance=0.0018,
        detail_bump_scale=180.0,
    )


def create_wall_material(scenario: dict[str, object]) -> bpy.types.Material:
    return create_principled_material(
        "slightly uneven painted plaster",
        scenario_color(scenario, "wall_color", (0.70, 0.66, 0.60, 1.0)),
        roughness=0.88,
        noise_bump=0.014,
    )


def create_baseboard_material(scenario: dict[str, object]) -> bpy.types.Material:
    return create_principled_material(
        "painted off white baseboard",
        scenario_color(scenario, "baseboard_color", (0.78, 0.74, 0.68, 1.0)),
        roughness=0.66,
        noise_bump=0.006,
    )


def cube_project_uvs(obj: bpy.types.Object, cube_size: float = 1.0) -> None:
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.cube_project(cube_size=float(cube_size), correct_aspect=True)
    bpy.ops.object.mode_set(mode="OBJECT")


def add_box(
    name: str,
    location: tuple[float, float, float],
    dimensions: tuple[float, float, float],
    material: bpy.types.Material,
    *,
    rotation_z: float = 0.0,
    rotation_euler: tuple[float, float, float] | None = None,
    bevel_width: float = 0.0,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(
        size=1.0,
        location=location,
        rotation=rotation_euler if rotation_euler is not None else (0.0, 0.0, rotation_z),
    )
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    if bevel_width > 0.0:
        obj.data.use_auto_smooth = True
        bevel = obj.modifiers.new("soft real-world edges", "BEVEL")
        bevel.width = float(bevel_width)
        bevel.segments = 2
        obj.modifiers.new("weighted normals", "WEIGHTED_NORMAL")
    return obj


def add_cylinder(
    name: str,
    location: tuple[float, float, float],
    radius: float,
    depth: float,
    material: bpy.types.Material,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=24,
        radius=float(radius),
        depth=float(depth),
        location=location,
    )
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material)
    bpy.ops.object.shade_smooth()
    return obj


def setup_world_lighting(scenario: dict[str, object]) -> None:
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    hdri_path = require_polyhaven_path("brown_photostudio_05", "brown_photostudio_05_2k.hdr")
    lighting = scenario["lighting"]
    assert isinstance(lighting, dict)

    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputWorld")
    background = nodes.new(type="ShaderNodeBackground")
    environment = nodes.new(type="ShaderNodeTexEnvironment")
    mapping = nodes.new(type="ShaderNodeMapping")
    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    environment.image = load_image(hdri_path, "Linear")
    mapping.inputs["Rotation"].default_value[2] = math.radians(float(lighting["hdri_rotation_deg"]))
    background.inputs["Strength"].default_value = float(lighting["hdri_strength"])
    links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], environment.inputs["Vector"])
    links.new(environment.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], output.inputs["Surface"])


def add_room_shell(scenario: dict[str, object]) -> None:
    wall_mat = create_wall_material(scenario)
    baseboard_mat = create_baseboard_material(scenario)
    add_box("matte back plaster wall", (0.0, 3.05, 1.45), (8.6, 0.08, 2.90), wall_mat)
    add_box("matte right plaster wall", (4.20, -0.55, 1.45), (0.08, 7.2, 2.90), wall_mat)
    add_box("back wall baseboard", (0.0, 2.995, 0.13), (8.5, 0.09, 0.16), baseboard_mat, bevel_width=0.006)
    add_box("right wall baseboard", (4.145, -0.55, 0.13), (0.09, 7.1, 0.16), baseboard_mat, bevel_width=0.006)


def add_floor_details(scenario: dict[str, object]) -> None:
    surface_marks = str(scenario.get("surface_marks", "none"))
    if surface_marks == "none":
        return

    rng = random.Random(int(scenario["seed"]) + 101)
    scratch_count = 14 if surface_marks == "subtle" else 46
    dust_count = 8 if surface_marks == "subtle" else 28
    scratch_light_color = (0.38, 0.32, 0.25, 1.0) if surface_marks == "subtle" else (0.66, 0.59, 0.49, 1.0)
    scratch_light = create_principled_material("fine muted floor scratches", scratch_light_color, 0.94)
    scratch_dark = create_principled_material("dark dust in floor scratches", (0.21, 0.17, 0.13, 1.0), 0.96)
    dust_mat = create_principled_material("small muted floor dust flecks", (0.34, 0.31, 0.26, 1.0), 0.98)

    for idx in range(scratch_count):
        x = rng.uniform(-2.8, 3.4)
        y = rng.uniform(-2.0, 2.5)
        length = rng.uniform(0.04, 0.18) if surface_marks == "subtle" else rng.uniform(0.08, 0.46)
        width = rng.uniform(0.002, 0.005) if surface_marks == "subtle" else rng.uniform(0.003, 0.010)
        mat = scratch_light if idx % 3 else scratch_dark
        add_box(
            f"random floor hairline scratch {idx:02d}",
            (x, y, 0.003 + 0.0002 * (idx % 5)),
            (length, width, 0.001),
            mat,
            rotation_z=rng.uniform(-0.18, 0.18),
        )

    for idx in range(dust_count):
        add_box(
            f"tiny floor dust fleck {idx:02d}",
            (rng.uniform(-2.6, 3.0), rng.uniform(-1.7, 2.2), 0.004),
            (rng.uniform(0.010, 0.026), rng.uniform(0.004, 0.012), 0.001),
            dust_mat,
            rotation_z=rng.uniform(0.0, math.pi),
        )


def add_background_clutter(scenario: dict[str, object]) -> None:
    rng = random.Random(int(scenario["seed"]) + 202)
    dark_wood = create_principled_material("distant dark furniture wood", (0.22, 0.13, 0.07, 1.0), 0.58, noise_bump=0.01)
    shadow_plastic = create_principled_material("matte dark plastic foot", (0.035, 0.033, 0.030, 1.0), 0.72)

    add_box("low background wooden rail", (2.62, 2.64, 0.78), (1.85, 0.18, 0.16), dark_wood, rotation_z=rng.uniform(-0.04, 0.04), bevel_width=0.012)
    for idx, x in enumerate((1.82, 3.36)):
        add_cylinder(
            f"out of focus furniture leg {idx:02d}",
            (x, 2.35 + rng.uniform(-0.05, 0.04), 0.50),
            0.055,
            1.0,
            shadow_plastic,
        )


def add_environment(scenario: dict[str, object]) -> None:
    setup_world_lighting(scenario)
    floor_mat = create_hardwood_floor_material(scenario)
    bpy.ops.mesh.primitive_plane_add(size=100.0, location=(0.0, 0.0, 0.0))
    floor = bpy.context.object
    floor.name = "warm_hardwood_room_floor"
    floor.data.materials.append(floor_mat)
    add_incline_ramp(scenario)
    add_room_shell(scenario)
    add_floor_details(scenario)
    add_background_clutter(scenario)

    lighting = scenario["lighting"]
    assert isinstance(lighting, dict)
    add_area_light(
        "large soft window-like key",
        (-2.9, -3.8, 4.1),
        float(lighting["key_power"]),
        float(lighting["key_size"]),
        LIGHT_TARGET,
        tuple(float(value) for value in lighting.get("key_color", (1.0, 1.0, 1.0, 1.0))),
    )
    add_area_light(
        "small warm floor bounce",
        (2.4, -2.0, 2.0),
        float(lighting["bounce_power"]),
        2.2,
        (0.2, -0.05, 0.35),
        tuple(float(value) for value in lighting.get("bounce_color", (1.0, 0.78, 0.55, 1.0))),
    )
    add_area_light(
        "wide cool wall fill",
        (3.6, -3.2, 2.7),
        float(lighting.get("wall_fill_power", 10.0)),
        4.0,
        (0.0, 0.25, 0.75),
        tuple(float(value) for value in lighting.get("fill_color", (0.82, 0.86, 0.90, 1.0))),
    )


def ramp_surface_metadata(scenario: dict[str, object]) -> dict[str, object] | None:
    physics = scenario.get("physics")
    if not isinstance(physics, dict) or not bool(physics.get("ramp_enabled", False)):
        return None
    location = tuple(float(value) for value in physics["ramp_location"])
    dimensions = tuple(float(value) for value in physics["ramp_dimensions"])
    pitch_deg = float(physics["ramp_pitch_deg"])
    pitch = math.radians(pitch_deg)
    half_length = 0.5 * dimensions[0]
    half_width = 0.5 * dimensions[1]
    half_thickness = 0.5 * dimensions[2]
    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)
    normal = (sin_pitch, 0.0, cos_pitch)
    origin = (
        location[0] + sin_pitch * half_thickness,
        location[1],
        location[2] + cos_pitch * half_thickness,
    )
    plane_d = -sum(a * b for a, b in zip(normal, origin))
    axis_u = (cos_pitch, 0.0, -sin_pitch)
    axis_v = (0.0, 1.0, 0.0)
    return {
        "name": "incline_ramp_top",
        "surface_role": "sloped_support",
        "plane_model": [normal[0], normal[1], normal[2], plane_d],
        "plane_origin": list(origin),
        "plane_axis_u": list(axis_u),
        "plane_axis_v": list(axis_v),
        "extent_uv": [[-half_length, -half_width], [half_length, half_width]],
        "pitch_deg": pitch_deg,
        "dimensions": list(dimensions),
    }


def add_incline_ramp(scenario: dict[str, object]) -> bpy.types.Object | None:
    surface = ramp_surface_metadata(scenario)
    if surface is None:
        return None
    physics = scenario["physics"]
    assert isinstance(physics, dict)
    location = tuple(float(value) for value in physics["ramp_location"])
    dimensions = tuple(float(value) for value in physics["ramp_dimensions"])
    mat = create_principled_material(
        "matte worn plywood incline ramp",
        scenario_color(scenario, "ramp_color", (0.54, 0.42, 0.30, 1.0)),
        roughness=0.78,
        noise_bump=0.018,
    )
    ramp_profile = str(physics.get("ramp_profile", "tilted_slab"))
    if ramp_profile == "grounded_wedge":
        mesh = bpy.data.meshes.new("grounded_incline_wedge_mesh")
        mesh.from_pydata(
            grounded_wedge_vertices(
                location=location,
                dimensions=dimensions,
                pitch_deg=float(physics["ramp_pitch_deg"]),
            ),
            [],
            GROUNDED_WEDGE_QUADS,
        )
        mesh.update()
        ramp = bpy.data.objects.new("grounded_incline_wedge", mesh)
        bpy.context.collection.objects.link(ramp)
        ramp.data.materials.append(mat)
        ramp.data.use_auto_smooth = True
        bevel = ramp.modifiers.new("soft real-world edges", "BEVEL")
        bevel.width = 0.006
        bevel.segments = 2
        ramp.modifiers.new("weighted normals", "WEIGHTED_NORMAL")
    else:
        pitch = math.radians(float(physics["ramp_pitch_deg"]))
        ramp = add_box(
            "finite_incline_ramp",
            location,
            dimensions,
            mat,
            rotation_euler=(0.0, pitch, 0.0),
            bevel_width=0.006,
        )
    cube_project_uvs(ramp, cube_size=1.0)
    return ramp


def floor_surface_metadata() -> dict[str, object]:
    return {
        "name": "room_floor",
        "surface_role": "horizontal_support",
        "plane_model": [0.0, 0.0, 1.0, 0.0],
        "plane_origin": [0.0, 0.0, 0.0],
        "plane_axis_u": [1.0, 0.0, 0.0],
        "plane_axis_v": [0.0, 1.0, 0.0],
        "extent_uv": [[-50.0, -50.0], [50.0, 50.0]],
    }


def create_rubber_ball_material(scenario: dict[str, object]) -> bpy.types.Material:
    mat = create_principled_material(
        "scuffed orange red rubber ball",
        scenario_color(scenario, "ball_base_color", (0.72, 0.075, 0.025, 1.0)),
        roughness=0.76,
    )
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        return mat

    set_input_default(bsdf, "Specular", 0.28)

    color_noise = add_noise_node(nodes, 18.0, 9.0, 0.60)
    color_ramp = add_color_ramp(
        nodes,
        0.18,
        (0.50, 0.035, 0.014, 1.0),
        1.0,
        (0.96, 0.18, 0.055, 1.0),
    )
    links.new(color_noise.outputs["Fac"], color_ramp.inputs["Fac"])

    scuff_noise = add_noise_node(nodes, 68.0, 12.0, 0.70)
    scuff_mask = add_color_ramp(
        nodes,
        0.72,
        (0.0, 0.0, 0.0, 1.0),
        0.96,
        (0.36, 0.36, 0.36, 1.0),
    )
    worn_color = nodes.new(type="ShaderNodeMixRGB")
    worn_color.blend_type = "MIX"
    worn_color.inputs["Color2"].default_value = (0.23, 0.10, 0.045, 1.0)
    links.new(color_ramp.outputs["Color"], worn_color.inputs["Color1"])
    links.new(scuff_noise.outputs["Fac"], scuff_mask.inputs["Fac"])
    links.new(scuff_mask.outputs["Color"], worn_color.inputs["Fac"])
    links.new(worn_color.outputs["Color"], bsdf.inputs["Base Color"])

    rubber_scale = scenario_material_float(scenario, "ball_texture_scale", 3.8)
    rubber_mapping = add_texture_mapping(nodes, links, (rubber_scale, rubber_scale, 1.0), "UV")
    rubber_normal = add_mapped_image_texture(
        nodes,
        links,
        rubber_mapping,
        require_ambientcg_path("Rubber002", "Rubber002_4K-JPG_NormalGL.jpg"),
        "Non-Color",
    )
    rubber_normal_map = nodes.new(type="ShaderNodeNormalMap")
    rubber_normal_map.inputs["Strength"].default_value = 0.115
    links.new(rubber_normal.outputs["Color"], rubber_normal_map.inputs["Color"])

    rubber_height = add_mapped_image_texture(
        nodes,
        links,
        rubber_mapping,
        require_ambientcg_path("Rubber002", "Rubber002_4K-JPG_Displacement.jpg"),
        "Non-Color",
    )
    rubber_height_bump = nodes.new(type="ShaderNodeBump")
    rubber_height_bump.inputs["Strength"].default_value = 0.045
    rubber_height_bump.inputs["Distance"].default_value = 0.005
    links.new(rubber_height.outputs["Color"], rubber_height_bump.inputs["Height"])
    links.new(rubber_normal_map.outputs["Normal"], rubber_height_bump.inputs["Normal"])

    pore_noise = add_noise_node(nodes, 230.0, 12.0, 0.66)
    pore_bump = nodes.new(type="ShaderNodeBump")
    pore_bump.inputs["Strength"].default_value = 0.018
    pore_bump.inputs["Distance"].default_value = 0.0025
    links.new(pore_noise.outputs["Fac"], pore_bump.inputs["Height"])
    links.new(rubber_height_bump.outputs["Normal"], pore_bump.inputs["Normal"])
    links.new(pore_bump.outputs["Normal"], bsdf.inputs["Normal"])

    rough_noise = add_noise_node(nodes, 34.0, 8.0, 0.72)
    rough_ramp = add_color_ramp(
        nodes,
        0.15,
        (0.62, 0.62, 0.62, 1.0),
        1.0,
        (0.90, 0.90, 0.90, 1.0),
    )
    links.new(rough_noise.outputs["Fac"], rough_ramp.inputs["Fac"])
    rubber_rough = add_mapped_image_texture(
        nodes,
        links,
        rubber_mapping,
        require_ambientcg_path("Rubber002", "Rubber002_4K-JPG_Roughness.jpg"),
        "Non-Color",
    )
    rough_mix = nodes.new(type="ShaderNodeMixRGB")
    rough_mix.blend_type = "MIX"
    rough_mix.inputs["Fac"].default_value = 0.48
    links.new(rough_ramp.outputs["Color"], rough_mix.inputs["Color1"])
    links.new(rubber_rough.outputs["Color"], rough_mix.inputs["Color2"])
    links.new(rough_mix.outputs["Color"], bsdf.inputs["Roughness"])

    return mat


def create_ball_seam_material() -> bpy.types.Material:
    mat = create_principled_material(
        "dark worn rubber seams",
        (0.055, 0.038, 0.030, 1.0),
        roughness=0.82,
        noise_bump=0.012,
    )
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf is not None:
        set_input_default(bsdf, "Specular", 0.18)
    return mat


def create_ball_scuff_material() -> bpy.types.Material:
    mat = create_principled_material(
        "thin dusty rubber scuffs",
        (0.38, 0.14, 0.060, 1.0),
        roughness=0.90,
        noise_bump=0.004,
    )
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf is not None:
        set_input_default(bsdf, "Specular", 0.12)
    return mat


def scuff_patch_mesh(
    patch_idx: int,
    normal_xyz: tuple[float, float, float],
    width: float,
    height: float,
    rotation: float,
    radius: float,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    normal = Vector(normal_xyz).normalized()
    helper = Vector((0.0, 0.0, 1.0)) if abs(normal.z) < 0.9 else Vector((0.0, 1.0, 0.0))
    tangent_u = normal.cross(helper).normalized()
    tangent_v = normal.cross(tangent_u).normalized()
    axis_u = tangent_u * math.cos(rotation) + tangent_v * math.sin(rotation)
    axis_v = tangent_v * math.cos(rotation) - tangent_u * math.sin(rotation)

    segments = 12
    verts = [tuple(normal * radius * 1.014)]
    for segment_idx in range(segments):
        angle = 2.0 * math.pi * segment_idx / segments
        jitter = 0.82 + 0.16 * ((patch_idx * 17 + segment_idx * 7) % 5) / 4.0
        point = (
            normal * radius
            + axis_u * (math.cos(angle) * width * jitter)
            + axis_v * (math.sin(angle) * height * jitter)
        )
        verts.append(tuple(point.normalized() * radius * 1.014))

    faces = [
        (0, segment_idx + 1, 1 if segment_idx == segments - 1 else segment_idx + 2)
        for segment_idx in range(segments)
    ]
    return verts, faces


def add_ball_surface_scuffs(
    ball: bpy.types.Object,
    radius: float,
    scenario: dict[str, object],
) -> None:
    material = create_ball_scuff_material()
    patches = list(BALL_SCUFF_PATCHES)
    for patch in scenario.get("ball_scuffs", []):
        if not isinstance(patch, dict):
            continue
        patches.append(
            (
                tuple(float(value) for value in patch["normal"]),
                float(patch["width"]),
                float(patch["height"]),
                float(patch["rotation"]),
            )
        )

    for patch_idx, patch in enumerate(patches):
        verts, faces = scuff_patch_mesh(patch_idx, *patch, radius)

        mesh = bpy.data.meshes.new(f"ball_surface_scuff_{patch_idx:02d}_mesh")
        mesh.from_pydata(verts, [], faces)
        mesh.update()
        scuff = bpy.data.objects.new(f"ball_surface_scuff_{patch_idx:02d}", mesh)
        bpy.context.collection.objects.link(scuff)
        scuff.data.materials.append(material)
        scuff.parent = ball
        scuff.location = (0.0, 0.0, 0.0)


def add_ball(radius: float, scenario: dict[str, object]) -> bpy.types.Object:
    ball_mat = create_rubber_ball_material(scenario)
    seam_mat = create_ball_seam_material()
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=96,
        ring_count=48,
        radius=radius,
        location=ball_initial_location(scenario),
    )
    ball = bpy.context.object
    ball.name = "ball"
    ball.data.materials.append(ball_mat)
    bpy.ops.object.shade_smooth()

    for name, rotation in BALL_SEAMS:
        bpy.ops.mesh.primitive_torus_add(
            major_radius=radius * 1.005,
            minor_radius=radius * 0.012,
            major_segments=144,
            minor_segments=8,
            location=ball.location,
            rotation=rotation,
        )
        seam = bpy.context.object
        seam.name = name
        seam.data.materials.append(seam_mat)
        seam.parent = ball
        seam.matrix_parent_inverse.identity()
        seam.location = (0.0, 0.0, 0.0)
        seam.rotation_euler = rotation
        bpy.ops.object.shade_smooth()
    add_ball_surface_scuffs(ball, radius, scenario)
    return ball


def add_block_wear(block: bpy.types.Object, scenario: dict[str, object]) -> None:
    surface_marks = str(scenario.get("surface_marks", "none"))
    if surface_marks == "none":
        return

    rng = random.Random(int(scenario["seed"]) + 404)
    wear_color = (0.42, 0.28, 0.16, 1.0) if surface_marks == "subtle" else (0.74, 0.56, 0.36, 1.0)
    wear_mat = create_principled_material("dry muted worn wood edges", wear_color, 0.84, noise_bump=0.006)
    dirt_mat = create_principled_material("small dark dents in block", (0.14, 0.075, 0.035, 1.0), 0.90, noise_bump=0.004)
    half = tuple(value * 0.5 for value in WOOD_BLOCK_DIMENSIONS)
    faces = ("x+", "x-", "y+", "y-", "z+")
    patch_count = 7 if surface_marks == "subtle" else 22
    for idx in range(patch_count):
        face = rng.choice(faces)
        mat = wear_mat if idx % 4 else dirt_mat
        if face == "z+":
            loc = (
                rng.uniform(-half[0] * 0.78, half[0] * 0.78),
                rng.uniform(-half[1] * 0.78, half[1] * 0.78),
                half[2] + 0.003,
            )
            dims = (rng.uniform(0.04, 0.16), rng.uniform(0.006, 0.022), 0.004)
        elif face in {"x+", "x-"}:
            sign = 1.0 if face == "x+" else -1.0
            loc = (
                sign * (half[0] + 0.003),
                rng.uniform(-half[1] * 0.82, half[1] * 0.82),
                rng.uniform(-half[2] * 0.68, half[2] * 0.80),
            )
            dims = (0.004, rng.uniform(0.035, 0.13), rng.uniform(0.006, 0.025))
        else:
            sign = 1.0 if face == "y+" else -1.0
            loc = (
                rng.uniform(-half[0] * 0.82, half[0] * 0.82),
                sign * (half[1] + 0.003),
                rng.uniform(-half[2] * 0.68, half[2] * 0.80),
            )
            dims = (rng.uniform(0.035, 0.13), 0.004, rng.uniform(0.006, 0.025))

        bpy.ops.mesh.primitive_cube_add(size=1.0)
        patch = bpy.context.object
        patch.name = f"wood_block_worn_patch_{idx:02d}"
        patch.parent = block
        patch.matrix_parent_inverse.identity()
        patch.location = loc
        patch.rotation_euler[2] = rng.uniform(-0.35, 0.35)
        patch.dimensions = dims
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        patch.data.materials.append(mat)


def add_wood_block(scenario: dict[str, object]) -> bpy.types.Object:
    physics = scenario["physics"]
    assert isinstance(physics, dict)
    location = tuple(float(value) for value in physics["block_location"])
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    block = bpy.context.object
    block.name = "wood_block"
    block.dimensions = WOOD_BLOCK_DIMENSIONS
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    cube_project_uvs(block, cube_size=1.0)
    block.data.materials.append(create_wood_material(scenario))
    block.data.use_auto_smooth = True
    bevel = block.modifiers.new("small rounded worn edges", "BEVEL")
    bevel.width = 0.034
    bevel.segments = 5
    block.modifiers.new("weighted corner normals", "WEIGHTED_NORMAL")
    add_block_wear(block, scenario)
    return block


def run_physics_simulation(
    args: argparse.Namespace,
    radius: float,
    scenario: dict[str, object],
) -> dict:
    python = shutil.which("python3") or shutil.which("python")
    if not python:
        raise RuntimeError("Cannot find python3/python for the PyBullet physics simulation.")

    script_path = Path(__file__).with_name("simulate_ball_block_impact.py")
    physics_path = args.out_dir / PHYSICS_TEMP_NAME
    physics = scenario["physics"]
    assert isinstance(physics, dict)
    block_location = physics["block_location"]
    ball_initial_location = physics["ball_initial_location"]
    ball_initial_velocity = physics["ball_initial_velocity"]
    subprocess.run(
        [
            python,
            str(script_path),
            "--out",
            str(physics_path),
            "--fps",
            str(int(args.fps)),
            "--duration-sec",
            str(float(args.duration_sec)),
            "--ball-radius",
            str(float(radius)),
            "--ball-enabled" if bool(physics.get("ball_enabled", True)) else "--no-ball-enabled",
            "--block-enabled" if bool(physics.get("block_enabled", True)) else "--no-block-enabled",
            "--ball-initial-location",
            str(float(ball_initial_location[0])),
            str(float(ball_initial_location[1])),
            str(float(ball_initial_location[2])),
            "--block-location",
            str(float(block_location[0])),
            str(float(block_location[1])),
            str(float(block_location[2])),
            "--block-yaw-deg",
            str(float(physics["block_yaw_deg"])),
            "--block-pitch-deg",
            str(float(physics.get("block_pitch_deg", 0.0))),
            "--ball-initial-velocity",
            str(float(ball_initial_velocity[0])),
            str(float(ball_initial_velocity[1])),
            str(float(ball_initial_velocity[2])),
            "--block-initial-velocity",
            str(float(physics.get("block_initial_velocity", (0.0, 0.0, 0.0))[0])),
            str(float(physics.get("block_initial_velocity", (0.0, 0.0, 0.0))[1])),
            str(float(physics.get("block_initial_velocity", (0.0, 0.0, 0.0))[2])),
            "--ball-mass",
            str(float(physics["ball_mass"])),
            "--block-mass",
            str(float(physics["block_mass"])),
            "--floor-friction",
            str(float(physics["floor_friction"])),
            "--ball-friction",
            str(float(physics["ball_friction"])),
            "--ball-restitution",
            str(float(physics["ball_restitution"])),
            "--block-friction",
            str(float(physics["block_friction"])),
            "--block-restitution",
            str(float(physics["block_restitution"])),
            *(["--ramp-enabled"] if bool(physics.get("ramp_enabled", False)) else []),
            "--ramp-location",
            str(float(physics.get("ramp_location", INCLINE_RAMP_LOCATION)[0])),
            str(float(physics.get("ramp_location", INCLINE_RAMP_LOCATION)[1])),
            str(float(physics.get("ramp_location", INCLINE_RAMP_LOCATION)[2])),
            "--ramp-dimensions",
            str(float(physics.get("ramp_dimensions", INCLINE_RAMP_DIMENSIONS)[0])),
            str(float(physics.get("ramp_dimensions", INCLINE_RAMP_DIMENSIONS)[1])),
            str(float(physics.get("ramp_dimensions", INCLINE_RAMP_DIMENSIONS)[2])),
            "--ramp-pitch-deg",
            str(float(physics.get("ramp_pitch_deg", INCLINE_RAMP_PITCH_DEG))),
            "--ramp-friction",
            str(float(physics.get("ramp_friction", 0.58))),
            "--ramp-restitution",
            str(float(physics.get("ramp_restitution", 0.05))),
            "--ramp-profile",
            str(physics.get("ramp_profile", "tilted_slab")),
            *(["--wall-enabled"] if bool(physics.get("wall_enabled", False)) else []),
            "--wall-location",
            str(float(physics.get("wall_location", (0.0, 3.05, 1.45))[0])),
            str(float(physics.get("wall_location", (0.0, 3.05, 1.45))[1])),
            str(float(physics.get("wall_location", (0.0, 3.05, 1.45))[2])),
            "--wall-dimensions",
            str(float(physics.get("wall_dimensions", (8.6, 0.08, 2.90))[0])),
            str(float(physics.get("wall_dimensions", (8.6, 0.08, 2.90))[1])),
            str(float(physics.get("wall_dimensions", (8.6, 0.08, 2.90))[2])),
            "--wall-friction",
            str(float(physics.get("wall_friction", 0.34))),
            "--wall-restitution",
            str(float(physics.get("wall_restitution", 0.82))),
        ],
        check=True,
    )
    records = json.loads(physics_path.read_text(encoding="utf-8"))
    physics_path.unlink(missing_ok=True)
    return records


def apply_physics_animation(
    ball: bpy.types.Object | None,
    block: bpy.types.Object | None,
    physics: dict,
) -> None:
    objects = [obj for obj in (ball, block) if obj is not None]
    for obj in objects:
        obj.rotation_mode = "QUATERNION"

    for frame_record in physics["frames"]:
        frame = int(frame_record["frame_index"])
        for obj, prefix in ((ball, "ball"), (block, "wood_block")):
            if obj is None or f"{prefix}_quaternion_xyzw" not in frame_record:
                continue
            quat_xyzw = frame_record[f"{prefix}_quaternion_xyzw"]
            obj.location = frame_record[f"{prefix}_location"]
            obj.rotation_quaternion = (
                quat_xyzw[3],
                quat_xyzw[0],
                quat_xyzw[1],
                quat_xyzw[2],
            )
            obj.keyframe_insert(data_path="location", frame=frame)
            obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)

    set_linear_keyframes(objects)


def set_linear_keyframes(objects: Iterable[bpy.types.Object]) -> None:
    for obj in objects:
        if obj.animation_data and obj.animation_data.action:
            for fcurve in obj.animation_data.action.fcurves:
                for key in fcurve.keyframe_points:
                    key.interpolation = "LINEAR"


def export_ground_truth(
    out_dir: Path,
    ball: bpy.types.Object | None,
    block: bpy.types.Object | None,
    camera: bpy.types.Object,
    frame_end: int,
    fps: int,
    physics: dict,
    scenario: dict[str, object],
) -> None:
    scene = bpy.context.scene
    static_scene_surfaces = [floor_surface_metadata()]
    ramp_surface = ramp_surface_metadata(scenario)
    if ramp_surface is not None:
        static_scene_surfaces.append(ramp_surface)
    objects: dict[str, object] = {}
    if ball is not None:
        objects["ball"] = {
            "object_name": ball.name,
            "radius_m_scene_units": BALL_RADIUS,
        }
    if block is not None:
        objects["wood_block"] = {
            "object_name": block.name,
            "dimensions_scene_units": list(WOOD_BLOCK_DIMENSIONS),
        }

    records = {
        "schema_version": 1,
        "fps": int(fps),
        "frame_start": 1,
        "frame_end": int(frame_end),
        "scenario_metadata_path": str((out_dir / SCENARIO_METADATA_NAME).resolve()),
        "physics": {
            key: value
            for key, value in physics.items()
            if key != "frames"
        },
        "objects": objects,
        "camera": {
            "object_name": camera.name,
            "lens_mm": float(camera.data.lens),
            "sensor_width_mm": float(camera.data.sensor_width),
            "resolution": [
                int(scene.render.resolution_x),
                int(scene.render.resolution_y),
            ],
        },
        "scenario": {
            "seed": int(scenario["seed"]),
            "realism_profile": scenario["realism_profile"],
            "motion": scenario["motion"],
        },
        "static_scene_surfaces": static_scene_surfaces,
        "frames": [],
    }
    physics_by_frame = {
        int(frame_record["frame_index"]): frame_record
        for frame_record in physics["frames"]
    }
    for frame in range(1, frame_end + 1):
        scene.frame_set(frame)
        physics_frame = physics_by_frame[frame]
        frame_record = {
            "frame_index": frame,
            "time_sec": (frame - 1) / float(fps),
            "camera_matrix_world": [[float(v) for v in row] for row in camera.matrix_world],
            "camera_world_to_camera_matrix": [
                [float(v) for v in row]
                for row in camera.matrix_world.inverted()
            ],
        }
        if ball is not None:
            frame_record.update(
                {
                    "ball_matrix_world": [[float(v) for v in row] for row in ball.matrix_world],
                    "ball_location": [float(v) for v in ball.location],
                    "ball_linear_velocity": physics_frame["ball_linear_velocity"],
                    "ball_angular_velocity": physics_frame["ball_angular_velocity"],
                    "ball_floor_gap": physics_frame["ball_floor_gap"],
                    "ball_block_gap": physics_frame.get("ball_block_gap"),
                    "ball_ramp_gap": physics_frame.get("ball_ramp_gap"),
                    "ball_wall_gap": physics_frame.get("ball_wall_gap"),
                }
            )
        if block is not None:
            frame_record.update(
                {
                    "wood_block_matrix_world": [[float(v) for v in row] for row in block.matrix_world],
                    "wood_block_location": [float(v) for v in block.location],
                    "wood_block_linear_velocity": physics_frame["wood_block_linear_velocity"],
                    "wood_block_angular_velocity": physics_frame["wood_block_angular_velocity"],
                    "wood_block_floor_gap": physics_frame.get("wood_block_floor_gap"),
                    "wood_block_ramp_gap": physics_frame.get("wood_block_ramp_gap"),
                }
            )
        records["frames"].append(frame_record)
    (out_dir / GROUND_TRUTH_NAME).write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )


def save_blend(out_dir: Path) -> None:
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path(out_dir, BLEND_NAME)))


def build_scene(args: argparse.Namespace, scenario: dict[str, object]) -> None:
    clear_scene()
    setup_render(args, scenario)
    write_scenario_metadata(args.out_dir, scenario)
    add_environment(scenario)
    camera = add_camera(scenario)
    radius = BALL_RADIUS
    ball = add_ball(radius, scenario) if ball_enabled(scenario) else None
    block = add_wood_block(scenario) if block_enabled(scenario) else None
    physics = run_physics_simulation(args, radius, scenario)
    apply_physics_animation(ball, block, physics)
    export_ground_truth(
        args.out_dir,
        ball,
        block,
        camera,
        bpy.context.scene.frame_end,
        int(args.fps),
        physics,
        scenario,
    )
    save_blend(args.out_dir)


def render_preview(args: argparse.Namespace) -> None:
    scene = bpy.context.scene
    preview_frame = max(scene.frame_start, min(int(args.preview_frame), scene.frame_end))
    scene.frame_set(preview_frame)
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(output_path(args.out_dir, f"preview_frame_{preview_frame:05d}.png"))
    bpy.ops.render.render(write_still=True)


def render_animation(args: argparse.Namespace, scenario: dict[str, object]) -> None:
    scene = bpy.context.scene
    scene.frame_set(scene.frame_start)
    configure_ffmpeg(scene, args.out_dir)
    bpy.ops.render.render(animation=True)
    write_compatible_mp4(args.out_dir, int(args.fps), scenario)


def main() -> None:
    args = parse_args()
    scenario = scenario_from_args(args)
    build_scene(args, scenario)
    if args.mode == "preview":
        render_preview(args)
        return
    render_animation(args, scenario)


if __name__ == "__main__":
    main()
