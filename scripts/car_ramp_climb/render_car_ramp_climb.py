from __future__ import annotations

import argparse
import copy
import json
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import bmesh
import bpy
from mathutils import Matrix, Vector


WORKSPACE_DIR = Path(__file__).resolve().parents[2]
POLYHAVEN_DIR = WORKSPACE_DIR / "assets" / "polyhaven"
AMBIENTCG_DIR = WORKSPACE_DIR / "assets" / "ambientcg"

OUTPUT_STEM = "car_ramp_climb"
DIRECT_MP4_NAME = f"{OUTPUT_STEM}.mp4"
BLEND_NAME = f"{OUTPUT_STEM}.blend"
GROUND_TRUTH_NAME = "ground_truth_transforms.json"
PHYSICS_TEMP_NAME = "physics_transforms.json"
SCENARIO_METADATA_NAME = "scenario_metadata.json"

# Geometry matches simulate_car_ramp_climb.py.
CAR_HALF_WIDTH = 0.0504
CAR_HALF_LENGTH = 0.11685
CAR_HALF_HEIGHT = 0.03255

FLOOR_Z = 0.0
RAMP_ANGLE_DEG = 20.0
RAMP_LENGTH = 0.9
RAMP_WIDTH = 0.35
RAMP_THICKNESS = 0.03
RAIL_HEIGHT = 0.035
RAIL_THICKNESS = 0.02

# assets/models/modern_house.glb: a full house-and-yard model, used as-is
# (native scale, in meters) as static background/environment instead of the
# earlier procedural ground+wall. The ramp assembly (ramp, rails, stone,
# car) is built exactly as before in its own local frame (local X = up-slope
# direction) and then parented as a group to PLACEMENT (a static empty with
# a fixed rotation+location) to drop it onto "Front lawn design" -- a ~4x4m
# stone-bordered paved square out in the open lawn (identified from a
# top-down orthographic render of the house, matching a patch the user
# pointed out directly) -- with the camera on the patio's far side looking
# back across it toward the house facade in the distance. Two earlier
# placements were tried and rejected: the side yard has a hedge planted
# right against that wall, blocking the camera entirely; the driveway
# placement (right up against the front wall) worked but put the ramp too
# close for the user's taste. Both the house's front wall and this patio
# face along Y, so PLACEMENT needs no rotation, only a translation.
HOUSE_GLB_PATH = WORKSPACE_DIR / "assets" / "models" / "modern_house.glb"
# The z offset (0.00929) lifts the whole ramp assembly onto the house's paved
# surface. The physics rests everything on a sim floor at z=0, but the house
# model's "Front lawn design" pavement here actually sits at z=+0.00929 (found
# by ray-casting the mesh straight down at this x,y), so a z=0 placement sank
# the ramp base, stone, and -- most visibly -- the landed car ~1cm into the
# pavement. Matching the placement floor to the pavement height puts the
# resting car flush on the ground instead of clipping through it.
PLACEMENT_LOCATION = (5.76, -7.89, 0.00929)
PLACEMENT_ROTATION_Z_DEG = 0.0

CAMERA_LOCATION = (5.76, -9.3, 0.48)
CAMERA_TARGET = (5.76, -6.5, 0.3)
CAMERA_LENS_MM = 28.0
LIGHT_TARGET = (5.76, -7.89, 0.4)


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preview", "animation", "frames"), default="animation")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resolution", nargs=2, type=int, default=(960, 540))
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-sec", type=float, default=3.0)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--preview-frame", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cpu"), default="cpu")
    parser.add_argument("--seed", type=int, default=31)
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
    return parser.parse_args(argv)


def require_path(base_dir: Path, filename: str) -> Path:
    file_path = base_dir / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Texture missing: {file_path}")
    return file_path


def load_image(path: Path, color_space: str) -> bpy.types.Image:
    name = f"{path.stem}_{color_space}"
    existing = bpy.data.images.get(name)
    if existing is not None and existing.filepath == str(path):
        return existing
    return bpy.data.images.load(str(path), check_existing=False)


def set_input_default(node: bpy.types.ShaderNode, input_name: str, value) -> None:
    socket = node.inputs.get(input_name)
    if socket is not None:
        socket.default_value = value


# --- Ramp surface materials -------------------------------------------------
# The one thing this scene varies between renders. "grip_orange" and
# "turf_green" are stylized procedural surfaces (colored + noise bump) --
# real griptape/turf close-up photo textures suitable for this scale are hard
# to source cleanly, and the point of the comparison is the friction value,
# not photorealistic grass blades. "asphalt_grey" and "asphalt_dark" both use
# the same real ambientCG Asphalt031 scan; the dark variant is the identical
# texture darkened through a Hue/Saturation node rather than a second
# download, matching the same trick used for toy_car_ball's floor/shelf
# contrast.


def create_procedural_ramp_material(name: str, base_color: tuple, roughness: float) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        return mat

    set_input_default(bsdf, "Base Color", base_color)
    set_input_default(bsdf, "Roughness", roughness)
    set_input_default(bsdf, "Metallic", 0.0)

    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    mapping = nodes.new(type="ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (40.0, 40.0, 40.0)
    links.new(tex_coord.outputs["UV"], mapping.inputs["Vector"])

    noise = nodes.new(type="ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 60.0
    noise.inputs["Detail"].default_value = 5.0
    noise.inputs["Roughness"].default_value = 0.65
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])

    bump = nodes.new(type="ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.35
    bump.inputs["Distance"].default_value = 0.003
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    return mat


def create_asphalt_material(name: str, *, darken: float = 1.0) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        return mat

    set_input_default(bsdf, "Roughness", 0.85)
    set_input_default(bsdf, "Metallic", 0.0)

    # Triplanar BOX projection off object coordinates (matching the rails and
    # stone) instead of the old UV map: the ramp is a non-uniformly scaled box,
    # so a flat UV unwrap stretched the square asphalt scan ~2.5:1 across the
    # top face and smeared the aggregate. A larger feature scale (was 2.5 tiles
    # per face, which shrank the grain to sub-pixel noise) plus a stronger
    # normal and an added coarse bump make the aggregate actually read as
    # texture at this camera distance.
    asset_dir = AMBIENTCG_DIR / "Asphalt031"
    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    mapping = nodes.new(type="ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (1.1, 1.1, 1.1)
    links.new(tex_coord.outputs["Object"], mapping.inputs["Vector"])

    def asphalt_image(fname: str, colorspace: str) -> bpy.types.ShaderNodeTexImage:
        img = load_image(require_path(asset_dir, fname), colorspace)
        img.colorspace_settings.name = colorspace
        tex = nodes.new(type="ShaderNodeTexImage")
        tex.image = img
        tex.projection = "BOX"
        tex.projection_blend = 0.3
        links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
        return tex

    try:
        color_tex = asphalt_image("Asphalt031_4K-JPG_Color.jpg", "sRGB")
        base_color_output = color_tex.outputs["Color"]
        if darken != 1.0:
            hue_sat = nodes.new(type="ShaderNodeHueSaturation")
            hue_sat.inputs["Value"].default_value = darken
            links.new(color_tex.outputs["Color"], hue_sat.inputs["Color"])
            base_color_output = hue_sat.outputs["Color"]
        # The bare scan is a pale, low-contrast grey; on this up-facing surface
        # under a bright sky it flattens to a near-uniform patch. A contrast
        # boost (and a touch darker) pulls the aggregate light/dark variation
        # back out so the surface reads as textured asphalt rather than a plain
        # grey strip.
        contrast = nodes.new(type="ShaderNodeBrightContrast")
        contrast.inputs["Bright"].default_value = -0.03
        contrast.inputs["Contrast"].default_value = 0.22
        links.new(base_color_output, contrast.inputs["Color"])
        links.new(contrast.outputs["Color"], bsdf.inputs["Base Color"])

        rough_tex = asphalt_image("Asphalt031_4K-JPG_Roughness.jpg", "Non-Color")
        links.new(rough_tex.outputs["Color"], bsdf.inputs["Roughness"])

        nor_tex = asphalt_image("Asphalt031_4K-JPG_NormalGL.jpg", "Non-Color")
        nor_map = nodes.new(type="ShaderNodeNormalMap")
        nor_map.inputs["Strength"].default_value = 1.3
        links.new(nor_tex.outputs["Color"], nor_map.inputs["Color"])

        grit = nodes.new(type="ShaderNodeTexNoise")
        grit.inputs["Scale"].default_value = 16.0
        grit.inputs["Detail"].default_value = 8.0
        grit.inputs["Roughness"].default_value = 0.75
        links.new(mapping.outputs["Vector"], grit.inputs["Vector"])
        bump = nodes.new(type="ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.5
        bump.inputs["Distance"].default_value = 0.005
        links.new(grit.outputs["Fac"], bump.inputs["Height"])
        links.new(nor_map.outputs["Normal"], bump.inputs["Normal"])
        links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    except FileNotFoundError as e:
        print(f"[WARN] Asphalt texture not found: {e}, using flat fallback")
        set_input_default(bsdf, "Base Color", (0.12 * darken, 0.12 * darken, 0.13 * darken, 1.0))

    return mat


RAMP_SURFACES = {
    "grip_orange": lambda: create_procedural_ramp_material(
        "ramp_grip_orange", (0.85, 0.35, 0.05, 1.0), 0.9,
    ),
    "turf_green": lambda: create_procedural_ramp_material(
        "ramp_turf_green", (0.15, 0.42, 0.12, 1.0), 0.75,
    ),
    "asphalt_grey": lambda: create_asphalt_material("ramp_asphalt_grey", darken=1.0),
    "asphalt_dark": lambda: create_asphalt_material("ramp_asphalt_dark", darken=0.45),
}


def create_brick_wall_material() -> bpy.types.Material:
    mat = bpy.data.materials.new("brick_wall")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        return mat

    set_input_default(bsdf, "Roughness", 0.85)
    set_input_default(bsdf, "Metallic", 0.0)

    asset_dir = POLYHAVEN_DIR / "red_brick"
    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    mapping = nodes.new(type="ShaderNodeMapping")
    # The wall is ~7m wide now (spans the whole ramp length as a parallel
    # backdrop) -- a much higher tiling scale than a small thin panel needs,
    # or the bricks would render many meters wide each.
    mapping.inputs["Scale"].default_value = (30.0, 30.0, 30.0)
    links.new(tex_coord.outputs["UV"], mapping.inputs["Vector"])

    try:
        diff_img = load_image(require_path(asset_dir, "red_brick_diff_4k.jpg"), "sRGB")
        diff_img.colorspace_settings.name = "sRGB"
        diff_tex = nodes.new(type="ShaderNodeTexImage")
        diff_tex.image = diff_img
        links.new(mapping.outputs["Vector"], diff_tex.inputs["Vector"])

        ao_img = load_image(require_path(asset_dir, "red_brick_ao_4k.jpg"), "Non-Color")
        ao_img.colorspace_settings.name = "Non-Color"
        ao_tex = nodes.new(type="ShaderNodeTexImage")
        ao_tex.image = ao_img
        links.new(mapping.outputs["Vector"], ao_tex.inputs["Vector"])

        mix_ao = nodes.new(type="ShaderNodeMixRGB")
        mix_ao.blend_type = "MULTIPLY"
        mix_ao.inputs["Fac"].default_value = 0.7
        links.new(diff_tex.outputs["Color"], mix_ao.inputs["Color1"])
        links.new(ao_tex.outputs["Color"], mix_ao.inputs["Color2"])
        links.new(mix_ao.outputs["Color"], bsdf.inputs["Base Color"])

        rough_img = load_image(require_path(asset_dir, "red_brick_rough_4k.jpg"), "Non-Color")
        rough_img.colorspace_settings.name = "Non-Color"
        rough_tex = nodes.new(type="ShaderNodeTexImage")
        rough_tex.image = rough_img
        links.new(mapping.outputs["Vector"], rough_tex.inputs["Vector"])
        links.new(rough_tex.outputs["Color"], bsdf.inputs["Roughness"])

        nor_img = load_image(require_path(asset_dir, "red_brick_nor_gl_4k.jpg"), "Non-Color")
        nor_img.colorspace_settings.name = "Non-Color"
        nor_tex = nodes.new(type="ShaderNodeTexImage")
        nor_tex.image = nor_img
        nor_map = nodes.new(type="ShaderNodeNormalMap")
        nor_map.inputs["Strength"].default_value = 1.1
        links.new(mapping.outputs["Vector"], nor_tex.inputs["Vector"])
        links.new(nor_tex.outputs["Color"], nor_map.inputs["Color"])
        links.new(nor_map.outputs["Normal"], bsdf.inputs["Normal"])

        # Extra per-brick bump depth on top of the normal map, and a subtle
        # patchy color variation -- a perfectly uniform tiled texture reads
        # as an obvious repeat under close, flat lighting; a bit of both
        # breaks that up like a real weathered wall.
        detail_noise = nodes.new(type="ShaderNodeTexNoise")
        detail_noise.inputs["Scale"].default_value = 6.0
        detail_noise.inputs["Detail"].default_value = 4.0
        links.new(mapping.outputs["Vector"], detail_noise.inputs["Vector"])

        bump = nodes.new(type="ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.15
        bump.inputs["Distance"].default_value = 0.01
        links.new(detail_noise.outputs["Fac"], bump.inputs["Height"])
        links.new(nor_map.outputs["Normal"], bump.inputs["Normal"])
        links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

        tint_noise = nodes.new(type="ShaderNodeTexNoise")
        tint_noise.inputs["Scale"].default_value = 3.0
        tint_noise.inputs["Detail"].default_value = 2.0
        links.new(mapping.outputs["Vector"], tint_noise.inputs["Vector"])

        tint_ramp = nodes.new(type="ShaderNodeValToRGB")
        tint_ramp.color_ramp.elements[0].color = (0.75, 0.75, 0.75, 1.0)
        tint_ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
        links.new(tint_noise.outputs["Fac"], tint_ramp.inputs["Fac"])

        tint_mix = nodes.new(type="ShaderNodeMixRGB")
        tint_mix.blend_type = "MULTIPLY"
        tint_mix.inputs["Fac"].default_value = 0.5
        links.new(mix_ao.outputs["Color"], tint_mix.inputs["Color1"])
        links.new(tint_ramp.outputs["Color"], tint_mix.inputs["Color2"])
        links.new(tint_mix.outputs["Color"], bsdf.inputs["Base Color"])
    except FileNotFoundError as e:
        print(f"[WARN] Brick texture not found: {e}, using flat fallback")
        set_input_default(bsdf, "Base Color", (0.45, 0.2, 0.15, 1.0))

    return mat


def create_pavement_material() -> bpy.types.Material:
    mat = bpy.data.materials.new("ground_pavement")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        return mat

    set_input_default(bsdf, "Roughness", 0.9)
    set_input_default(bsdf, "Metallic", 0.0)

    asset_dir = POLYHAVEN_DIR / "concrete_floor_damaged_01"
    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    mapping = nodes.new(type="ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (4.0, 4.0, 4.0)
    links.new(tex_coord.outputs["UV"], mapping.inputs["Vector"])

    try:
        diff_img = load_image(require_path(asset_dir, "concrete_floor_damaged_01_diff_4k.jpg"), "sRGB")
        diff_img.colorspace_settings.name = "sRGB"
        diff_tex = nodes.new(type="ShaderNodeTexImage")
        diff_tex.image = diff_img
        links.new(mapping.outputs["Vector"], diff_tex.inputs["Vector"])
        links.new(diff_tex.outputs["Color"], bsdf.inputs["Base Color"])

        rough_img = load_image(require_path(asset_dir, "concrete_floor_damaged_01_rough_4k.jpg"), "Non-Color")
        rough_img.colorspace_settings.name = "Non-Color"
        rough_tex = nodes.new(type="ShaderNodeTexImage")
        rough_tex.image = rough_img
        links.new(mapping.outputs["Vector"], rough_tex.inputs["Vector"])
        links.new(rough_tex.outputs["Color"], bsdf.inputs["Roughness"])

        nor_img = load_image(require_path(asset_dir, "concrete_floor_damaged_01_nor_gl_4k.jpg"), "Non-Color")
        nor_img.colorspace_settings.name = "Non-Color"
        nor_tex = nodes.new(type="ShaderNodeTexImage")
        nor_tex.image = nor_img
        nor_map = nodes.new(type="ShaderNodeNormalMap")
        nor_map.inputs["Strength"].default_value = 0.5
        links.new(mapping.outputs["Vector"], nor_tex.inputs["Vector"])
        links.new(nor_tex.outputs["Color"], nor_map.inputs["Color"])
        links.new(nor_map.outputs["Normal"], bsdf.inputs["Normal"])
    except FileNotFoundError as e:
        print(f"[WARN] Concrete texture not found: {e}, using flat fallback")
        set_input_default(bsdf, "Base Color", (0.5, 0.48, 0.46, 1.0))

    return mat


def create_rail_material() -> bpy.types.Material:
    # Real stained-pine PBR scan instead of a flat brown BSDF: the near side
    # rail is the largest ramp surface the camera sees, so a photographed wood
    # grain (colour + roughness + normal) is what sells the ramp as a built
    # wooden track rather than a plastic-looking box. Triplanar BOX projection
    # off object coordinates keeps the grain a consistent real-world size on
    # every face of the thin rail without the stretching a single flat UV
    # unwrap would give a long, thin box.
    mat = bpy.data.materials.new("ramp_rail_wood")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        return mat

    set_input_default(bsdf, "Roughness", 0.6)
    set_input_default(bsdf, "Metallic", 0.0)

    asset_dir = POLYHAVEN_DIR / "stained_pine"
    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    mapping = nodes.new(type="ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (1.6, 1.6, 1.6)
    links.new(tex_coord.outputs["Object"], mapping.inputs["Vector"])

    def wood_image(fname: str, colorspace: str) -> bpy.types.ShaderNodeTexImage:
        img = load_image(require_path(asset_dir, fname), colorspace)
        img.colorspace_settings.name = colorspace
        tex = nodes.new(type="ShaderNodeTexImage")
        tex.image = img
        tex.projection = "BOX"
        tex.projection_blend = 0.3
        links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
        return tex

    try:
        diff_tex = wood_image("stained_pine_diff_4k.jpg", "sRGB")
        ao_tex = wood_image("stained_pine_ao_4k.jpg", "Non-Color")
        mix_ao = nodes.new(type="ShaderNodeMixRGB")
        mix_ao.blend_type = "MULTIPLY"
        mix_ao.inputs["Fac"].default_value = 0.6
        links.new(diff_tex.outputs["Color"], mix_ao.inputs["Color1"])
        links.new(ao_tex.outputs["Color"], mix_ao.inputs["Color2"])
        links.new(mix_ao.outputs["Color"], bsdf.inputs["Base Color"])

        rough_tex = wood_image("stained_pine_rough_4k.jpg", "Non-Color")
        links.new(rough_tex.outputs["Color"], bsdf.inputs["Roughness"])

        nor_tex = wood_image("stained_pine_nor_gl_4k.jpg", "Non-Color")
        nor_map = nodes.new(type="ShaderNodeNormalMap")
        nor_map.inputs["Strength"].default_value = 0.7
        links.new(nor_tex.outputs["Color"], nor_map.inputs["Color"])
        links.new(nor_map.outputs["Normal"], bsdf.inputs["Normal"])
    except FileNotFoundError as e:
        print(f"[WARN] Wood rail texture not found: {e}, using flat fallback")
        set_input_default(bsdf, "Base Color", (0.35, 0.22, 0.13, 1.0))

    return mat


def create_stone_material() -> bpy.types.Material:
    # A cast weathered-concrete support block, using the real
    # concrete_floor_damaged_01 scan (the same family as the ground pavement)
    # instead of the old flat noise+colour-ramp grey, which read as smooth
    # foam. Triplanar BOX projection wraps colour/roughness/normal around the
    # block with no UV seams, and a slight multiply tint plus a distinct
    # mapping scale keep it reading as a separate poured block rather than a
    # bump in the driveway it sits on. A light procedural bump on top of the
    # scan normal adds coarse aggregate/pitting the flat scan alone misses.
    mat = bpy.data.materials.new("prop_stone")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        return mat

    set_input_default(bsdf, "Roughness", 0.9)
    set_input_default(bsdf, "Metallic", 0.0)

    asset_dir = POLYHAVEN_DIR / "concrete_floor_damaged_01"
    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    mapping = nodes.new(type="ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (5.0, 5.0, 5.0)
    links.new(tex_coord.outputs["Object"], mapping.inputs["Vector"])

    def concrete_image(fname: str, colorspace: str) -> bpy.types.ShaderNodeTexImage:
        img = load_image(require_path(asset_dir, fname), colorspace)
        img.colorspace_settings.name = colorspace
        tex = nodes.new(type="ShaderNodeTexImage")
        tex.image = img
        tex.projection = "BOX"
        tex.projection_blend = 0.3
        links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
        return tex

    try:
        diff_tex = concrete_image("concrete_floor_damaged_01_diff_4k.jpg", "sRGB")
        ao_tex = concrete_image("concrete_floor_damaged_01_ao_4k.jpg", "Non-Color")
        mix_ao = nodes.new(type="ShaderNodeMixRGB")
        mix_ao.blend_type = "MULTIPLY"
        mix_ao.inputs["Fac"].default_value = 0.6
        links.new(diff_tex.outputs["Color"], mix_ao.inputs["Color1"])
        links.new(ao_tex.outputs["Color"], mix_ao.inputs["Color2"])

        tint = nodes.new(type="ShaderNodeMixRGB")
        tint.blend_type = "MULTIPLY"
        tint.inputs["Fac"].default_value = 1.0
        tint.inputs["Color2"].default_value = (0.72, 0.70, 0.66, 1.0)
        links.new(mix_ao.outputs["Color"], tint.inputs["Color1"])
        links.new(tint.outputs["Color"], bsdf.inputs["Base Color"])

        rough_tex = concrete_image("concrete_floor_damaged_01_rough_4k.jpg", "Non-Color")
        links.new(rough_tex.outputs["Color"], bsdf.inputs["Roughness"])

        nor_tex = concrete_image("concrete_floor_damaged_01_nor_gl_4k.jpg", "Non-Color")
        nor_map = nodes.new(type="ShaderNodeNormalMap")
        nor_map.inputs["Strength"].default_value = 0.8
        links.new(nor_tex.outputs["Color"], nor_map.inputs["Color"])

        grit = nodes.new(type="ShaderNodeTexNoise")
        grit.inputs["Scale"].default_value = 14.0
        grit.inputs["Detail"].default_value = 6.0
        grit.inputs["Roughness"].default_value = 0.7
        links.new(mapping.outputs["Vector"], grit.inputs["Vector"])
        bump = nodes.new(type="ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.2
        bump.inputs["Distance"].default_value = 0.004
        links.new(grit.outputs["Fac"], bump.inputs["Height"])
        links.new(nor_map.outputs["Normal"], bump.inputs["Normal"])
        links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    except FileNotFoundError as e:
        print(f"[WARN] Concrete block texture not found: {e}, using flat fallback")
        set_input_default(bsdf, "Base Color", (0.42, 0.4, 0.37, 1.0))

    return mat


# --- Car import ---------------------------------------------------------
# assets/models/toy_car_model.glb is a Sketchfab-style toy car made of ~1000
# small parts under a deep empty hierarchy. After import the raw joined bbox
# is approximately width(x) 0.71m, length(y) 1.22m, height(z) 0.46m, so it is
# uniformly scaled by CAR_TARGET_LENGTH / its own raw length to match the
# real-world footprint the physics box already uses (0.1008m/0.2337m/0.0651m).
# The model's nose already points toward local -Y, matching CAR_YAW_QUAT_XYZW,
# so no extra yaw rotation is needed before scaling.
CAR_GLB_PATH = WORKSPACE_DIR / "assets" / "models" / "toy_car_model.glb"
CAR_TARGET_LENGTH = 2.0 * CAR_HALF_LENGTH


def world_bbox(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    return Vector((min(xs), min(ys), min(zs))), Vector((max(xs), max(ys), max(zs)))


def import_glb_meshes(glb_path: Path) -> tuple[list[bpy.types.Object], list[str]]:
    if not glb_path.exists():
        raise FileNotFoundError(f"Model not found: {glb_path}")
    existing = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(glb_path))
    imported = [obj for obj in bpy.context.scene.objects if obj not in existing]
    mesh_objs = [obj for obj in imported if obj.type == "MESH"]
    if not mesh_objs:
        raise RuntimeError(f"No mesh found in {glb_path}")
    imported_names = [obj.name for obj in imported]
    return mesh_objs, imported_names


def bake_and_center(mesh_objs: list[bpy.types.Object], new_name: str) -> bpy.types.Object:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in mesh_objs:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objs[0]
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    if len(mesh_objs) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for obj in mesh_objs:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = mesh_objs[0]
        bpy.ops.object.join()

    merged = bpy.context.view_layer.objects.active
    bbox_min, bbox_max = world_bbox(merged)
    center = (bbox_min + bbox_max) / 2.0
    bpy.context.scene.cursor.location = center
    bpy.ops.object.origin_set(type="ORIGIN_CURSOR")
    merged.location = (0.0, 0.0, 0.0)
    merged.rotation_mode = "QUATERNION"
    merged.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
    merged.name = new_name
    return merged


def import_toy_car_master() -> bpy.types.Object:
    mesh_objs, imported_names = import_glb_meshes(CAR_GLB_PATH)

    # Detach from the imported hierarchy and bake transforms.
    bpy.ops.object.select_all(action="DESELECT")
    for obj in mesh_objs:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objs[0]
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.object.select_all(action="DESELECT")

    # Center the whole car in X/Y so the parent empty sits at the geometric
    # center, matching the physics box origin.
    all_corners: list[Vector] = []
    for obj in mesh_objs:
        all_corners.extend(obj.matrix_world @ Vector(c) for c in obj.bound_box)
    x_mid = (min(c.x for c in all_corners) + max(c.x for c in all_corners)) / 2.0
    y_mid = (min(c.y for c in all_corners) + max(c.y for c in all_corners)) / 2.0
    for obj in mesh_objs:
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bmesh.ops.translate(bm, verts=bm.verts, vec=(-x_mid, -y_mid, 0.0))
        bm.to_mesh(obj.data)
        bm.free()
        obj.data.update()

    # Create the parent empty that represents the car rigid body.
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0.0, 0.0, 0.0))
    car_empty = bpy.context.object
    car_empty.name = "toy_car"
    car_empty.rotation_mode = "QUATERNION"

    # Merge all imported meshes into a single body object.
    if mesh_objs:
        bpy.ops.object.select_all(action="DESELECT")
        for obj in mesh_objs:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = mesh_objs[0]
        bpy.ops.object.join()
        body = bpy.context.active_object
        body.name = "toy_car_body"
        body.parent = car_empty

    # Scale the entire car to the physics footprint.
    body_obj = bpy.data.objects.get("toy_car_body")
    if body_obj is not None:
        corners = [body_obj.matrix_world @ Vector(c) for c in body_obj.bound_box]
        raw_length = max(c.y for c in corners) - min(c.y for c in corners)
    else:
        raw_length = 1.0
    scale = CAR_TARGET_LENGTH / raw_length

    car_empty.scale = (scale, scale, scale)
    bpy.ops.object.select_all(action="DESELECT")
    car_empty.select_set(True)
    for child in car_empty.children:
        child.select_set(True)
    bpy.context.view_layer.objects.active = car_empty
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    # Raise geometry so the lowest point aligns with the physics box bottom.
    lowest_z = min(
        (child.matrix_world @ Vector(c)).z
        for child in car_empty.children
        for c in child.bound_box
    )
    offset_z = -CAR_HALF_HEIGHT - lowest_z
    for child in car_empty.children:
        if child.type != "MESH":
            continue
        bm = bmesh.new()
        bm.from_mesh(child.data)
        bmesh.ops.translate(bm, verts=bm.verts, vec=(0.0, 0.0, offset_z))
        bm.to_mesh(child.data)
        bm.free()
        child.data.update()

    # Clean up any leftover imported objects (empties, etc.).
    for name in imported_names:
        obj = bpy.data.objects.get(name)
        if obj is not None and obj != car_empty and obj not in set(car_empty.children_recursive):
            bpy.data.objects.remove(obj, do_unlink=True)

    return car_empty


def place_object(
    name: str, location: tuple[float, float, float], obj: bpy.types.Object,
) -> bpy.types.Object:
    obj.name = name
    obj.location = location
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
    return obj


def look_at(obj: bpy.types.Object, target: tuple[float, float, float]) -> None:
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_area_light(
    name: str,
    location: tuple[float, float, float],
    power: float,
    size: float,
    target: tuple[float, float, float],
    color: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> bpy.types.Object:
    bpy.ops.object.light_add(type="AREA", location=location)
    light = bpy.context.object
    light.name = name
    if hasattr(light.data, "power"):
        light.data.power = float(power)
    else:
        light.data.energy = float(power)
    light.data.size = float(size)
    light.data.color = color
    light.visible_camera = False
    look_at(light, target)
    return light


def output_path(out_dir: Path, filename: str) -> Path:
    return (out_dir / filename).resolve()


def write_scenario_metadata(out_dir: Path, scenario: dict[str, object]) -> None:
    output_path(out_dir, SCENARIO_METADATA_NAME).write_text(
        json.dumps(scenario, indent=2), encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, object]:
    data = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def recursive_update(base: dict[str, object], updates: dict[str, object]) -> dict[str, object]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = recursive_update(merged[key], value)  # type: ignore[arg-type]
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def create_scenario(args: argparse.Namespace) -> dict[str, object]:
    if args.scenario_json is not None:
        scenario = read_json(args.scenario_json)
        scenario.setdefault(
            "scenario_source", str(args.scenario_json.expanduser().resolve()),
        )
    else:
        seed = int(args.seed)
        scenario = {
            "schema_version": 1,
            "seed": seed,
            "render": {
                "fps": int(args.fps),
                "duration_sec": float(args.duration_sec),
                "resolution": [int(args.resolution[0]), int(args.resolution[1])],
                "samples": int(args.samples),
                "device": str(args.device),
                "mode": str(args.mode),
            },
            "camera": {
                "location": list(CAMERA_LOCATION),
                "target": list(CAMERA_TARGET),
                "lens_mm": CAMERA_LENS_MM,
            },
            "physics": {
                "car_mass": 0.35,
                "car_restitution": 0.05,
                "ramp_angle_deg": RAMP_ANGLE_DEG,
                "ramp_length": RAMP_LENGTH,
                "ramp_width": RAMP_WIDTH,
                "ramp_thickness": RAMP_THICKNESS,
                "ramp_friction": 0.25,
                "floor_friction": 0.9,
                # Raised slightly from 2.6 so the car clears the ramp lip with
                # enough speed to complete its rotation in the air and land
                # flat, bottom-down (at 2.6 it under-rotates and settles on its
                # roof). Kept at 2.7 rather than higher so the car lands close
                # enough to the ramp to stay inside this camera's frame after
                # it comes to rest. See simulate_car_ramp_climb.py's note.
                "launch_speed": 2.7,
                "gravity": [0.0, 0.0, -9.8],
            },
            "surface": "asphalt_grey",
            "jitter": {
                "preview_frame": int(args.preview_frame),
            },
        }

    if args.scenario_overrides_json is not None:
        overrides = read_json(args.scenario_overrides_json)
        scenario = recursive_update(scenario, overrides)
        scenario["scenario_overrides_path"] = str(
            args.scenario_overrides_json.expanduser().resolve()
        )

    return scenario


def run_physics_simulation(args: argparse.Namespace, scenario: dict[str, object]) -> dict[str, Any]:
    physics_python = WORKSPACE_DIR.parent / "miniconda" / "envs" / "physics" / "bin" / "python"
    if physics_python.exists():
        python = str(physics_python)
    else:
        python = shutil.which("python3") or shutil.which("python")
    if not python:
        raise RuntimeError("Cannot find python3/python for the PyBullet physics simulation.")

    physics = scenario["physics"]
    assert isinstance(physics, dict)

    script_path = Path(__file__).with_name("simulate_car_ramp_climb.py")
    physics_path = args.out_dir / PHYSICS_TEMP_NAME
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
            "--car-mass",
            str(float(physics["car_mass"])),
            "--car-restitution",
            str(float(physics["car_restitution"])),
            "--ramp-angle-deg",
            str(float(physics["ramp_angle_deg"])),
            "--ramp-length",
            str(float(physics["ramp_length"])),
            "--ramp-width",
            str(float(physics["ramp_width"])),
            "--ramp-thickness",
            str(float(physics["ramp_thickness"])),
            "--ramp-friction",
            str(float(physics["ramp_friction"])),
            "--floor-friction",
            str(float(physics["floor_friction"])),
            "--launch-speed",
            str(float(physics["launch_speed"])),
            "--gravity-z",
            str(float(physics["gravity"][2])),
        ],
        check=True,
    )
    records = json.loads(physics_path.read_text(encoding="utf-8"))
    physics_path.unlink(missing_ok=True)
    return records


def set_linear_keyframes(objects) -> None:
    for obj in objects:
        if obj.animation_data and obj.animation_data.action:
            for fcurve in obj.animation_data.action.fcurves:
                for key in fcurve.keyframe_points:
                    key.interpolation = "LINEAR"


def apply_physics_animation(car: bpy.types.Object, physics: dict) -> None:
    car.rotation_mode = "QUATERNION"

    for frame_record in physics["frames"]:
        frame = int(frame_record["frame_index"])
        data = frame_record["car"]
        quat = data["quaternion_xyzw"]
        car.location = data["location"]
        car.rotation_quaternion = (quat[3], quat[0], quat[1], quat[2])
        car.keyframe_insert(data_path="location", frame=frame)
        car.keyframe_insert(data_path="rotation_quaternion", frame=frame)

    set_linear_keyframes([car])


def export_ground_truth(
    out_dir: Path,
    car: bpy.types.Object,
    camera: bpy.types.Object,
    frame_end: int,
    fps: int,
    physics: dict,
    scenario: dict[str, object],
) -> None:
    scene = bpy.context.scene
    records = {
        "schema_version": 1,
        "fps": int(fps),
        "frame_start": 1,
        "frame_end": int(frame_end),
        "scenario_metadata_path": str(output_path(out_dir, SCENARIO_METADATA_NAME)),
        "physics": {key: value for key, value in physics.items() if key != "frames"},
        "objects": {"car": {"object_name": car.name}},
        "camera": {
            "object_name": camera.name,
            "lens_mm": float(camera.data.lens),
            "sensor_width_mm": float(camera.data.sensor_width),
            "resolution": [
                int(scene.render.resolution_x), int(scene.render.resolution_y),
            ],
        },
        "scenario": {"seed": int(scenario["seed"]), "surface": scenario.get("surface")},
        "frames": [],
    }
    physics_by_frame = {
        int(frame_record["frame_index"]): frame_record for frame_record in physics["frames"]
    }
    for frame in range(1, frame_end + 1):
        scene.frame_set(frame)
        physics_frame = physics_by_frame[frame]
        records["frames"].append({
            "frame_index": frame,
            "time_sec": (frame - 1) / float(fps),
            "car": {
                "matrix_world": [[float(v) for v in row] for row in car.matrix_world],
                "location": [float(v) for v in car.location],
                "linear_velocity": physics_frame["car"]["linear_velocity"],
                "angular_velocity": physics_frame["car"]["angular_velocity"],
                "ramp_local_x": physics_frame["car"]["ramp_local_x"],
            },
            "camera_matrix_world": [[float(v) for v in row] for row in camera.matrix_world],
            "camera_world_to_camera_matrix": [
                [float(v) for v in row] for row in camera.matrix_world.inverted()
            ],
        })
    output_path(out_dir, GROUND_TRUTH_NAME).write_text(
        json.dumps(records, indent=2), encoding="utf-8",
    )


def build_scene(
    args: argparse.Namespace, scenario: dict[str, object],
) -> tuple[bpy.types.Object, bpy.types.Object]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)

    scene = bpy.context.scene
    scene.render.resolution_x = args.resolution[0]
    scene.render.resolution_y = args.resolution[1]
    scene.render.fps = args.fps
    scene.render.engine = "CYCLES"
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.cycles.samples = args.samples
    scene.cycles.device = "GPU" if args.device == "auto" else "CPU"
    scene.cycles.max_bounces = 12
    scene.cycles.transmission_bounces = 8

    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    world_nodes = world.node_tree.nodes
    world_links = world.node_tree.links
    for node in world_nodes:
        world_nodes.remove(node)

    # Outdoor partly-cloudy sky HDRI -- a flat solid-color background read as
    # an obviously fake studio backdrop; this scene is meant to look like a
    # toy ramp set up in a real outdoor courtyard/patio.
    env_tex = world_nodes.new(type="ShaderNodeTexEnvironment")
    env_tex.location = (-300, 0)
    hdri_path = POLYHAVEN_DIR / "outdoor" / "kloofendal_48d_partly_cloudy_puresky_2k.hdr"
    if hdri_path.exists():
        hdri_img = bpy.data.images.load(str(hdri_path), check_existing=True)
        env_tex.image = hdri_img
        print("[INFO] Using HDRI background: kloofendal_48d_partly_cloudy_puresky (outdoor)")
    else:
        print(f"[WARN] HDRI not found at {hdri_path}, using solid fallback")
        bg_node = world_nodes.new(type="ShaderNodeBackground")
        bg_node.inputs["Color"].default_value = (0.5, 0.55, 0.6, 1.0)
        bg_node.inputs["Strength"].default_value = 1.1
        env_tex = bg_node

    mapping_node = world_nodes.new(type="ShaderNodeMapping")
    mapping_node.location = (-550, 0)
    mapping_node.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(200))
    tex_coord_node = world_nodes.new(type="ShaderNodeTexCoord")
    tex_coord_node.location = (-750, 0)
    output_node = world_nodes.new(type="ShaderNodeOutputWorld")

    if isinstance(env_tex, bpy.types.ShaderNodeTexEnvironment):
        world_links.new(tex_coord_node.outputs["Generated"], mapping_node.inputs["Vector"])
        world_links.new(mapping_node.outputs["Vector"], env_tex.inputs["Vector"])
        bg_node = world_nodes.new(type="ShaderNodeBackground")
        bg_node.location = (0, 0)
        bg_node.inputs["Strength"].default_value = 1.0
        world_links.new(env_tex.outputs["Color"], bg_node.inputs["Color"])
        world_links.new(bg_node.outputs["Background"], output_node.inputs["Surface"])
    else:
        world_links.new(env_tex.outputs["Background"], output_node.inputs["Surface"])

    bpy.ops.object.camera_add(location=CAMERA_LOCATION)
    camera = bpy.context.object
    camera.data.lens = CAMERA_LENS_MM
    scene.camera = camera
    target_obj = bpy.data.objects.new("camera_target", None)
    scene.collection.objects.link(target_obj)
    target_obj.location = CAMERA_TARGET
    constraint = camera.constraints.new("TRACK_TO")
    constraint.target = target_obj
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"

    bpy.ops.object.light_add(type="SUN", location=(1.5, -1.0, 3.0))
    sun = bpy.context.object
    sun.data.energy = 4.5
    sun.data.color = (1.0, 0.97, 0.9)
    sun.data.angle = math.radians(1.5)
    sun.rotation_euler = (math.radians(45), math.radians(10), math.radians(60))

    add_area_light(
        "fill_light", location=(-0.8, -0.8, 1.0), power=25, size=0.8, target=LIGHT_TARGET,
    )

    physics = scenario["physics"]
    assert isinstance(physics, dict)
    ramp_angle = math.radians(float(physics["ramp_angle_deg"]))
    ramp_length = float(physics["ramp_length"])
    ramp_width = float(physics["ramp_width"])
    ramp_thickness = float(physics["ramp_thickness"])
    cos_a = math.cos(ramp_angle)
    sin_a = math.sin(ramp_angle)
    ramp_center_z = FLOOR_Z + ramp_length / 2.0 * sin_a + ramp_thickness / 2.0 * cos_a

    # Static background: the house model supplies both the ground (grass,
    # driveway, path) and the wall the ramp is set up against, replacing the
    # earlier procedural ground plane + backdrop wall.
    if not HOUSE_GLB_PATH.exists():
        raise FileNotFoundError(f"House model not found: {HOUSE_GLB_PATH}")
    bpy.ops.import_scene.gltf(filepath=str(HOUSE_GLB_PATH))

    # A static empty that positions the whole ramp assembly (built below in
    # its own local frame) into the house's side yard, against the garage
    # wall -- see the PLACEMENT_LOCATION/PLACEMENT_ROTATION_Z_DEG comment
    # above for the coordinate mapping.
    placement = bpy.data.objects.new("scene_placement", None)
    scene.collection.objects.link(placement)
    placement.location = PLACEMENT_LOCATION
    placement.rotation_euler = (0.0, 0.0, math.radians(PLACEMENT_ROTATION_Z_DEG))

    def attach_to_placement(obj: bpy.types.Object) -> None:
        # Identity, not placement's inverse: the ramp assembly's own
        # location/rotation (physics-frame local coordinates, including the
        # animated keyframes applied later) should compose directly with
        # PLACEMENT's transform, not be canceled out by it.
        obj.parent = placement
        obj.matrix_parent_inverse = Matrix.Identity(4)

    # Ramp: a tilted box (surface material swaps per scenario) with two side
    # rails so the car reads as running in a track instead of floating on a
    # bare plank.
    surface_name = str(scenario.get("surface", "turf_green"))
    surface_factory = RAMP_SURFACES.get(surface_name, RAMP_SURFACES["turf_green"])
    ramp_mat = surface_factory()

    ramp_euler = (0.0, ramp_angle, 0.0)
    bpy.ops.mesh.primitive_cube_add(
        size=1.0, location=(0.0, 0.0, ramp_center_z), rotation=ramp_euler,
    )
    ramp = bpy.context.object
    ramp.name = "ramp_surface"
    ramp.dimensions = (ramp_length, ramp_width, ramp_thickness)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    ramp.data.materials.append(ramp_mat)
    # Chamfer the ramp edges: a razor-sharp box edge is an instant give-away
    # that this is CG. A small bevel catches a highlight along each edge the
    # way a real cut board or slab does.
    ramp.data.use_auto_smooth = True
    ramp_bevel = ramp.modifiers.new("bevel", "BEVEL")
    ramp_bevel.width = 0.006
    ramp_bevel.segments = 2
    attach_to_placement(ramp)

    rail_mat = create_rail_material()
    for side in (-1.0, 1.0):
        bpy.ops.mesh.primitive_cube_add(
            size=1.0,
            location=(0.0, side * (ramp_width / 2.0 + RAIL_THICKNESS / 2.0), ramp_center_z),
            rotation=ramp_euler,
        )
        rail = bpy.context.object
        rail.name = f"ramp_rail_{'l' if side < 0 else 'r'}"
        rail.dimensions = (ramp_length, RAIL_THICKNESS, ramp_thickness + RAIL_HEIGHT)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        rail.location.z += (RAIL_HEIGHT / 2.0) * math.cos(ramp_angle)
        rail.location.x += (RAIL_HEIGHT / 2.0) * math.sin(ramp_angle)
        rail.data.materials.append(rail_mat)
        rail.data.use_auto_smooth = True
        rail_bevel = rail.modifiers.new("bevel", "BEVEL")
        rail_bevel.width = 0.004
        rail_bevel.segments = 2
        attach_to_placement(rail)

    # A stone prop wedged under the ramp's high (top) end -- without
    # something physically propping it up, an inclined ramp just floating
    # in contact with the wall reads as obviously fake.
    support_top_x = -(ramp_length / 2.0) * cos_a - (ramp_thickness / 2.0) * sin_a
    support_top_z = (ramp_length / 2.0) * sin_a - (ramp_thickness / 2.0) * cos_a + ramp_center_z
    stone_mat = create_stone_material()
    stone_depth = 0.16
    bpy.ops.mesh.primitive_cube_add(
        size=1.0, location=(support_top_x, 0.0, support_top_z / 2.0),
    )
    stone = bpy.context.object
    stone.name = "support_stone"
    stone.dimensions = (stone_depth, ramp_width * 0.85, support_top_z)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    stone.data.use_auto_smooth = True
    stone_bevel = stone.modifiers.new("bevel", "BEVEL")
    # Wider, rounder chamfer than before: poured/weathered concrete has
    # noticeably knocked-back edges, not the near-sharp corners of the old
    # 12mm/2-segment bevel.
    stone_bevel.width = 0.02
    stone_bevel.segments = 3
    stone.data.materials.append(stone_mat)
    attach_to_placement(stone)

    car_empty = import_toy_car_master()
    car = place_object("toy_car", (0.0, 0.0, ramp_center_z + 0.2), car_empty)
    attach_to_placement(car)

    frame_end = max(2, int(round(float(args.duration_sec) * int(args.fps))))
    scene.frame_start = 1
    scene.frame_end = frame_end

    return car, camera


def render_preview(args: argparse.Namespace) -> None:
    scene = bpy.context.scene
    preview_frame = max(scene.frame_start, min(int(args.preview_frame), scene.frame_end))
    scene.frame_set(preview_frame)
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(output_path(args.out_dir, "preview.png"))
    bpy.ops.render.render(write_still=True)


def render_animation(args: argparse.Namespace) -> None:
    scene = bpy.context.scene
    scene.frame_set(scene.frame_start)
    scene.render.filepath = str(output_path(args.out_dir, DIRECT_MP4_NAME))
    bpy.ops.render.render(animation=True)


def render_frames(args: argparse.Namespace) -> None:
    scene = bpy.context.scene
    scene.frame_set(scene.frame_start)
    scene.render.image_settings.file_format = "PNG"
    frame_end = min(20, scene.frame_end)
    scene.frame_end = frame_end
    scene.render.filepath = str(output_path(args.out_dir, "frame_"))
    bpy.ops.render.render(animation=True)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    scenario = create_scenario(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_scenario_metadata(out_dir, scenario)

    car, camera = build_scene(args, scenario)
    physics = run_physics_simulation(args, scenario)
    apply_physics_animation(car, physics)
    export_ground_truth(
        out_dir, car, camera, bpy.context.scene.frame_end, int(args.fps), physics, scenario,
    )

    if args.mode == "preview":
        render_preview(args)
    elif args.mode == "frames":
        render_frames(args)
    else:
        render_animation(args)

    bpy.ops.wm.save_as_mainfile(filepath=str(output_path(out_dir, BLEND_NAME)))
    print(f"[INFO] Render complete. Output: {out_dir}")


if __name__ == "__main__":
    main()
