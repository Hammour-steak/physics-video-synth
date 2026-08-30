from __future__ import annotations

import argparse
import copy
import json
import math
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bpy
from mathutils import Vector


WORKSPACE_DIR = Path(__file__).resolve().parents[2]
POLYHAVEN_DIR = WORKSPACE_DIR / "assets" / "polyhaven"
AMBIENTCG_DIR = WORKSPACE_DIR / "assets" / "ambientcg"
MODELS_DIR = WORKSPACE_DIR / "assets" / "models"
BALL_GLB_PATH = MODELS_DIR / "basketball_and1_xcelerate.glb"

OUTPUT_STEM = "bouncing_ball"
DIRECT_MP4_NAME = f"{OUTPUT_STEM}.mp4"
BLEND_NAME = f"{OUTPUT_STEM}.blend"
GROUND_TRUTH_NAME = "ground_truth_transforms.json"
PHYSICS_TEMP_NAME = "physics_transforms.json"
SCENARIO_METADATA_NAME = "scenario_metadata.json"

FLOOR_SIZE = 10.0
FLOOR_THICKNESS = 0.1
FLOOR_Z = 0.0
BALL_RADIUS = 0.25

CAMERA_LOCATION = (1.9, -7.7, 2.00)
CAMERA_TARGET = (0.0, 0.0, 0.45)
LIGHT_TARGET = (0.0, 0.0, 0.45)


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


FLOOR_TEXTURE = PBRTextureSet(
    asset_name="wood_floor_worn",
    material_name="worn wood floor",
    diffuse_name="wood_floor_worn_diff_4k.jpg",
    roughness_name="wood_floor_worn_rough_4k.jpg",
    normal_name="wood_floor_worn_nor_gl_4k.jpg",
    ao_name="wood_floor_worn_ao_4k.jpg",
    height_name="wood_floor_worn_disp_4k.png",
    normal_strength=0.45,
    height_bump_strength=0.050,
    height_bump_distance=0.008,
)


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
    parser.add_argument("--preview-frame", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cpu"), default="cpu")
    parser.add_argument("--seed", type=int, default=13)
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


def polyhaven_asset_dir(asset_name: str) -> Path:
    return POLYHAVEN_DIR / asset_name


def require_polyhaven_path(asset_name: str, filename: str) -> Path:
    asset_dir = polyhaven_asset_dir(asset_name)
    if not asset_dir.exists():
        raise FileNotFoundError(f"Poly Haven asset directory missing: {asset_dir}")
    file_path = asset_dir / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Poly Haven texture missing: {file_path}")
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


def create_floor_material(scenario: dict[str, object]) -> bpy.types.Material:
    mat = bpy.data.materials.new("wood_floor_surface")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")

    if bsdf is not None:
        set_input_default(bsdf, "Roughness", 0.45)
        set_input_default(bsdf, "Metallic", 0.0)
        set_input_default(bsdf, "Base Color", (0.45, 0.32, 0.20, 1.0))

        tex_coord = nodes.new(type="ShaderNodeTexCoord")
        mapping = nodes.new(type="ShaderNodeMapping")
        mapping.inputs["Scale"].default_value = (4.0, 4.0, 4.0)
        links.new(tex_coord.outputs["UV"], mapping.inputs["Vector"])

        try:
            diff_path = require_polyhaven_path(FLOOR_TEXTURE.asset_name, FLOOR_TEXTURE.diffuse_name)
            diff_img = load_image(diff_path, "sRGB")
            diff_img.colorspace_settings.name = 'sRGB'
            diff_tex = nodes.new(type="ShaderNodeTexImage")
            diff_tex.image = diff_img
            links.new(mapping.outputs["Vector"], diff_tex.inputs["Vector"])
            links.new(diff_tex.outputs["Color"], bsdf.inputs["Base Color"])

            rough_path = require_polyhaven_path(FLOOR_TEXTURE.asset_name, FLOOR_TEXTURE.roughness_name)
            rough_img = load_image(rough_path, "Non-Color")
            rough_img.colorspace_settings.name = 'Non-Color'
            rough_tex = nodes.new(type="ShaderNodeTexImage")
            rough_tex.image = rough_img
            links.new(mapping.outputs["Vector"], rough_tex.inputs["Vector"])
            links.new(rough_tex.outputs["Color"], bsdf.inputs["Roughness"])

            nor_path = require_polyhaven_path(FLOOR_TEXTURE.asset_name, FLOOR_TEXTURE.normal_name)
            nor_img = load_image(nor_path, "Non-Color")
            nor_img.colorspace_settings.name = 'Non-Color'
            nor_tex = nodes.new(type="ShaderNodeTexImage")
            nor_tex.image = nor_img
            nor_map = nodes.new(type="ShaderNodeNormalMap")
            nor_map.inputs["Strength"].default_value = FLOOR_TEXTURE.normal_strength
            links.new(mapping.outputs["Vector"], nor_tex.inputs["Vector"])
            links.new(nor_tex.outputs["Color"], nor_map.inputs["Color"])
            links.new(nor_map.outputs["Normal"], bsdf.inputs["Normal"])

            print(f"[INFO] Using floor texture: {FLOOR_TEXTURE.asset_name}")
        except FileNotFoundError as e:
            print(f"[WARN] Floor texture not found: {e}, using procedural fallback")
            set_input_default(bsdf, "Base Color", (0.45, 0.32, 0.20, 1.0))

    return mat


def require_ambientcg_path(asset_name: str, filename: str) -> Path:
    path = AMBIENTCG_DIR / asset_name / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing render asset: {path}. Run scripts/download_render_assets.py first."
        )
    return path


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


def create_steel_ball_material(scenario: dict[str, object]) -> bpy.types.Material:
    mat = bpy.data.materials.new("steel_ball")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    nodes.clear()

    output = nodes.new(type="ShaderNodeOutputMaterial")
    output.location = (800, 0)

    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (400, 0)
    bsdf.inputs["Metallic"].default_value = 0.95
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    tex_coord.location = (-1000, 0)

    mapping = nodes.new(type="ShaderNodeMapping")
    mapping.location = (-800, 0)
    mapping.inputs["Scale"].default_value = (3.0, 3.0, 3.0)
    links.new(tex_coord.outputs["UV"], mapping.inputs["Vector"])

    metal_dir = AMBIENTCG_DIR / "Metal032"
    rubber_dir = AMBIENTCG_DIR / "Rubber002"

    def load_tex(path, filename, colorspace="Non-Color"):
        full_path = path / filename
        img = bpy.data.images.load(str(full_path), check_existing=True)
        img.colorspace_settings.name = colorspace
        tex = nodes.new(type="ShaderNodeTexImage")
        tex.image = img
        links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
        return tex

    try:
        color_tex = load_tex(metal_dir, "Metal032_4K-JPG_Color.jpg", colorspace="sRGB")
        color_tex.location = (-550, 300)

        hue_sat = nodes.new(type="ShaderNodeHueSaturation")
        hue_sat.location = (-350, 300)
        hue_sat.inputs["Saturation"].default_value = 0.6
        links.new(color_tex.outputs["Color"], hue_sat.inputs["Color"])
        links.new(hue_sat.outputs["Color"], bsdf.inputs["Base Color"])

        rough_tex = load_tex(metal_dir, "Metal032_4K-JPG_Roughness.jpg")
        rough_tex.location = (-550, -100)

        rubber_rough_tex = load_tex(rubber_dir, "Rubber002_4K-JPG_Roughness.jpg")
        rubber_rough_tex.location = (-550, -250)

        mix_rough = nodes.new(type="ShaderNodeMix")
        mix_rough.location = (-350, -180)
        mix_rough.data_type = "RGBA"
        mix_rough.inputs["Factor"].default_value = 0.4
        links.new(rough_tex.outputs["Color"], mix_rough.inputs["A"])
        links.new(rubber_rough_tex.outputs["Color"], mix_rough.inputs["B"])

        rough_adjust = nodes.new(type="ShaderNodeMath")
        rough_adjust.location = (-150, -180)
        rough_adjust.operation = "MULTIPLY"
        rough_adjust.inputs[1].default_value = 0.6
        links.new(mix_rough.outputs["Result"], rough_adjust.inputs[0])
        links.new(rough_adjust.outputs["Value"], bsdf.inputs["Roughness"])

        normal_tex = load_tex(metal_dir, "Metal032_4K-JPG_NormalGL.jpg")
        normal_tex.location = (-550, -400)

        normal_map = nodes.new(type="ShaderNodeNormalMap")
        normal_map.location = (-300, -400)
        normal_map.inputs["Strength"].default_value = 0.2
        links.new(normal_tex.outputs["Color"], normal_map.inputs["Color"])
        links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])

        disp_tex = load_tex(metal_dir, "Metal032_4K-JPG_Displacement.jpg")
        disp_tex.location = (-550, -600)

        bump = nodes.new(type="ShaderNodeBump")
        bump.location = (-300, -600)
        bump.inputs["Strength"].default_value = 0.01
        bump.inputs["Distance"].default_value = 0.0005
        links.new(disp_tex.outputs["Color"], bump.inputs["Height"])
        links.new(bump.outputs["Normal"], normal_map.inputs["Color"])

    except RuntimeError as e:
        print(f"[WARN] Metal texture not found ({e}), using procedural fallback")
        bsdf.inputs["Base Color"].default_value = (0.45, 0.45, 0.48, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.2

    return mat


def create_rubber_ball_material(scenario: dict[str, object]) -> bpy.types.Material:
    """Rubber ball material matching scripts/ball_block/render_ball_block_impact.py."""
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


BALL_SEAMS = (
    ("ball_equator_seam", (0.0, 0.0, 0.0)),
    ("ball_vertical_seam", (math.radians(90.0), 0.0, 0.0)),
    ("ball_diagonal_seam", (0.0, math.radians(63.0), 0.0)),
)

BALL_SCUFF_PATCHES = (
    ((0.56, -0.62, 0.54), 0.033, 0.010, 0.2),
    ((0.38, -0.82, 0.42), 0.020, 0.008, 1.1),
    ((-0.66, 0.48, 0.57), 0.024, 0.007, 2.3),
    ((-0.28, 0.74, 0.61), 0.030, 0.009, 0.7),
    ((0.22, 0.30, -0.93), 0.028, 0.008, 1.5),
    ((-0.20, -0.44, -0.88), 0.022, 0.006, 0.4),
)


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


def add_box(
    name: str,
    location: tuple[float, float, float],
    dimensions: tuple[float, float, float],
    material: bpy.types.Material,
    *,
    rotation_z: float = 0.0,
    bevel_width: float = 0.0,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location, rotation=(0.0, 0.0, rotation_z))
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


def look_at(obj: bpy.types.Object, target: tuple[float, float, float]) -> None:
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


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


def add_room_shell(scenario: dict[str, object]) -> None:
    wall_mat = create_wall_material(scenario)
    baseboard_mat = create_baseboard_material(scenario)
    add_box("matte back plaster wall", (0.0, 3.05, 1.45), (8.6, 0.08, 2.90), wall_mat)
    add_box("matte right plaster wall", (4.20, -0.55, 1.45), (0.08, 7.2, 2.90), wall_mat)
    add_box("back wall baseboard", (0.0, 2.995, 0.13), (8.5, 0.09, 0.16), baseboard_mat, bevel_width=0.006)
    add_box("right wall baseboard", (4.145, -0.55, 0.13), (0.09, 7.1, 0.16), baseboard_mat, bevel_width=0.006)


def create_floor(scenario: dict[str, object]) -> bpy.types.Object:
    bpy.ops.mesh.primitive_plane_add(size=100.0, location=(0.0, 0.0, FLOOR_Z))
    floor = bpy.context.object
    floor.name = "floor"
    floor.data.materials.append(create_floor_material(scenario))
    return floor


def add_ball(radius: float, location: tuple[float, float, float], scenario: dict[str, object]) -> bpy.types.Object:
    """Import the basketball GLB, rescale it to `radius`, and place it.

    The GLB carries its own material (orange rubber with grooves and lines);
    the procedural rubber/seam/scuff pipeline that used to build the ball
    from primitives is no longer applied -- keeping the GLB's authored look
    is the whole point of switching to it.
    """
    if not BALL_GLB_PATH.exists():
        raise FileNotFoundError(f"Basketball model missing: {BALL_GLB_PATH}")

    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(BALL_GLB_PATH))
    imported = [obj for obj in bpy.context.scene.objects if obj not in before]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    empties = [obj for obj in imported if obj.type != "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh found in {BALL_GLB_PATH}")

    # Bake the glTF parent transforms into vertex data and join into one
    # object so the bounding box tells us the true world-space diameter.
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    if len(meshes) > 1:
        bpy.ops.object.join()
    ball = bpy.context.view_layer.objects.active
    ball.name = "bouncing_ball"

    # Drop the empties the glTF importer left behind.
    for empty in empties:
        leftover = bpy.data.objects.get(empty.name)
        if leftover is not None and leftover is not ball:
            bpy.data.objects.remove(leftover, do_unlink=True)

    # Uniform rescale so the mesh's largest dimension matches the physics
    # radius (2 * radius across).
    max_dim = max(ball.dimensions)
    if max_dim <= 0.0:
        raise RuntimeError(f"Basketball mesh has zero dimension: {ball.dimensions}")
    scale = (2.0 * radius) / max_dim
    ball.scale = (scale, scale, scale)
    bpy.ops.object.select_all(action="DESELECT")
    ball.select_set(True)
    bpy.context.view_layer.objects.active = ball
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    # Recentre the origin on the bounding-box centre so location is the
    # ball's true world centre -- the physics keyframes assume that.
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")

    ball.location = location
    ball.rotation_mode = "QUATERNION"
    ball.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
    bpy.ops.object.shade_smooth()
    return ball


def output_path(out_dir: Path, filename: str) -> Path:
    return (out_dir / filename).resolve()


def write_scenario_metadata(out_dir: Path, scenario: dict[str, object]) -> None:
    output_path(out_dir, SCENARIO_METADATA_NAME).write_text(
        json.dumps(scenario, indent=2),
        encoding="utf-8",
    )


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
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = recursive_update(merged[key], value)  # type: ignore[arg-type]
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def create_scenario(args: argparse.Namespace) -> dict[str, object]:
    if args.scenario_json is not None:
        scenario = read_json(args.scenario_json)
        scenario.setdefault(
            "scenario_source",
            str(args.scenario_json.expanduser().resolve()),
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
                "hdri_rotation_z_deg": 180.0,
            },
            "camera": {
                "location": list(CAMERA_LOCATION),
                "target": list(CAMERA_TARGET),
                "lens_mm": 50.0,
            },
            "lighting": {
                "hdri_rotation_deg": 128.0,
                "hdri_strength": 0.30,
                "key_power": 285.0,
                "key_size": 5.1,
                "key_color": [1.0, 1.0, 1.0, 1.0],
                "bounce_power": 55.0,
                "bounce_color": [1.0, 0.78, 0.55, 1.0],
                "wall_fill_power": 10.0,
                "fill_color": [0.82, 0.86, 0.90, 1.0],
            },
            "materials": {
                "wall_color": [0.70, 0.66, 0.60, 1.0],
                "baseboard_color": [0.78, 0.74, 0.68, 1.0],
                "ball_base_color": [0.72, 0.075, 0.025, 1.0],
                "floor_tint": [0.52, 0.34, 0.20, 1.0],
            },
            "physics": {
                "ball_radius": BALL_RADIUS,
                "ball_mass": 1.0,
                "ball_friction": 0.38,
                "ball_restitution": 0.78,
                "ball_initial_location": [0.0, 0.0, BALL_RADIUS + 1.32],
                "ball_initial_velocity": [0.5, 0.0, -0.18],
                "floor_friction": 0.82,
                "gravity": [0.0, 0.0, -9.81],
            },
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
    python = shutil.which("python3") or shutil.which("python")
    if not python:
        raise RuntimeError("Cannot find python3/python for the PyBullet physics simulation.")

    physics = scenario["physics"]
    assert isinstance(physics, dict)

    script_path = Path(__file__).with_name("simulate_bouncing_ball.py")
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
            "--ball-radius",
            str(float(physics["ball_radius"])),
            "--ball-mass",
            str(float(physics["ball_mass"])),
            "--ball-initial-location",
            str(float(physics["ball_initial_location"][0])),
            str(float(physics["ball_initial_location"][1])),
            str(float(physics["ball_initial_location"][2])),
            "--ball-initial-velocity",
            str(float(physics["ball_initial_velocity"][0])),
            str(float(physics["ball_initial_velocity"][1])),
            str(float(physics["ball_initial_velocity"][2])),
            "--floor-friction",
            str(float(physics["floor_friction"])),
            "--ball-friction",
            str(float(physics["ball_friction"])),
            "--ball-restitution",
            str(float(physics["ball_restitution"])),
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


def apply_physics_animation(
    ball: bpy.types.Object,
    physics: dict,
) -> None:
    ball.rotation_mode = "QUATERNION"

    for frame_record in physics["frames"]:
        frame = int(frame_record["frame_index"])

        ball_quat = frame_record["ball_quaternion_xyzw"]
        ball.location = frame_record["ball_location"]
        ball.rotation_quaternion = (
            ball_quat[3],
            ball_quat[0],
            ball_quat[1],
            ball_quat[2],
        )
        ball.keyframe_insert(data_path="location", frame=frame)
        ball.keyframe_insert(data_path="rotation_quaternion", frame=frame)

    set_linear_keyframes([ball])


def export_ground_truth(
    out_dir: Path,
    ball: bpy.types.Object,
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
        "objects": {
            "ball": {
                "object_name": ball.name,
                "radius_m_scene_units": float(ball.dimensions[0] / 2),
            },
        },
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
        },
        "frames": [],
    }
    physics_by_frame = {
        int(frame_record["frame_index"]): frame_record
        for frame_record in physics["frames"]
    }
    for frame in range(1, frame_end + 1):
        scene.frame_set(frame)
        physics_frame = physics_by_frame[frame]
        records["frames"].append(
            {
                "frame_index": frame,
                "time_sec": (frame - 1) / float(fps),
                "ball_matrix_world": [[float(v) for v in row] for row in ball.matrix_world],
                "ball_location": [float(v) for v in ball.location],
                "ball_linear_velocity": physics_frame["ball_linear_velocity"],
                "ball_angular_velocity": physics_frame["ball_angular_velocity"],
                "ball_floor_gap": physics_frame["ball_floor_gap"],
                "camera_matrix_world": [[float(v) for v in row] for row in camera.matrix_world],
                "camera_world_to_camera_matrix": [
                    [float(v) for v in row]
                    for row in camera.matrix_world.inverted()
                ],
            }
        )
    output_path(out_dir, GROUND_TRUTH_NAME).write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )


def scene_constants(scenario: dict[str, object]) -> dict[str, float]:
    physics = scenario["physics"]
    assert isinstance(physics, dict)
    return {
        "ball_radius": float(physics["ball_radius"]),
    }


def build_scene(args: argparse.Namespace, scenario: dict[str, object]) -> tuple[bpy.types.Object, bpy.types.Object]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    const = scene_constants(scenario)
    ball_radius = const["ball_radius"]

    bpy.ops.wm.read_factory_settings(use_empty=True)

    scene = bpy.context.scene
    scene.render.resolution_x = args.resolution[0]
    scene.render.resolution_y = args.resolution[1]
    scene.render.fps = args.fps
    scene.render.engine = "CYCLES"
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'
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

    lighting = scenario.get("lighting", {})
    hdri_rotation_deg = float(lighting.get("hdri_rotation_deg", 180.0))
    hdri_strength = float(lighting.get("hdri_strength", 0.8))

    env_tex = world_nodes.new(type="ShaderNodeTexEnvironment")
    env_tex.location = (-300, 0)

    hdri_path = POLYHAVEN_DIR / "brown_photostudio_05" / "brown_photostudio_05_2k.hdr"
    if hdri_path.exists():
        env_tex.image = load_image(hdri_path, "Linear")
        print(f"[INFO] Using HDRI background: brown_photostudio_05")
    else:
        print(f"[WARN] HDRI not found at {hdri_path}, using solid fallback")

    mapping_node = world_nodes.new(type="ShaderNodeMapping")
    mapping_node.location = (-550, 0)
    mapping_node.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(hdri_rotation_deg))

    tex_coord_node = world_nodes.new(type="ShaderNodeTexCoord")
    tex_coord_node.location = (-750, 0)

    output_node = world_nodes.new(type="ShaderNodeOutputWorld")

    if isinstance(env_tex, bpy.types.ShaderNodeTexEnvironment) and env_tex.image is not None:
        world_links.new(tex_coord_node.outputs["Generated"], mapping_node.inputs["Vector"])
        world_links.new(mapping_node.outputs["Vector"], env_tex.inputs["Vector"])
        bg_node = world_nodes.new(type="ShaderNodeBackground")
        bg_node.location = (0, 0)
        bg_node.inputs["Strength"].default_value = hdri_strength
        world_links.new(env_tex.outputs["Color"], bg_node.inputs["Color"])
        world_links.new(bg_node.outputs["Background"], output_node.inputs["Surface"])
    else:
        bg_node = world_nodes.new(type="ShaderNodeBackground")
        bg_node.inputs["Color"].default_value = (0.30, 0.35, 0.40, 1.0)
        bg_node.inputs["Strength"].default_value = hdri_strength
        world_links.new(bg_node.outputs["Background"], output_node.inputs["Surface"])

    bpy.ops.object.camera_add(location=CAMERA_LOCATION)
    camera = bpy.context.object
    camera.data.lens = 50
    scene.camera = camera

    target = bpy.data.objects.new("camera_target", None)
    scene.collection.objects.link(target)
    target.location = CAMERA_TARGET
    constraint = camera.constraints.new("TRACK_TO")
    constraint.target = target
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"

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
        (0.2, 0.0, 0.35),
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

    create_floor(scenario)
    add_room_shell(scenario)

    physics = scenario["physics"]
    assert isinstance(physics, dict)
    ball_initial_location = tuple(float(v) for v in physics["ball_initial_location"])
    ball = add_ball(
        radius=ball_radius,
        location=ball_initial_location,
        scenario=scenario,
    )

    frame_end = max(2, int(round(float(args.duration_sec) * int(args.fps))))
    scene.frame_start = 1
    scene.frame_end = frame_end

    return ball, camera


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

    ball, camera = build_scene(args, scenario)
    physics = run_physics_simulation(args, scenario)
    apply_physics_animation(ball, physics)
    export_ground_truth(
        out_dir,
        ball,
        camera,
        bpy.context.scene.frame_end,
        int(args.fps),
        physics,
        scenario,
    )

    if args.mode == "preview":
        render_preview(args)
    elif args.mode == "frames":
        render_frames(args)
    else:
        render_animation(args)

    bpy.ops.wm.save_as_mainfile(filepath=str(output_path(args.out_dir, BLEND_NAME)))
    print(f"[INFO] Render complete. Output: {out_dir}")


if __name__ == "__main__":
    main()
