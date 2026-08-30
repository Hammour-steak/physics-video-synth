from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import bpy
from mathutils import Vector


WORKSPACE_DIR = Path(__file__).resolve().parents[1]
POLYHAVEN_DIR = WORKSPACE_DIR / "assets" / "polyhaven"
AMBIENTCG_DIR = WORKSPACE_DIR / "assets" / "ambientcg"

OUTPUT_STEM = "ramp_collision"
DIRECT_MP4_NAME = f"{OUTPUT_STEM}_blender_direct.mp4"
BLEND_NAME = f"{OUTPUT_STEM}.blend"

TABLE_SIZE = 0.6
TABLE_HEIGHT = 0.02
BOOK_LENGTH = 0.22
BOOK_WIDTH = 0.16
BOOK_THICKNESS = 0.025
BOOK_ANGLE_DEG = 12.0
MARBLE_RADIUS = 0.012

CAMERA_LOCATION = (0.15, 0.55, 0.15)
CAMERA_TARGET = (0.0, 0.0, 0.05)


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


TABLE_TEXTURE = PBRTextureSet(
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


def create_wood_table_material(scenario: dict[str, object]) -> bpy.types.Material:
    mat = bpy.data.materials.new("wood_table_surface")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    
    if bsdf is not None:
        set_input_default(bsdf, "Roughness", 0.4)
        set_input_default(bsdf, "Metallic", 0.0)
        set_input_default(bsdf, "Base Color", (0.45, 0.32, 0.20, 1.0))
        
        tex_coord = nodes.new(type="ShaderNodeTexCoord")
        mapping = nodes.new(type="ShaderNodeMapping")
        mapping.inputs["Scale"].default_value = (2.0, 2.0, 2.0)
        links.new(tex_coord.outputs["UV"], mapping.inputs["Vector"])
        
        try:
            diff_path = require_polyhaven_path(TABLE_TEXTURE.asset_name, TABLE_TEXTURE.diffuse_name)
            diff_img = load_image(diff_path, "sRGB")
            diff_img.colorspace_settings.name = 'sRGB'
            diff_tex = nodes.new(type="ShaderNodeTexImage")
            diff_tex.image = diff_img
            links.new(mapping.outputs["Vector"], diff_tex.inputs["Vector"])
            links.new(diff_tex.outputs["Color"], bsdf.inputs["Base Color"])
            
            rough_path = require_polyhaven_path(TABLE_TEXTURE.asset_name, TABLE_TEXTURE.roughness_name)
            rough_img = load_image(rough_path, "Non-Color")
            rough_img.colorspace_settings.name = 'Non-Color'
            rough_tex = nodes.new(type="ShaderNodeTexImage")
            rough_tex.image = rough_img
            links.new(mapping.outputs["Vector"], rough_tex.inputs["Vector"])
            links.new(rough_tex.outputs["Color"], bsdf.inputs["Roughness"])
            
            nor_path = require_polyhaven_path(TABLE_TEXTURE.asset_name, TABLE_TEXTURE.normal_name)
            nor_img = load_image(nor_path, "Non-Color")
            nor_img.colorspace_settings.name = 'Non-Color'
            nor_tex = nodes.new(type="ShaderNodeTexImage")
            nor_tex.image = nor_img
            nor_map = nodes.new(type="ShaderNodeNormalMap")
            nor_map.inputs["Strength"].default_value = TABLE_TEXTURE.normal_strength
            links.new(mapping.outputs["Vector"], nor_tex.inputs["Vector"])
            links.new(nor_tex.outputs["Color"], nor_map.inputs["Color"])
            links.new(nor_map.outputs["Normal"], bsdf.inputs["Normal"])
            
            print(f"[INFO] Using wood table texture: {TABLE_TEXTURE.asset_name}")
        except FileNotFoundError as e:
            print(f"[WARN] Wood table texture not found: {e}, using procedural fallback")
            set_input_default(bsdf, "Base Color", (0.45, 0.32, 0.20, 1.0))
    
    return mat


def create_steel_ball_material(scenario: dict[str, object]) -> bpy.types.Material:
    
    mat = bpy.data.materials.new("steel_ball")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    nodes.clear()
    
    output = nodes.new(type="ShaderNodeOutputMaterial")
    output.location = (800, 0)
    
    # Principled BSDF (reflective metal)
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (400, 0)
    bsdf.inputs["Metallic"].default_value = 0.95
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    
    # Texture coordinate & mapping
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
        # --- Color with subtle variation ---
        color_tex = load_tex(metal_dir, "Metal032_4K-JPG_Color.jpg", colorspace="sRGB")
        color_tex.location = (-550, 300)
        
        # Desaturate slightly for more realistic steel
        hue_sat = nodes.new(type="ShaderNodeHueSaturation")
        hue_sat.location = (-350, 300)
        hue_sat.inputs["Saturation"].default_value = 0.6
        links.new(color_tex.outputs["Color"], hue_sat.inputs["Color"])
        links.new(hue_sat.outputs["Color"], bsdf.inputs["Base Color"])
        
        # --- Combined roughness: Metal032 base + Rubber002 micro-detail ---
        rough_tex = load_tex(metal_dir, "Metal032_4K-JPG_Roughness.jpg")
        rough_tex.location = (-550, -100)
        
        # Rubber002 for extra micro-surface variation
        rubber_rough_tex = load_tex(rubber_dir, "Rubber002_4K-JPG_Roughness.jpg")
        rubber_rough_tex.location = (-550, -250)
        
        # Mix the two roughness sources
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
        
        # --- Normal map ---
        normal_tex = load_tex(metal_dir, "Metal032_4K-JPG_NormalGL.jpg")
        normal_tex.location = (-550, -400)
        
        normal_map = nodes.new(type="ShaderNodeNormalMap")
        normal_map.location = (-300, -400)
        normal_map.inputs["Strength"].default_value = 0.2
        links.new(normal_tex.outputs["Color"], normal_map.inputs["Color"])
        links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])
        
        # --- Displacement via Bump ---
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


def create_book(length: float, width: float, thickness: float, angle_deg: float, scenario: dict[str, object]) -> tuple[bpy.types.Object, list[bpy.types.Object]]:
    angle_rad = math.radians(angle_deg)
    
    cover_board_thickness = 0.002
    page_thickness = thickness - 2 * cover_board_thickness - 0.0005
    page_length = length
    page_width = width - 0.007
    
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
    cover = bpy.context.object
    cover.name = "book_cover_top"
    cover.dimensions = (length, width, cover_board_thickness)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    
    cover_mat = bpy.data.materials.new("textured_book_cover")
    cover_mat.use_nodes = True
    cover_nodes = cover_mat.node_tree.nodes
    cover_links = cover_mat.node_tree.links
    cover_bsdf = cover_nodes.get("Principled BSDF")
    
    if cover_bsdf is not None:
        set_input_default(cover_bsdf, "Roughness", 0.5)
        set_input_default(cover_bsdf, "Metallic", 0.0)
        set_input_default(cover_bsdf, "Sheen", 0.1)
        set_input_default(cover_bsdf, "Sheen Tint", 0.25)
        
        tex_coord = cover_nodes.new(type="ShaderNodeTexCoord")
        mapping = cover_nodes.new(type="ShaderNodeMapping")
        mapping.inputs["Scale"].default_value = (1.0, 1.0, 1.0)
        cover_links.new(tex_coord.outputs["UV"], mapping.inputs["Vector"])
        
        try:
            # 统一使用指定封面：无文字编织棉布封面
            cover_name = "book_cover_plain"
            cover_file = "book_pattern_col1_4k.jpg"
            cover_tex_path = require_polyhaven_path(cover_name, cover_file)
            cover_image = load_image(cover_tex_path, "sRGB")
            cover_tex = cover_nodes.new(type="ShaderNodeTexImage")
            cover_tex.image = cover_image
            cover_links.new(mapping.outputs["Vector"], cover_tex.inputs["Vector"])
            cover_links.new(cover_tex.outputs["Color"], cover_bsdf.inputs["Base Color"])
            print(f"[INFO] Using book cover texture: {cover_name} ({cover_file})")
        except FileNotFoundError:
            print("[WARN] Book cover texture not found, using procedural fallback")
            set_input_default(cover_bsdf, "Base Color", (0.12, 0.08, 0.06, 1.0))
    
    cover.data.materials.append(cover_mat)
    cover.data.use_auto_smooth = True
    cover_bevel = cover.modifiers.new("bevel", "BEVEL")
    cover_bevel.width = 0.0003
    cover_bevel.segments = 2
    
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
    bottom = bpy.context.object
    bottom.name = "book_cover_bottom"
    bottom.dimensions = (length, width, cover_board_thickness)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bottom.data.materials.append(cover_mat)
    bottom.data.use_auto_smooth = True
    bottom_bevel = bottom.modifiers.new("bevel", "BEVEL")
    bottom_bevel.width = 0.0003
    bottom_bevel.segments = 2
    
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
    spine = bpy.context.object
    spine.name = "book_spine"
    spine.dimensions = (length, 0.0012, thickness - 0.0006)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    spine.data.materials.append(cover_mat)
    spine.data.use_auto_smooth = True
    
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
    pages = bpy.context.object
    pages.name = "book_pages"
    pages.dimensions = (page_length, page_width, page_thickness)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    
    page_mat = bpy.data.materials.new("cream_book_pages")
    page_mat.use_nodes = True
    page_nodes = page_mat.node_tree.nodes
    page_links = page_mat.node_tree.links
    page_bsdf = page_nodes.get("Principled BSDF")
    
    if page_bsdf is not None:
        set_input_default(page_bsdf, "Base Color", (0.95, 0.90, 0.80, 1.0))
        set_input_default(page_bsdf, "Roughness", 0.88)
        set_input_default(page_bsdf, "Metallic", 0.0)
        set_input_default(page_bsdf, "Sheen", 0.03)
    
    pages.data.materials.append(page_mat)
    pages.data.use_auto_smooth = True
    
    spine_center_y = -width / 2 + 0.0006
    cover_center_z = page_thickness / 2 + cover_board_thickness / 2 + 0.0003
    
    cover.location = (0.0, 0.0, cover_center_z)
    bottom.location = (0.0, 0.0, -cover_center_z)
    spine.location = (0.0, spine_center_y, 0.0)
    pages.location = (0.0015, 0.0, 0.0)
    
    parent = bpy.data.objects.new("book_parent", None)
    bpy.context.collection.objects.link(parent)
    parent.rotation_euler = (0.0, angle_rad, 0.0)
    for obj in [cover, bottom, spine, pages]:
        obj.parent = parent
    
    # Lowest point of the book in parent-local space: open end (+X), bottom of bottom cover
    book_lowest_local_z = -cover_center_z - cover_board_thickness / 2
    book_lowest_x = length / 2
    book_lowest_world_z = -book_lowest_x * math.sin(angle_rad) + book_lowest_local_z * math.cos(angle_rad)
    parent_z = TABLE_HEIGHT - book_lowest_world_z
    parent.location = (0.0, 0.0, parent_z)
    
    return parent, [cover, bottom, spine, pages], cover_center_z, parent_z


def create_glass_marble(radius: float, location: tuple[float, float, float], name: str, scenario: dict[str, object]) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=location)
    marble = bpy.context.object
    marble.name = name
    marble_mat = create_steel_ball_material(scenario)
    marble.data.materials.append(marble_mat)
    return marble


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    
    scenario = {
        "seed": args.seed,
    }
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # HDRI environment texture for home study background
    env_tex = world_nodes.new(type="ShaderNodeTexEnvironment")
    env_tex.location = (-300, 0)
    
    hdri_path = POLYHAVEN_DIR / "wooden_lounge" / "wooden_lounge_2k.hdr"
    if hdri_path.exists():
        hdri_img = bpy.data.images.load(str(hdri_path), check_existing=True)
        env_tex.image = hdri_img
        print(f"[INFO] Using HDRI background: wooden_lounge")
    else:
        print(f"[WARN] HDRI not found at {hdri_path}, using solid fallback")
        # Fallback: solid color on the background node
        bg_node = world_nodes.new(type="ShaderNodeBackground")
        bg_node.inputs["Color"].default_value = (0.30, 0.35, 0.40, 1.0)
        bg_node.inputs["Strength"].default_value = 1.0
        env_tex = bg_node
    
    mapping_node = world_nodes.new(type="ShaderNodeMapping")
    mapping_node.location = (-550, 0)
    mapping_node.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(95))
    
    tex_coord_node = world_nodes.new(type="ShaderNodeTexCoord")
    tex_coord_node.location = (-750, 0)
    
    output_node = world_nodes.new(type="ShaderNodeOutputWorld")
    
    if isinstance(env_tex, bpy.types.ShaderNodeTexEnvironment):
        world_links.new(tex_coord_node.outputs["Generated"], mapping_node.inputs["Vector"])
        world_links.new(mapping_node.outputs["Vector"], env_tex.inputs["Vector"])
        bg_node = world_nodes.new(type="ShaderNodeBackground")
        bg_node.location = (0, 0)
        bg_node.inputs["Strength"].default_value = 0.8
        world_links.new(env_tex.outputs["Color"], bg_node.inputs["Color"])
        world_links.new(bg_node.outputs["Background"], output_node.inputs["Surface"])
    else:
        world_links.new(env_tex.outputs["Background"], output_node.inputs["Surface"])
    
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
    
    bpy.ops.object.light_add(type="SUN", location=(2.0, -2.0, 3.0))
    sun = bpy.context.object
    sun.data.energy = 0.3
    sun.rotation_euler = (math.radians(45), math.radians(15), math.radians(30))
    
    bpy.ops.object.light_add(type="AREA", location=(-1.5, 1.0, 2.0))
    fill_light = bpy.context.object
    fill_light.data.energy = 50
    fill_light.data.size = 2.0
    
    bpy.ops.object.light_add(type="AREA", location=(0.5, -0.5, 1.5))
    rim_light = bpy.context.object
    rim_light.data.energy = 30
    rim_light.data.size = 1.5
    
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, TABLE_HEIGHT / 2))
    table_surface = bpy.context.object
    table_surface.name = "table_surface"
    table_surface.dimensions = (TABLE_SIZE, TABLE_SIZE, TABLE_HEIGHT)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    
    table_mat = create_wood_table_material(scenario)
    table_surface.data.materials.append(table_mat)
    
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, -TABLE_HEIGHT / 2))
    table_base = bpy.context.object
    table_base.name = "table_base"
    table_base.dimensions = (TABLE_SIZE, TABLE_SIZE, TABLE_HEIGHT)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    
    base_mat = bpy.data.materials.new("table_base_wood")
    base_mat.use_nodes = True
    base_bsdf = base_mat.node_tree.nodes.get("Principled BSDF")
    if base_bsdf is not None:
        set_input_default(base_bsdf, "Base Color", (0.35, 0.25, 0.15, 1.0))
        set_input_default(base_bsdf, "Roughness", 0.6)
    table_base.data.materials.append(base_mat)
    
    book_result = create_book(
        length=BOOK_LENGTH,
        width=BOOK_WIDTH,
        thickness=BOOK_THICKNESS,
        angle_deg=BOOK_ANGLE_DEG,
        scenario=scenario,
    )
    if isinstance(book_result, tuple) and len(book_result) == 4:
        book, book_parts, cover_center_z, parent_z = book_result
    else:
        book, book_parts, cover_center_z, parent_z = book_result[0], book_result[1], book_result[2], book_result[3]
    
    angle_rad = math.radians(BOOK_ANGLE_DEG)
    
    cover_board_thickness = 0.002
    page_thickness = BOOK_THICKNESS - 2 * cover_board_thickness - 0.0005
    cover_center_z = page_thickness / 2 + cover_board_thickness / 2 + 0.0003
    
    # --- World-space collision plane for the top book surface ---
    # Plane center in parent-local frame (center of top surface)
    coll_plane_lx = 0.0
    coll_plane_lz = cover_center_z + cover_board_thickness / 2
    coll_plane_origin_x = coll_plane_lx * math.cos(angle_rad) + coll_plane_lz * math.sin(angle_rad)
    coll_plane_origin_z = parent_z - coll_plane_lx * math.sin(angle_rad) + coll_plane_lz * math.cos(angle_rad)
    
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(coll_plane_origin_x, 0.0, coll_plane_origin_z))
    coll_plane = bpy.context.object
    coll_plane.name = "book_collision_plane"
    coll_plane.rotation_euler = (0.0, angle_rad, 0.0)
    coll_plane.scale = (BOOK_LENGTH / 2 - 0.015, BOOK_WIDTH / 2, 1.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    coll_plane.hide_render = True
    
    # --- Support pillar under the spine end of the tilted book ---
    # Book bottom is a sloped plane; use a narrow pillar (small X extent)
    # so the slope difference within its footprint is negligible.
    pillar_size = BOOK_WIDTH / 3
    pillar_half = pillar_size / 2
    
    # Compute book bottom Z at the pillar's center, not at the spine corner.
    # Pillar center world X is slightly right of the spine corner so the pillar
    # sits mostly under the book.
    support_local_x = -BOOK_LENGTH / 2
    support_local_z = -(cover_center_z + cover_board_thickness / 2)
    pillar_center_wx = support_local_x * math.cos(angle_rad) + support_local_z * math.sin(angle_rad) + pillar_half
    # Find local x at pillar center
    pillar_center_lx = (pillar_center_wx - support_local_z * math.sin(angle_rad)) / math.cos(angle_rad)
    pillar_center_wz = parent_z - pillar_center_lx * math.sin(angle_rad) + support_local_z * math.cos(angle_rad)
    support_height = pillar_center_wz - TABLE_HEIGHT
    
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(pillar_center_wx, 0.0, TABLE_HEIGHT + support_height / 2))
    support_block = bpy.context.object
    support_block.name = "book_support"
    support_block.dimensions = (pillar_size, pillar_size, support_height)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    
    support_mat = bpy.data.materials.new("support_block")
    support_mat.use_nodes = True
    support_bsdf = support_mat.node_tree.nodes.get("Principled BSDF")
    if support_bsdf is not None:
        set_input_default(support_bsdf, "Base Color", (0.25, 0.20, 0.15, 1.0))
        set_input_default(support_bsdf, "Roughness", 0.55)
    support_block.data.materials.append(support_mat)
    support_block.data.use_auto_smooth = True
    support_bevel = support_block.modifiers.new("bevel", "BEVEL")
    support_bevel.width = 0.0005
    support_bevel.segments = 2
    
    # Falling marble at the HIGH end (spine side) of the book, resting on top
    
    # Position in parent's local frame (parent is rotated)
    fm_local_x = -BOOK_LENGTH / 2 + MARBLE_RADIUS + 0.002
    fm_local_z = cover_center_z + cover_board_thickness / 2 + MARBLE_RADIUS + 0.0002
    falling_marble_x = fm_local_x * math.cos(angle_rad) + fm_local_z * math.sin(angle_rad)
    falling_marble_z = parent_z - fm_local_x * math.sin(angle_rad) + fm_local_z * math.cos(angle_rad)
    
    falling_marble = create_glass_marble(
        radius=MARBLE_RADIUS,
        location=(falling_marble_x, 0.0, falling_marble_z),
        name="falling_marble",
        scenario=scenario,
    )
    
    # Stationary marbles at the LOW end (open side) of the book, on the table surface
    # Ball exits at low end top corner, falls nearly straight down
    low_top_local_x = BOOK_LENGTH / 2
    low_top_local_z = cover_center_z + cover_board_thickness / 2
    low_end_rightmost_x = low_top_local_x * math.cos(angle_rad) + low_top_local_z * math.sin(angle_rad)
    # Ball drops from here and flies forward; place marbles further out
    marble_base_x = low_end_rightmost_x + MARBLE_RADIUS + 0.04
    stationary_base_z = TABLE_HEIGHT + MARBLE_RADIUS
    
    stationary_marbles = []
    # 2 个静止小球：一个在 y=0 下落轨迹上，另一个沿 y 方向远离轨迹
    stationary_positions = [
        (marble_base_x, 0.0, stationary_base_z),
        (marble_base_x + 0.03, -0.15, stationary_base_z),
    ]
    
    for i, pos in enumerate(stationary_positions):
        marble = create_glass_marble(
            radius=MARBLE_RADIUS,
            location=pos,
            name=f"stationary_marble_{i}",
            scenario=scenario,
        )
        stationary_marbles.append(marble)
    
    if scene.rigidbody_world is None:
        bpy.ops.rigidbody.world_add()
    
    rw = scene.rigidbody_world
    rw.solver_iterations = 100
    rw.substeps_per_frame = 10
    scene.gravity = (0.0, 0.0, -1.0)
    
    def add_rigidbody(obj, body_type):
        if obj.rigid_body is None:
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.rigidbody.object_add(type=body_type)
        obj.rigid_body.type = body_type
        return obj.rigid_body
    
    table_surface_rb = add_rigidbody(table_surface, 'PASSIVE')
    table_surface_rb.friction = 0.4
    table_surface_rb.restitution = 0.2
    table_surface_rb.collision_shape = 'BOX'
    table_surface_rb.collision_margin = 0.0
    
    table_base_rb = add_rigidbody(table_base, 'PASSIVE')
    table_base_rb.friction = 0.4
    table_base_rb.restitution = 0.2
    table_base_rb.collision_shape = 'BOX'
    table_base_rb.collision_margin = 0.0
    
    # Book parts are too thin for reliable collision; proxy handles all top surface collision
    # for part in book_parts:
    #     part_rb = add_rigidbody(part, 'PASSIVE')
    #     part_rb.friction = 0.3
    #     part_rb.restitution = 0.1
    #     part_rb.collision_shape = 'BOX'
    #     part_rb.collision_margin = 0.0
    
    support_block_rb = add_rigidbody(support_block, 'PASSIVE')
    support_block_rb.friction = 0.5
    support_block_rb.restitution = 0.15
    support_block_rb.collision_shape = 'BOX'
    support_block_rb.collision_margin = 0.0
    
    coll_plane_rb = add_rigidbody(coll_plane, 'PASSIVE')
    coll_plane_rb.friction = 0.7
    coll_plane_rb.restitution = 0.1
    coll_plane_rb.collision_shape = 'MESH'
    coll_plane_rb.collision_margin = 0.0
    
    falling_marble_rb = add_rigidbody(falling_marble, 'ACTIVE')
    falling_marble_rb.mass = 0.05
    falling_marble_rb.friction = 0.45
    falling_marble_rb.restitution = 0.6
    falling_marble_rb.linear_damping = 0.05
    falling_marble_rb.angular_damping = 0.05
    falling_marble_rb.collision_shape = 'SPHERE'
    falling_marble_rb.collision_margin = 0.0
    falling_marble_rb.kinematic = True
    
    for marble in stationary_marbles:
        marble_rb = add_rigidbody(marble, 'ACTIVE')
        marble_rb.mass = 0.05
        marble_rb.friction = 0.4
        marble_rb.restitution = 0.3
        marble_rb.linear_damping = 0.1
        marble_rb.angular_damping = 0.1
        marble_rb.collision_shape = 'SPHERE'
        marble_rb.collision_margin = 0.0
        marble_rb.kinematic = False
    
    frame_end = int(args.fps * args.duration_sec)
    scene.frame_start = 1
    scene.frame_end = frame_end
    
    scene.frame_set(1)
    falling_marble.rigid_body.kinematic = True
    falling_marble.keyframe_insert(data_path="location", frame=1)
    falling_marble.keyframe_insert(data_path="rotation_euler", frame=1)
    falling_marble.rigid_body.keyframe_insert(data_path="kinematic", frame=1)
    
    scene.frame_set(2)
    falling_marble.rigid_body.kinematic = False
    falling_marble.rigid_body.keyframe_insert(data_path="kinematic", frame=2)
    
    # Bake rigid body simulation before rendering
    rw.point_cache.frame_start = scene.frame_start
    rw.point_cache.frame_end = scene.frame_end
    override = {'point_cache': rw.point_cache}
    bpy.ops.ptcache.free_bake(override)
    bpy.ops.ptcache.bake(override, bake=True)
    print(f"[INFO] Rigid body baked: frame {scene.frame_start}-{scene.frame_end}")
    
    print("[DIAGNOSTIC] Falling ball X positions:")
    for frame in range(1, min(frame_end + 1, 50)):
        scene.frame_set(frame)
        pos = falling_marble.location
        print(f"  Frame {frame}: x={pos.x:.6f}, z={pos.z:.6f}")
    
    print("[DIAGNOSTIC] Stationary marble positions (all frames):")
    for i, marble in enumerate(stationary_marbles):
        prev_pos = None
        for frame in range(1, min(frame_end + 1, 50)):
            scene.frame_set(frame)
            pos = marble.location
            if prev_pos is None:
                prev_pos = pos
            else:
                dx = abs(pos.x - prev_pos.x)
                dy = abs(pos.y - prev_pos.y)
                dz = abs(pos.z - prev_pos.z)
                if dx > 0.000001 or dy > 0.000001 or dz > 0.000001:
                    print(f"  Marble {i} moved at frame {frame}: ({pos.x:.6f}, {pos.y:.6f}, {pos.z:.6f})")
                    prev_pos = pos
    
    if args.mode == "preview":
        scene.frame_set(args.preview_frame)
        scene.render.image_settings.file_format = 'PNG'
        scene.render.filepath = str(out_dir / "preview.png")
        bpy.ops.render.render(write_still=True)
    elif args.mode == "frames":
        scene.render.image_settings.file_format = 'PNG'
        scene.frame_end = min(20, frame_end)
        scene.render.filepath = str(out_dir / "frame_")
        bpy.ops.render.render(animation=True)
    else:
        scene.render.filepath = str(out_dir / DIRECT_MP4_NAME)
        bpy.ops.render.render(animation=True)
    
    bpy.ops.wm.save_as_mainfile(filepath=str(out_dir / BLEND_NAME))
    
    print(f"[INFO] Render complete. Output: {out_dir}")


if __name__ == "__main__":
    main()