"""Blender/Cycles renderer for the twin-ramp head-on collision.

Builds the apparatus described in twin_ramp_geometry -- a base plank with a
solid wooden wedge at each end -- puts a glass marble at each crest, runs
simulate_twin_ramp_collision.py, replays the result as keyframes, and renders.

Run through Blender, not python:

    blender -b --python render_twin_ramp_collision.py -- --out-dir ...
"""

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

import bpy
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from twin_ramp_geometry import build_track  # noqa: E402


WORKSPACE_DIR = Path(__file__).resolve().parents[2]
POLYHAVEN_DIR = WORKSPACE_DIR / "assets" / "polyhaven"
AMBIENTCG_DIR = WORKSPACE_DIR / "assets" / "ambientcg"
MODELS_DIR = WORKSPACE_DIR / "assets" / "models"
MARBLE_GLB = MODELS_DIR / "marble_yellow_ball.glb"

OUTPUT_STEM = "twin_ramp_collision"
BLEND_NAME = f"{OUTPUT_STEM}.blend"
PHYSICS_TEMP_NAME = "physics_transforms.json"
GROUND_TRUTH_NAME = "ground_truth_transforms.json"
SCENARIO_METADATA_NAME = "scenario_metadata.json"

# Framing: square to the track, lifted just enough to see the valley floor --
# the "R2" candidate from survey_cameras.py, chosen from eight. Shooting the
# apparatus straight on is what makes the two ramps read as a matched pair and
# puts the meeting point dead centre, which is the whole subject of the shot.
#
# The standoff is not a taste decision: the rig is 1.29 m crest to crest, and a
# 48 mm lens on a 36 mm sensor only covers that from about 1.8 m. A longer lens
# further back is preferred over a wide one up close because perspective is the
# enemy here -- on a wide lens the near ramp grows and the two stop matching.
CAMERA_LOCATION = (0.0, -2.05, 0.40)
CAMERA_TARGET = (0.0, 0.0, 0.045)
CAMERA_LENS_MM = 48.0


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preview", "animation", "frames"), default="animation")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resolution", nargs=2, type=int, default=(1280, 720))
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-sec", type=float, default=3.0)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--preview-frame", type=int, default=28)
    parser.add_argument(
        "--preview-frames", nargs="+", type=int, default=None,
        help="Render several stills instead of one, as preview_f<N>.png. Preview mode only.",
    )
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--scenario-json", type=Path, default=None,
        help="Optional complete scenario_metadata.json to render instead of building one.",
    )
    parser.add_argument(
        "--scenario-overrides-json", type=Path, default=None,
        help="Optional JSON object recursively merged onto the built scenario.",
    )
    return parser.parse_args(argv)


# --- Scenario -----------------------------------------------------------------

def deep_merge(base: dict, overrides: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def build_scenario(args: argparse.Namespace) -> dict[str, Any]:
    if args.scenario_json is not None:
        scenario = json.loads(args.scenario_json.read_text(encoding="utf-8"))
    else:
        scenario = {
            "schema_version": 1,
            "seed": int(args.seed),
            "scene": OUTPUT_STEM,
            "render": {
                "mode": str(args.mode),
                "fps": int(args.fps),
                "duration_sec": float(args.duration_sec),
                "resolution": [int(args.resolution[0]), int(args.resolution[1])],
                "samples": int(args.samples),
                "device": str(args.device),
                # A real 24 fps camera has a shutter open for about half a frame,
                # and this scene needs that: the balls cross a quarter of their
                # own diameter between samples on the approach and the impact
                # itself is sub-frame. Without blur the collision reads as two
                # balls teleporting past each other.
                "motion_blur": True,
                "motion_blur_shutter": 0.5,
            },
            # Hue is a rotation applied to the model's own swirl texture, not a
            # flat colour: 0.5 leaves it as authored, and the offset from 0.5 is
            # the rotation. Kept in the scenario so a case can recolour the balls
            # without touching code.
            "balls": {
                "a": {"hue": 0.12, "saturation": 1.05},
                "b": {"hue": 0.50, "saturation": 1.15},
            },
            "camera": {
                "location": list(CAMERA_LOCATION),
                "target": list(CAMERA_TARGET),
                "lens_mm": CAMERA_LENS_MM,
            },
            # Geometry and material constants are shared verbatim with
            # simulate_twin_ramp_collision.py's defaults; see its --ramp-angle-deg
            # note for why the ramp's rise is what it is.
            "physics": {
                "ramp_angle_deg": 8.0,
                "ramp_run": 0.554,
                "ramp_width": 0.24,
                "ramp_body_thickness": 0.10,
                "valley_half": 0.09,
                "plank_thickness": 0.018,
                "plank_length": 1.42,
                "plank_width": 0.28,
                # A 64 mm glass marble: 2500 kg/m^3 over 1.373e-4 m^3.
                "ball_radius": 0.032,
                "ball_mass": 0.343,
                "ball_friction": 0.30,
                "ball_restitution": 0.87,
                "ball_rolling_friction": 0.0012,
                "ball_spinning_friction": 0.004,
                # Per-ball fields, explicit baseline copies of the globals
                # above so a PCVE mass/friction/restitution edit has a
                # well-defined `from` value to diff against. ball_a is the
                # +X-side marble (hue-shifted from the yellow base GLB to
                # purple); ball_b is the -X-side one (hue 0.50 is Blender's
                # no-op so it keeps the base yellow).
                "ball_a_mass": 0.343,
                "ball_a_friction": 0.30,
                "ball_a_restitution": 0.87,
                "ball_a_rolling_friction": 0.0012,
                "ball_b_mass": 0.343,
                "ball_b_friction": 0.30,
                "ball_b_restitution": 0.87,
                "ball_b_rolling_friction": 0.0012,
                # Two-slot presence list (ball_a on +X, ball_b on -X). A
                # PCVE DELETE edit writes 0 at the ball's slot; the sim
                # skips creating that body and its frame slot is filled
                # with present=false.
                "active": [1, 1],
                "track_friction": 0.55,
                "track_restitution": 0.14,
                "release_inset": 0.030,
                "release_inset_bias": 0.004,
                "hold_sec": 0.25,
                "substeps": 16,
                "gravity_z": -9.8,
            },
        }
    if args.scenario_overrides_json is not None:
        scenario = deep_merge(
            scenario, json.loads(args.scenario_overrides_json.read_text(encoding="utf-8")),
        )
    return scenario


# --- Blender helpers ----------------------------------------------------------

def set_input(node, name, value) -> None:
    socket = node.inputs.get(name)
    if socket is not None:
        socket.default_value = value


def look_at(obj: bpy.types.Object, target) -> None:
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def enable_gpu() -> None:
    prefs = bpy.context.preferences.addons.get("cycles")
    if prefs is None:
        return
    cprefs = prefs.preferences
    for backend in ("OPTIX", "CUDA"):
        try:
            cprefs.compute_device_type = backend
        except TypeError:
            continue
        cprefs.get_devices()
        devices = [d for d in cprefs.devices if d.type == backend]
        if devices:
            for device in cprefs.devices:
                device.use = device.type == backend
            print(f"[INFO] Cycles device backend: {backend} ({len(devices)} device(s))")
            return
    print("[WARN] No OPTIX/CUDA device found; Cycles will fall back to CPU.")


def load_texture(nodes, links, vector_socket, path: Path, colorspace: str, projection: str):
    if not path.exists():
        raise FileNotFoundError(path)
    image = bpy.data.images.load(str(path), check_existing=True)
    image.colorspace_settings.name = colorspace
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = image
    tex.projection = projection
    if projection == "BOX":
        tex.projection_blend = 0.25
    links.new(vector_socket, tex.inputs["Vector"])
    return tex


def mesh_bounds(objects) -> tuple[Vector, Vector] | None:
    lo = Vector((1e9, 1e9, 1e9))
    hi = Vector((-1e9, -1e9, -1e9))
    found = False
    for obj in objects:
        if obj.type != "MESH" or not obj.data.vertices:
            continue
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            lo = Vector((min(lo.x, world.x), min(lo.y, world.y), min(lo.z, world.z)))
            hi = Vector((max(hi.x, world.x), max(hi.y, world.y), max(hi.z, world.z)))
            found = True
    return (lo, hi) if found else None


def multiply_tint(nodes, links, source_socket, tint: tuple[float, float, float]):
    """Multiply a texture by a flat colour, returning the result socket.

    Deliberately the legacy ShaderNodeMixRGB rather than 3.4's ShaderNodeMix.
    The newer node carries a separate set of A/B sockets per data type, so its
    RGBA inputs are not at the indices a by-index link lands on; wiring it that
    way silently leaves Base Color unconnected, and every surface renders as the
    flat tint with only its normal map showing. MixRGB's sockets are named.
    """
    mix = nodes.new("ShaderNodeMixRGB")
    mix.blend_type = "MULTIPLY"
    mix.inputs["Fac"].default_value = 1.0
    links.new(source_socket, mix.inputs["Color1"])
    mix.inputs["Color2"].default_value = (*tint, 1.0)
    return mix.outputs["Color"]


# --- Materials ----------------------------------------------------------------

def create_wood_material(
    name: str,
    *,
    scale: float,
    tint: tuple[float, float, float],
    rotation_z: float = 0.0,
    roughness_scale: float = 1.0,
) -> bpy.types.Material:
    """Wood for the plank and the wedges, box-projected off ambientCG Wood049.

    The wedges and the plank are cut from the same board in the fiction of the
    scene, so they share the texture; they are told apart by mapping scale and
    grain direction rather than by using two different scans, which is what
    actually happens when you build something out of one sheet of ply.

    Box projection is used because none of these meshes is UV-unwrapped -- they
    are generated here from raw vertices -- and box projection gives a boxy
    prop clean grain on every face without an unwrap step.
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    set_input(bsdf, "Base Color", (*tint, 1.0))
    set_input(bsdf, "Roughness", 0.72)
    set_input(bsdf, "Metallic", 0.0)
    set_input(bsdf, "Specular", 0.08)

    tex_coord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (scale, scale, scale)
    mapping.inputs["Rotation"].default_value = (0.0, 0.0, rotation_z)
    links.new(tex_coord.outputs["Object"], mapping.inputs["Vector"])
    vector = mapping.outputs["Vector"]

    wood_dir = AMBIENTCG_DIR / "Wood049"
    try:
        color = load_texture(
            nodes, links, vector, wood_dir / "Wood049_4K-JPG_Color.jpg", "sRGB", "BOX",
        )
        links.new(
            multiply_tint(nodes, links, color.outputs["Color"], tint),
            bsdf.inputs["Base Color"],
        )

        rough = load_texture(
            nodes, links, vector, wood_dir / "Wood049_4K-JPG_Roughness.jpg", "Non-Color", "BOX",
        )
        rough_adjust = nodes.new("ShaderNodeMath")
        rough_adjust.operation = "MULTIPLY"
        rough_adjust.inputs[1].default_value = float(roughness_scale)
        links.new(rough.outputs["Color"], rough_adjust.inputs[0])
        # Floored hard, and Specular dropped to 0.08 above. Wood049 ships glossy
        # enough to mirror the window: the left wedge's slope sits almost exactly
        # on the reflection angle from the window to the camera, and at the map's
        # own roughness it returned a broad specular smear that read as 1.7x the
        # slope's albedo, desaturated to grey. Bare sawn wood is not a varnished
        # tabletop, and at 0.72 the reflection becomes the faint sheen it should
        # be while the grain survives.
        rough_floor = nodes.new("ShaderNodeMath")
        rough_floor.operation = "MAXIMUM"
        rough_floor.inputs[1].default_value = 0.72
        links.new(rough_adjust.outputs["Value"], rough_floor.inputs[0])
        links.new(rough_floor.outputs["Value"], bsdf.inputs["Roughness"])

        normal = load_texture(
            nodes, links, vector, wood_dir / "Wood049_4K-JPG_NormalGL.jpg", "Non-Color", "BOX",
        )
        normal_map = nodes.new("ShaderNodeNormalMap")
        normal_map.inputs["Strength"].default_value = 0.45
        links.new(normal.outputs["Color"], normal_map.inputs["Color"])
        links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])
    except FileNotFoundError as exc:
        print(f"[WARN] Wood049 texture missing ({exc}); using a flat wood colour.")
    return mat


def create_polyhaven_material(
    name: str,
    asset: str,
    *,
    scale: float,
    base_color: tuple[float, float, float],
    normal_strength: float = 0.6,
) -> bpy.types.Material:
    """A UV-mapped Poly Haven PBR set, for the room floor and the table top.

    These two are real flat quads with real UVs, unlike the generated apparatus
    meshes, so they take an ordinary FLAT projection rather than the box
    projection create_wood_material needs.
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    set_input(bsdf, "Base Color", (*base_color, 1.0))
    set_input(bsdf, "Roughness", 0.62)
    set_input(bsdf, "Specular", 0.3)

    tex_coord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (scale, scale, scale)
    links.new(tex_coord.outputs["UV"], mapping.inputs["Vector"])
    vector = mapping.outputs["Vector"]

    asset_dir = POLYHAVEN_DIR / asset
    try:
        color = load_texture(
            nodes, links, vector, asset_dir / f"{asset}_diff_4k.jpg", "sRGB", "FLAT",
        )
        links.new(color.outputs["Color"], bsdf.inputs["Base Color"])
        rough = load_texture(
            nodes, links, vector, asset_dir / f"{asset}_rough_4k.jpg", "Non-Color", "FLAT",
        )
        links.new(rough.outputs["Color"], bsdf.inputs["Roughness"])
        normal = load_texture(
            nodes, links, vector, asset_dir / f"{asset}_nor_gl_4k.jpg", "Non-Color", "FLAT",
        )
        normal_map = nodes.new("ShaderNodeNormalMap")
        normal_map.inputs["Strength"].default_value = normal_strength
        links.new(normal.outputs["Color"], normal_map.inputs["Color"])
        links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])
    except FileNotFoundError as exc:
        print(f"[WARN] Poly Haven set {asset} missing ({exc}); using a flat colour.")
    return mat


def create_paint_material(name: str, color: tuple[float, float, float]) -> bpy.types.Material:
    """Matt emulsion for the room's walls."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    set_input(bsdf, "Base Color", (*color, 1.0))
    set_input(bsdf, "Roughness", 0.88)
    set_input(bsdf, "Specular", 0.16)
    return mat


# --- Apparatus ----------------------------------------------------------------

def build_wedge(track, side: int, material: bpy.types.Material) -> bpy.types.Object:
    """A solid triangular-prism ramp, built to match its collision box exactly.

    The mesh is written out by hand rather than modelled from a cube because its
    hypotenuse has to sit on the plane twin_ramp_geometry computed for the
    collider; a scaled-and-rotated cube would agree only to whatever the
    rotation happened to round to.
    """
    profile = track.wedge_profile(side)
    half_w = track.ramp_width / 2.0
    verts = [(x, -half_w, z) for x, z in profile] + [(x, half_w, z) for x, z in profile]
    faces = [
        (0, 1, 2),          # near cheek
        (5, 4, 3),          # far cheek
        (0, 3, 4, 1),       # underside, sitting on the plank
        (1, 4, 5, 2),       # vertical back
        (0, 2, 5, 3),       # the sloped face the ball runs on
    ]

    mesh = bpy.data.meshes.new(f"ramp_{'a' if side > 0 else 'b'}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.validate()
    obj = bpy.data.objects.new(f"ramp_{'a' if side > 0 else 'b'}", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(material)

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.select_set(False)

    # A sawn wedge has an eased toe, not a knife edge. 0.8 mm is 2.5% of the
    # ball's radius, far inside the contact tolerance, so the render keeps
    # agreeing with the collider it was built from.
    bevel = obj.modifiers.new("bevel", "BEVEL")
    bevel.width = 0.0008
    bevel.segments = 2
    bevel.limit_method = "ANGLE"
    return obj


def build_plank(track, material: bpy.types.Material) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, track.plank_thickness / 2.0))
    plank = bpy.context.object
    plank.name = "base_plank"
    plank.dimensions = (track.plank_length, track.plank_width, track.plank_thickness)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    plank.data.materials.append(material)
    bevel = plank.modifiers.new("bevel", "BEVEL")
    bevel.width = 0.0012
    bevel.segments = 2
    return plank


def import_marble(name: str, diameter: float, hue: float, saturation: float):
    """Import the marble model, size it, centre it, and recolour it.

    The model ships as one 0.712-unit sphere carrying a photographed swirl on an
    image texture. That texture is doing real work beyond looking like glass: a
    featureless sphere is rotationally invariant on screen, and the swirl is the
    only thing that shows these balls arrive *rolling* rather than sliding --
    which is the whole reason the rebound is as weak as it is.

    Colour is therefore changed with a hue rotation on the texture rather than by
    overwriting the base colour, which would throw the swirl away along with the
    yellow. Each ball gets its own copy of the material, otherwise both balls
    would share one datablock and recolouring the second would recolour the
    first.
    """
    if not MARBLE_GLB.exists():
        raise FileNotFoundError(f"Marble model not found: {MARBLE_GLB}")

    scene = bpy.context.scene
    before = set(scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(MARBLE_GLB))
    imported = [o for o in scene.objects if o not in before]
    meshes = [o for o in imported if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"{MARBLE_GLB.name} imported no mesh.")

    ball = meshes[0]
    ball.name = name
    # Free the mesh from the model's empty hierarchy, keeping where it ended up,
    # then throw the hierarchy away.
    matrix = ball.matrix_world.copy()
    ball.parent = None
    ball.matrix_world = matrix
    for obj in imported:
        if obj is not ball:
            bpy.data.objects.remove(obj, do_unlink=True)

    bpy.context.view_layer.objects.active = ball
    ball.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # Scale to the physical diameter, then put the origin at the sphere's centre.
    # The simulator hands over centre positions and orientations, so the object's
    # origin has to *be* the centre or the ball will orbit its own offset.
    lo, hi = mesh_bounds([ball])
    span = max(hi.z - lo.z, 1e-9)
    ball.scale = tuple(float(diameter) / span for _ in range(3))
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    ball.location = (0.0, 0.0, 0.0)
    ball.select_set(False)

    material = ball.data.materials[0].copy() if ball.data.materials else None
    if material is None:
        raise RuntimeError(f"{MARBLE_GLB.name} carries no material to recolour.")
    material.name = f"{name}_marble"
    ball.data.materials[0] = material
    nodes, links = material.node_tree.nodes, material.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    base = bsdf.inputs.get("Base Color")
    if base is not None and base.is_linked:
        source = base.links[0].from_socket
        hue_sat = nodes.new("ShaderNodeHueSaturation")
        hue_sat.inputs["Hue"].default_value = float(hue)
        hue_sat.inputs["Saturation"].default_value = float(saturation)
        links.new(source, hue_sat.inputs["Color"])
        links.new(hue_sat.outputs["Color"], base)
    else:
        print(f"[WARN] {name}: base colour is not textured; hue shift skipped.")

    for poly in ball.data.polygons:
        poly.use_smooth = True
    ball.rotation_mode = "QUATERNION"
    print(f"[INFO] Marble {name}: {span:.4f} model units -> {diameter:.3f} m, hue {hue}")
    return ball


# --- Room ---------------------------------------------------------------------
#
# The physics frame puts z = 0 on the surface the balls run on, and the room is
# built downwards from there: z = 0 is the *table top*, and the floor is a table
# height below it. That way the simulator never has to know a room exists.
#
# The room is here because without it the shot was unreadable. The apparatus
# used to stand on a single 10 m texture plane with a 2k HDRI behind it, and at
# these low camera heights that gives you an unbounded wood surface running to a
# smear -- no table edge, no floor, no wall, nothing to say whether you are
# looking at a table top or a floor, or what room you are in. The far table
# edge, the floor visible beyond it, and the wall behind are what answer that,
# so all three have to be inside the frame.
TABLE_TOP_Z = 0.0
TABLE_THICKNESS = 0.038
TABLE_SIZE = (1.92, 0.88)
TABLE_LEG = 0.072
FLOOR_Z = -0.745
WALL_Y = 1.45
WALL_TOP_Z = 2.45          # ceiling height above the floor: 2.45 - (-0.745)
ROOM_HALF_X = 2.35
ROOM_FRONT_Y = -2.95       # wall behind the camera
# Window in the back wall, as (centre x, sill z, width, height). It sits low
# enough to be in shot behind the table, which is what makes the light look like
# it comes from somewhere. Centred on x = 0 because the chosen camera shoots the
# rig square on: off to one side it broke the symmetry of an otherwise matched
# pair of ramps. The ceiling bounce is left off-centre so the lighting still has
# some modelling in it rather than going perfectly flat.
WINDOW = (0.0, 0.16, 1.15, 0.98)


def add_box(name: str, dims, location, material: bpy.types.Material) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dims
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    return obj


def build_room() -> None:
    floor_material = create_polyhaven_material(
        "room_floor", "wood_floor_worn", scale=2.2, base_color=(0.28, 0.20, 0.13),
    )
    bpy.ops.mesh.primitive_plane_add(size=9.0, location=(0.0, 0.0, FLOOR_Z))
    floor = bpy.context.object
    floor.name = "room_floor"
    floor.data.materials.append(floor_material)

    wall_material = create_paint_material("wall_paint", (0.60, 0.585, 0.55))
    win_x, win_sill, win_w, win_h = WINDOW
    win_head = win_sill + win_h
    left_edge = win_x - win_w / 2.0
    right_edge = win_x + win_w / 2.0
    wall_thickness = 0.10
    wall_y = WALL_Y + wall_thickness / 2.0

    def wall_panel(name, x0, x1, z0, z1):
        if x1 - x0 <= 1e-6 or z1 - z0 <= 1e-6:
            return
        add_box(
            name,
            (x1 - x0, wall_thickness, z1 - z0),
            ((x0 + x1) / 2.0, wall_y, (z0 + z1) / 2.0),
            wall_material,
        )

    # Back wall, built as four panels around the window opening.
    wall_panel("wall_back_left", -ROOM_HALF_X, left_edge, FLOOR_Z, WALL_TOP_Z)
    wall_panel("wall_back_right", right_edge, ROOM_HALF_X, FLOOR_Z, WALL_TOP_Z)
    wall_panel("wall_back_under", left_edge, right_edge, FLOOR_Z, win_sill)
    wall_panel("wall_back_over", left_edge, right_edge, win_head, WALL_TOP_Z)

    # The room is closed on all six sides, and that is a lighting decision as
    # much as a framing one. Left open at the top it was still an exterior: every
    # up-facing surface -- which here is every surface that matters, both slopes
    # and the whole valley -- saw the bare world HDRI straight overhead and
    # rendered at 210/255 against an albedo of 120. Putting a ceiling on it is
    # what makes the window the actual light source.
    room_depth = (WALL_Y + wall_thickness) - ROOM_FRONT_Y
    room_mid_y = (ROOM_FRONT_Y + WALL_Y + wall_thickness) / 2.0
    for sign, name in ((-1, "wall_left"), (1, "wall_right")):
        add_box(
            name,
            (wall_thickness, room_depth, WALL_TOP_Z - FLOOR_Z),
            (sign * ROOM_HALF_X, room_mid_y, (FLOOR_Z + WALL_TOP_Z) / 2.0),
            wall_material,
        )
    add_box(
        "wall_front",
        (2.0 * ROOM_HALF_X, wall_thickness, WALL_TOP_Z - FLOOR_Z),
        (0.0, ROOM_FRONT_Y, (FLOOR_Z + WALL_TOP_Z) / 2.0),
        wall_material,
    )
    add_box(
        "ceiling",
        (2.0 * ROOM_HALF_X, room_depth, wall_thickness),
        (0.0, room_mid_y, WALL_TOP_Z + wall_thickness / 2.0),
        create_paint_material("ceiling_paint", (0.86, 0.86, 0.84)),
    )

    # Skirting, the cheapest detail that reads as "a room somebody lives in"
    # rather than as three planes meeting at right angles.
    skirting = create_paint_material("skirting_paint", (0.80, 0.78, 0.74))
    add_box(
        "skirting_back",
        (2.0 * ROOM_HALF_X, 0.018, 0.09),
        (0.0, WALL_Y - 0.009, FLOOR_Z + 0.045),
        skirting,
    )

    # Daylight: a pale panel filling the opening, plus an area light just
    # outside it. The panel alone is far too weak to light a room, and the area
    # light alone leaves a black rectangle where the window should be.
    sky = bpy.data.materials.new("window_sky")
    sky.use_nodes = True
    sky_nodes, sky_links = sky.node_tree.nodes, sky.node_tree.links
    for node in list(sky_nodes):
        sky_nodes.remove(node)
    sky_out = sky_nodes.new("ShaderNodeOutputMaterial")
    emission = sky_nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (0.86, 0.91, 1.0, 1.0)
    emission.inputs["Strength"].default_value = 2.6
    sky_links.new(emission.outputs["Emission"], sky_out.inputs["Surface"])
    add_box(
        "window_sky",
        (win_w, 0.01, win_h),
        (win_x, WALL_Y + wall_thickness + 0.02, win_sill + win_h / 2.0),
        sky,
    )

    # Frame and a centre mullion. Without them the opening renders as a plain
    # white rectangle that reads as a lightbox rather than as a window, which
    # left the room hard to place even once it had walls and a floor.
    frame_material = create_paint_material("window_frame", (0.86, 0.85, 0.82))
    jamb = 0.045
    frame_y = WALL_Y - 0.012
    for name, dims, loc in (
        ("window_sill_board", (win_w + 2 * jamb, 0.055, jamb),
         (win_x, frame_y, win_sill - jamb / 2.0)),
        ("window_head", (win_w + 2 * jamb, 0.030, jamb),
         (win_x, frame_y, win_head + jamb / 2.0)),
        ("window_jamb_left", (jamb, 0.030, win_h),
         (left_edge - jamb / 2.0, frame_y, win_sill + win_h / 2.0)),
        ("window_jamb_right", (jamb, 0.030, win_h),
         (right_edge + jamb / 2.0, frame_y, win_sill + win_h / 2.0)),
        ("window_mullion", (0.032, 0.026, win_h),
         (win_x, frame_y, win_sill + win_h / 2.0)),
    ):
        add_box(name, dims, loc, frame_material)


def build_table() -> bpy.types.Object:
    top_material = create_polyhaven_material(
        "table_top_wood", "wood_table", scale=3.8, base_color=(0.42, 0.30, 0.19),
        normal_strength=0.45,
    )
    frame_material = create_paint_material("table_frame", (0.30, 0.22, 0.15))

    width, depth = TABLE_SIZE
    top = add_box(
        "table_top", (width, depth, TABLE_THICKNESS),
        (0.0, 0.0, TABLE_TOP_Z - TABLE_THICKNESS / 2.0), top_material,
    )
    bevel = top.modifiers.new("bevel", "BEVEL")
    bevel.width = 0.002
    bevel.segments = 2

    apron_z = TABLE_TOP_Z - TABLE_THICKNESS - 0.045
    inset = 0.10
    add_box("table_apron_front", (width - 2 * inset, 0.020, 0.090),
            (0.0, -(depth / 2.0 - inset), apron_z), frame_material)
    add_box("table_apron_back", (width - 2 * inset, 0.020, 0.090),
            (0.0, depth / 2.0 - inset, apron_z), frame_material)

    leg_x = width / 2.0 - inset
    leg_y = depth / 2.0 - inset
    leg_h = (TABLE_TOP_Z - TABLE_THICKNESS) - FLOOR_Z
    for sx in (-1, 1):
        for sy in (-1, 1):
            add_box(
                f"table_leg_{'n' if sx < 0 else 'p'}{'n' if sy < 0 else 'p'}",
                (TABLE_LEG, TABLE_LEG, leg_h),
                (sx * leg_x, sy * leg_y, FLOOR_Z + leg_h / 2.0),
                frame_material,
            )
    return top


# Dressing for the table. Decorative only -- none of these is in the physics
# world, and none is within reach of a ball, which stays on the plank the whole
# time. They are placed behind the rig and out towards the ends rather than
# anywhere near the middle: the camera shoots the apparatus square on, and the
# valley where the two balls actually meet has to stay clean.
#
# Each entry is (glb file, target height in metres, (x, y) on the table,
# z-rotation in degrees). The heights are given in real metres and applied by
# measuring the model, because the GLBs in assets/models are not authored to a
# common scale -- living_room_interior_free.glb, for one, is out by a factor of
# seven.
PROPS = (
    ("potted_plant.glb", 0.300, (0.74, 0.34), -25.0),
)


def import_prop(filename: str, target_height: float, xy, rotation_z_deg: float):
    """Import a GLB prop, scale it to a real height and stand it on the table.

    The scale is measured rather than assumed, and the model is then seated by
    its own bounding box so it rests on the table top instead of floating or
    sinking -- these models do not share an origin convention any more than they
    share a unit.
    """
    path = MODELS_DIR / filename
    if not path.exists():
        print(f"[WARN] Prop model missing, skipping: {path}")
        return None

    scene = bpy.context.scene
    before = set(scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    imported = [o for o in scene.objects if o not in before]
    for obj in list(imported):
        if obj.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
            imported.remove(obj)

    holder = bpy.data.objects.new(f"prop_{path.stem}", None)
    scene.collection.objects.link(holder)
    for obj in imported:
        if obj.parent is None:
            obj.parent = holder
            obj.matrix_parent_inverse = Matrix.Identity(4)
    bpy.context.view_layer.update()

    bounds = mesh_bounds(imported)
    if bounds is None:
        print(f"[WARN] Prop {filename} imported no geometry; skipping.")
        bpy.data.objects.remove(holder, do_unlink=True)
        return None
    lo, hi = bounds
    height = max(hi.z - lo.z, 1e-6)
    factor = float(target_height) / height
    holder.scale = (factor, factor, factor)
    holder.rotation_euler = (0.0, 0.0, math.radians(float(rotation_z_deg)))
    bpy.context.view_layer.update()

    lo, hi = mesh_bounds(imported)
    holder.location = (
        float(xy[0]) - (lo.x + hi.x) / 2.0,
        float(xy[1]) - (lo.y + hi.y) / 2.0,
        TABLE_TOP_Z - lo.z,
    )
    bpy.context.view_layer.update()
    print(f"[INFO] Prop {filename}: model height {height:.3f} -> {target_height:.3f} m "
          f"(x{factor:.4f})")
    return holder


def dress_table() -> None:
    for filename, height, xy, rotation in PROPS:
        import_prop(filename, height, xy, rotation)


# --- Physics ------------------------------------------------------------------

def run_physics(args: argparse.Namespace, scenario: dict) -> dict:
    candidate = WORKSPACE_DIR.parent / "miniconda" / "envs" / "physics" / "bin" / "python"
    python = str(candidate) if candidate.exists() else (
        shutil.which("python3") or shutil.which("python")
    )
    if not python:
        raise RuntimeError("Cannot find a python with pybullet for the simulation.")

    phys = scenario["physics"]
    render = scenario["render"]
    out = args.out_dir / PHYSICS_TEMP_NAME
    command = [
        python, str(Path(__file__).with_name("simulate_twin_ramp_collision.py")),
        "--out", str(out),
        "--fps", str(int(render["fps"])),
        "--duration-sec", str(float(render["duration_sec"])),
        "--substeps", str(int(phys["substeps"])),
        "--hold-sec", str(float(phys["hold_sec"])),
        "--ramp-angle-deg", str(float(phys["ramp_angle_deg"])),
        "--ramp-run", str(float(phys["ramp_run"])),
        "--ramp-width", str(float(phys["ramp_width"])),
        "--ramp-body-thickness", str(float(phys["ramp_body_thickness"])),
        "--valley-half", str(float(phys["valley_half"])),
        "--plank-thickness", str(float(phys["plank_thickness"])),
        "--plank-length", str(float(phys["plank_length"])),
        "--plank-width", str(float(phys["plank_width"])),
        "--ball-radius", str(float(phys["ball_radius"])),
        "--ball-mass", str(float(phys["ball_mass"])),
        "--ball-friction", str(float(phys["ball_friction"])),
        "--ball-restitution", str(float(phys["ball_restitution"])),
        "--ball-rolling-friction", str(float(phys["ball_rolling_friction"])),
        "--ball-spinning-friction", str(float(phys["ball_spinning_friction"])),
        "--track-friction", str(float(phys["track_friction"])),
        "--track-restitution", str(float(phys["track_restitution"])),
        "--release-inset", str(float(phys["release_inset"])),
        "--release-inset-bias", str(float(phys["release_inset_bias"])),
        "--gravity-z", str(float(phys["gravity_z"])),
        "--ball-a-mass",              str(float(phys["ball_a_mass"])),
        "--ball-a-friction",          str(float(phys["ball_a_friction"])),
        "--ball-a-restitution",       str(float(phys["ball_a_restitution"])),
        "--ball-a-rolling-friction",  str(float(phys["ball_a_rolling_friction"])),
        "--ball-b-mass",              str(float(phys["ball_b_mass"])),
        "--ball-b-friction",          str(float(phys["ball_b_friction"])),
        "--ball-b-restitution",       str(float(phys["ball_b_restitution"])),
        "--ball-b-rolling-friction",  str(float(phys["ball_b_rolling_friction"])),
        "--ball-a-active", str(int(phys.get("active", [1, 1])[0])),
        "--ball-b-active", str(int(phys.get("active", [1, 1])[1])),
    ]
    subprocess.run(command, check=True)
    data = json.loads(out.read_text(encoding="utf-8"))
    out.unlink(missing_ok=True)

    q = data["quality"]
    if q["contact_frame"] is None:
        print("[WARN] The two balls never touched in this scenario.")
    elif not q["contact_inside_valley"]:
        print("[WARN] The balls met on a ramp rather than on the flat valley.")
    if q["left_track"]:
        print("[WARN] A ball left the plank.")
    return data


def apply_keyframes(ball: bpy.types.Object, frames: list, key: str) -> None:
    ball.rotation_mode = "QUATERNION"
    for record in frames:
        data = record["balls"][key]
        ball.location = data["location"]
        q = data["quaternion_xyzw"]
        ball.rotation_quaternion = (q[3], q[0], q[1], q[2])
        frame = int(record["frame_index"])
        ball.keyframe_insert(data_path="location", frame=frame)
        ball.keyframe_insert(data_path="rotation_quaternion", frame=frame)
    if ball.animation_data and ball.animation_data.action:
        for fcurve in ball.animation_data.action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = "LINEAR"


# --- Scene --------------------------------------------------------------------

def build_scene(args: argparse.Namespace, scenario: dict, physics: dict):
    scene = bpy.context.scene
    render = scenario["render"]
    phys = scenario["physics"]

    scene.render.resolution_x = int(render["resolution"][0])
    scene.render.resolution_y = int(render["resolution"][1])
    scene.render.fps = int(render["fps"])
    scene.render.engine = "CYCLES"
    scene.cycles.samples = int(render["samples"])
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 12
    scene.cycles.glossy_bounces = 8
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.view_settings.look = "Medium Contrast"
    # The apparatus is all up-facing surfaces under a full environment
    # hemisphere, so it collects far more irradiance than the vertical faces
    # do. At exposure 0 the slopes came out around 1.7x their own albedo and
    # Filmic desaturated them to a pale grey -- the wood read as primed MDF and
    # the grain disappeared, even though an emission-only pass confirmed the
    # texture was reaching every face correctly.
    scene.view_settings.exposure = -0.35

    scene.render.use_motion_blur = bool(render.get("motion_blur", True))
    scene.render.motion_blur_shutter = float(render.get("motion_blur_shutter", 0.5))

    if str(render["device"]) == "auto":
        enable_gpu()
        scene.cycles.device = "GPU"
    else:
        scene.cycles.device = "CPU"

    # World. Now that the room is enclosed, the HDRI is never the background --
    # it only leaks in through the window, as an outdoors would. It is kept
    # (rather than replaced by a flat sky colour) because the balls are mirrors
    # and it is what gives their reflections something with structure in it.
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    for node in list(nodes):
        nodes.remove(node)
    output = nodes.new("ShaderNodeOutputWorld")
    background = nodes.new("ShaderNodeBackground")
    hdri = POLYHAVEN_DIR / "wooden_lounge" / "wooden_lounge_2k.hdr"
    if hdri.exists():
        tex_coord = nodes.new("ShaderNodeTexCoord")
        mapping = nodes.new("ShaderNodeMapping")
        mapping.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(120.0))
        env = nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(str(hdri), check_existing=True)
        links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], env.inputs["Vector"])
        links.new(env.outputs["Color"], background.inputs["Color"])
        background.inputs["Strength"].default_value = 0.9
    else:
        print(f"[WARN] HDRI not found at {hdri}; falling back to a flat sky colour.")
        background.inputs["Color"].default_value = (0.5, 0.53, 0.58, 1.0)
        background.inputs["Strength"].default_value = 1.8
    links.new(background.outputs["Background"], output.inputs["Surface"])

    track = build_track(
        ramp_angle_deg=phys["ramp_angle_deg"],
        ramp_run=phys["ramp_run"],
        valley_half=phys["valley_half"],
        ramp_width=phys["ramp_width"],
        ramp_body_thickness=phys["ramp_body_thickness"],
        plank_thickness=phys["plank_thickness"],
        plank_length=phys["plank_length"],
        plank_width=phys["plank_width"],
    )

    build_room()
    build_table()
    dress_table()
    plank_material = create_wood_material(
        "plank_wood", scale=1.6, tint=(0.88, 0.80, 0.70), rotation_z=0.0,
    )
    ramp_material = create_wood_material(
        "ramp_wood", scale=2.4, tint=(0.66, 0.58, 0.49),
        rotation_z=math.radians(90.0), roughness_scale=1.1,
    )
    build_plank(track, plank_material)
    build_wedge(track, 1, ramp_material)
    build_wedge(track, -1, ramp_material)

    # The two balls are the same model recoloured, so they stay physically
    # identical while remaining individually trackable through the impact --
    # which matters, because the whole point of the shot is which ball ends up
    # where afterwards.
    diameter = 2.0 * float(phys["ball_radius"])
    ball_cfg = scenario.get("balls", {})
    a_cfg = ball_cfg.get("a", {"hue": 0.12, "saturation": 1.05})
    b_cfg = ball_cfg.get("b", {"hue": 0.50, "saturation": 1.15})
    ball_a = import_marble(
        "ball_a", diameter, hue=float(a_cfg["hue"]), saturation=float(a_cfg["saturation"]),
    )
    ball_b = import_marble(
        "ball_b", diameter, hue=float(b_cfg["hue"]), saturation=float(b_cfg["saturation"]),
    )
    # DELETE edit: sim skipped creating that ball; hide it from the render
    # rather than trying to animate frozen frames.
    a_present = bool(physics["objects"]["ball_a"].get("present", True))
    b_present = bool(physics["objects"]["ball_b"].get("present", True))
    if a_present:
        apply_keyframes(ball_a, physics["frames"], "a")
    else:
        ball_a.hide_viewport = True
        ball_a.hide_render = True
    if b_present:
        apply_keyframes(ball_b, physics["frames"], "b")
    else:
        ball_b.hide_viewport = True
        ball_b.hide_render = True

    cam_cfg = scenario["camera"]
    bpy.ops.object.camera_add(location=tuple(cam_cfg["location"]))
    camera = bpy.context.object
    camera.data.lens = float(cam_cfg["lens_mm"])
    camera.data.sensor_width = 36.0
    camera.data.clip_start = 0.01
    scene.camera = camera
    target = bpy.data.objects.new("camera_target", None)
    scene.collection.objects.link(target)
    target.location = tuple(cam_cfg["target"])
    constraint = camera.constraints.new("TRACK_TO")
    constraint.target = target
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"

    # Daylight through the window, which is behind and slightly to the right of
    # the apparatus, so the balls throw their shadows towards the camera and the
    # two sloped faces are lit from opposite sides -- that difference is what
    # keeps the two ramps from reading as one flat shape.
    win_x, win_sill, win_w, win_h = WINDOW
    bpy.ops.object.light_add(
        type="AREA", location=(win_x, WALL_Y - 0.06, win_sill + win_h / 2.0),
    )
    window_light = bpy.context.object
    window_light.name = "window_light"
    window_light.data.shape = "RECTANGLE"
    window_light.data.size = win_w
    window_light.data.size_y = win_h
    window_light.data.energy = 150.0
    window_light.data.color = (0.94, 0.96, 1.0)
    window_light.visible_camera = False
    # Pointed straight out of the wall, not at the apparatus. Aiming it at the
    # rig tilts a vertical window into an overhead source, and since every
    # surface that matters here faces up, that blew the two slopes to 232/255 --
    # flat white, no grain, no shading difference between the ramps at all.
    look_at(window_light, (win_x, WALL_Y - 2.0, win_sill + win_h / 2.0))

    # A ceiling bounce standing in for the light the room would throw back down.
    # Without it the enclosed room swallows everything the window does not hit
    # directly and the near faces of the wedges go to mud.
    bpy.ops.object.light_add(type="AREA", location=(-0.30, -0.35, 1.55))
    bounce = bpy.context.object
    bounce.name = "ceiling_bounce"
    bounce.data.energy = 12.0
    bounce.data.size = 2.4
    bounce.data.color = (1.0, 0.98, 0.94)
    bounce.visible_camera = False
    look_at(bounce, (0.0, 0.0, 0.02))

    # Soft fill from the camera side. The window is behind the rig, so every
    # face turned towards the lens is a shadow face: without this the plank's
    # front edge and the wedge cheeks sat two stops under their own slopes and
    # the rig read as pale board with black trim rather than as one piece of
    # wood. Kept large and weak so it lifts the near faces without touching the
    # tops, which the window already handles.
    bpy.ops.object.light_add(type="AREA", location=(0.0, -1.55, 0.55))
    fill = bpy.context.object
    fill.name = "camera_fill"
    fill.data.energy = 24.0
    fill.data.size = 2.4
    fill.data.color = (1.0, 0.98, 0.95)
    fill.visible_camera = False
    look_at(fill, (0.0, 0.0, 0.03))

    scene.frame_start = 1
    scene.frame_end = int(physics["frame_end"])
    scene.frame_set(1)
    return ball_a, ball_b, camera, track


# --- Outputs ------------------------------------------------------------------

def export_ground_truth(out_dir: Path, ball_a, ball_b, camera, track, physics: dict, scenario: dict) -> None:
    scene = bpy.context.scene
    fps = int(physics["fps"])
    frame_end = int(physics["frame_end"])
    radius = float(scenario["physics"]["ball_radius"])
    records = {
        "schema_version": 1,
        "fps": fps,
        "frame_start": 1,
        "frame_end": frame_end,
        "physics": {k: v for k, v in physics.items() if k != "frames"},
        "objects": {
            "ball_a": {
                "object_name": ball_a.name, "radius": radius, "side": 1,
                "present": not bool(ball_a.hide_render),
            },
            "ball_b": {
                "object_name": ball_b.name, "radius": radius, "side": -1,
                "present": not bool(ball_b.hide_render),
            },
            "track": {
                "track_z": track.track_z,
                "valley_half": track.valley_half,
                "ramp_run": track.run,
                "ramp_rise": track.rise,
                "ramp_angle_deg": math.degrees(track.angle_rad),
            },
        },
        "camera": {
            "object_name": camera.name,
            "lens_mm": float(camera.data.lens),
            "sensor_width_mm": float(camera.data.sensor_width),
            "resolution": [int(scene.render.resolution_x), int(scene.render.resolution_y)],
        },
        "scenario": scenario,
        "frames": [],
    }
    by_frame = {int(f["frame_index"]): f for f in physics["frames"]}
    for frame in range(1, frame_end + 1):
        scene.frame_set(frame)
        pf = by_frame[frame]
        entry = {
            "frame_index": frame,
            "time_sec": (frame - 1) / float(fps),
            "released": pf["released"],
            "in_contact": pf["in_contact"],
            "gap_between_balls": pf["gap_between_balls"],
            "camera_matrix_world": [[float(v) for v in row] for row in camera.matrix_world],
        }
        for name, obj in (("ball_a", ball_a), ("ball_b", ball_b)):
            key = name.split("_")[1]
            if bool(obj.hide_render):
                entry[name] = {"present": False}
                continue
            entry[name] = {
                "present": True,
                "matrix_world": [[float(v) for v in row] for row in obj.matrix_world],
                "location": [float(v) for v in obj.location],
                "linear_velocity": pf["balls"][key]["linear_velocity"],
                "angular_velocity": pf["balls"][key]["angular_velocity"],
                "speed": pf["balls"][key]["speed"],
                "height_above_track": pf["balls"][key]["height_above_track"],
            }
        records["frames"].append(entry)
    (out_dir / GROUND_TRUTH_NAME).write_text(json.dumps(records, indent=2), encoding="utf-8")


def render_preview(args: argparse.Namespace) -> None:
    scene = bpy.context.scene
    scene.render.image_settings.file_format = "PNG"
    frames = args.preview_frames or [args.preview_frame]
    for frame in frames:
        clamped = max(scene.frame_start, min(int(frame), scene.frame_end))
        scene.frame_set(clamped)
        name = "preview.png" if len(frames) == 1 else f"preview_f{clamped:03d}.png"
        scene.render.filepath = str(args.out_dir / name)
        bpy.ops.render.render(write_still=True)


def render_frames(args: argparse.Namespace) -> None:
    scene = bpy.context.scene
    scene.render.image_settings.file_format = "PNG"
    scene.frame_set(scene.frame_start)
    scene.render.filepath = str(args.out_dir / "frame_")
    bpy.ops.render.render(animation=True)


def render_animation(args: argparse.Namespace) -> None:
    scene = bpy.context.scene
    scene.frame_set(scene.frame_start)
    scene.render.filepath = str(args.out_dir / f"{OUTPUT_STEM}.mp4")
    bpy.ops.render.render(animation=True)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    scenario = build_scenario(args)
    (args.out_dir / SCENARIO_METADATA_NAME).write_text(
        json.dumps(scenario, indent=2), encoding="utf-8",
    )

    physics = run_physics(args, scenario)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    ball_a, ball_b, camera, track = build_scene(args, scenario, physics)
    export_ground_truth(args.out_dir, ball_a, ball_b, camera, track, physics, scenario)

    if args.mode == "preview":
        render_preview(args)
    elif args.mode == "frames":
        render_frames(args)
    else:
        render_animation(args)

    bpy.ops.wm.save_as_mainfile(filepath=str(args.out_dir / BLEND_NAME))
    print(f"[INFO] Render complete. Output: {args.out_dir}")


if __name__ == "__main__":
    main()
