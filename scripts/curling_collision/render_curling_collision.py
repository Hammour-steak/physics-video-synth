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
from mathutils import Vector


WORKSPACE_DIR = Path(__file__).resolve().parents[2]
POLYHAVEN_DIR = WORKSPACE_DIR / "assets" / "polyhaven"

OUTPUT_STEM = "curling_collision"
DIRECT_MP4_NAME = f"{OUTPUT_STEM}.mp4"
BLEND_NAME = f"{OUTPUT_STEM}.blend"
GROUND_TRUTH_NAME = "ground_truth_transforms.json"
PHYSICS_TEMP_NAME = "physics_transforms.json"
SCENARIO_METADATA_NAME = "scenario_metadata.json"

# Geometry matches simulate_curling_collision.py.
STONE_RADIUS = 0.145
STONE_HEIGHT = 0.114
FLOOR_Z = 0.0
ICE_HALF_LENGTH = 20.0
ICE_HALF_WIDTH = 14.0

CAMERA_LOCATION = (1.2, -8.5, 4.3)
CAMERA_TARGET = (0.0, 0.0, FLOOR_Z + 0.1)
LIGHT_TARGET = CAMERA_TARGET


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
    parser.add_argument("--duration-sec", type=float, default=6.0)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--preview-frame", type=int, default=30)
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


def set_input_default(node: bpy.types.ShaderNode, input_name: str, value) -> None:
    socket = node.inputs.get(input_name)
    if socket is not None:
        socket.default_value = value


def create_ice_material() -> bpy.types.Material:
    mat = bpy.data.materials.new("curling_ice")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        return mat

    set_input_default(bsdf, "Roughness", 0.12)
    set_input_default(bsdf, "Metallic", 0.0)
    set_input_default(bsdf, "Transmission", 0.05)

    # Painted "house" (the concentric target rings) centered on the ice
    # sheet's origin, i.e. the collision point, fading to plain ice color
    # beyond its outer (blue) ring -- a radial-distance lookup, not a
    # repeating/periodic ring texture, so there is exactly one house rather
    # than the sheet being tiled with rings everywhere.
    tex_coord = nodes.new(type="ShaderNodeTexCoord")

    length_node = nodes.new(type="ShaderNodeVectorMath")
    length_node.operation = "LENGTH"
    links.new(tex_coord.outputs["Object"], length_node.inputs[0])

    # Regulation "house" radii: button (1 ft) 0.152 m, 4-ft circle 0.610 m,
    # 8-ft circle 1.219 m, 12-ft circle (outer) 1.829 m -- roughly 12.5x the
    # stone's own 0.146 m radius, not the small ~0.5 m target used earlier.
    HOUSE_OUTER_RADIUS = 1.829
    ring_range = nodes.new(type="ShaderNodeMapRange")
    ring_range.inputs["From Min"].default_value = 0.0
    ring_range.inputs["From Max"].default_value = HOUSE_OUTER_RADIUS
    ring_range.inputs["To Min"].default_value = 0.0
    ring_range.inputs["To Max"].default_value = 1.0
    ring_range.clamp = True
    links.new(length_node.outputs["Value"], ring_range.inputs["Value"])

    ring_ramp = nodes.new(type="ShaderNodeValToRGB")
    ring_ramp.color_ramp.interpolation = "CONSTANT"
    elems = ring_ramp.color_ramp.elements
    elems[0].position = 0.0
    elems[0].color = (0.92, 0.93, 0.95, 1.0)  # button (white, 1-foot)
    elems[1].position = 0.152 / HOUSE_OUTER_RADIUS
    elems[1].color = (0.85, 0.08, 0.08, 1.0)  # red (4-foot)
    e2 = elems.new(0.610 / HOUSE_OUTER_RADIUS)
    e2.color = (0.92, 0.93, 0.95, 1.0)  # white (8-foot)
    e3 = elems.new(1.219 / HOUSE_OUTER_RADIUS)
    e3.color = (0.06, 0.28, 0.62, 1.0)  # blue (12-foot, outer)
    links.new(ring_range.outputs["Result"], ring_ramp.inputs["Fac"])

    mask_range = nodes.new(type="ShaderNodeMapRange")
    mask_range.inputs["From Min"].default_value = HOUSE_OUTER_RADIUS - 0.04
    mask_range.inputs["From Max"].default_value = HOUSE_OUTER_RADIUS
    mask_range.inputs["To Min"].default_value = 0.0
    mask_range.inputs["To Max"].default_value = 1.0
    mask_range.clamp = True
    links.new(length_node.outputs["Value"], mask_range.inputs["Value"])

    # Center line: the stones travel along local X, so the painted line runs
    # along X at Y=0, the length of the sheet -- drawn into the plain-ice
    # color so the house rings still take priority near the target.
    abs_y = nodes.new(type="ShaderNodeVectorMath")
    abs_y.operation = "ABSOLUTE"
    links.new(tex_coord.outputs["Object"], abs_y.inputs[0])

    line_sep = nodes.new(type="ShaderNodeSeparateXYZ")
    links.new(abs_y.outputs["Vector"], line_sep.inputs["Vector"])

    line_mask = nodes.new(type="ShaderNodeMapRange")
    line_mask.inputs["From Min"].default_value = 0.015
    line_mask.inputs["From Max"].default_value = 0.05
    line_mask.inputs["To Min"].default_value = 1.0
    line_mask.inputs["To Max"].default_value = 0.0
    line_mask.clamp = True
    links.new(line_sep.outputs["Y"], line_mask.inputs["Value"])

    plain_ice_with_line = nodes.new(type="ShaderNodeMixRGB")
    plain_ice_with_line.inputs["Color1"].default_value = (0.83, 0.91, 0.96, 1.0)  # plain ice
    plain_ice_with_line.inputs["Color2"].default_value = (0.65, 0.08, 0.08, 1.0)  # centerline
    links.new(line_mask.outputs["Result"], plain_ice_with_line.inputs["Fac"])

    ice_color = nodes.new(type="ShaderNodeMixRGB")
    links.new(ring_ramp.outputs["Color"], ice_color.inputs["Color1"])
    links.new(plain_ice_with_line.outputs["Color"], ice_color.inputs["Color2"])
    links.new(mask_range.outputs["Result"], ice_color.inputs["Fac"])
    links.new(ice_color.outputs["Color"], bsdf.inputs["Base Color"])

    # Pebbled ice: real curling ice is sprayed with fine water droplets that
    # freeze into a bumpy "pebble" texture (what lets stones curl and glide
    # with so little friction), not a perfectly smooth mirror surface.
    pebble_noise = nodes.new(type="ShaderNodeTexNoise")
    pebble_noise.inputs["Scale"].default_value = 180.0
    pebble_noise.inputs["Detail"].default_value = 4.0
    pebble_noise.inputs["Roughness"].default_value = 0.6
    links.new(tex_coord.outputs["Object"], pebble_noise.inputs["Vector"])

    pebble_bump = nodes.new(type="ShaderNodeBump")
    pebble_bump.inputs["Strength"].default_value = 0.05
    pebble_bump.inputs["Distance"].default_value = 0.002
    links.new(pebble_noise.outputs["Fac"], pebble_bump.inputs["Height"])
    links.new(pebble_bump.outputs["Normal"], bsdf.inputs["Normal"])

    return mat


# assets/models/curling_stone.glb is a photogrammetry scan (Global Digital
# Heritage, CC-BY-NC -- internal/research use only, not for redistribution)
# of a real curling stone, handle included. Its raw units aren't meters: the
# scanned puck's horizontal footprint measures 40 units across, matching a
# regulation stone's 0.292 m diameter, which gives the scale factor below.
# The puck body (excluding the handle) sits roughly in the lower ~65% of the
# mesh's vertical extent; that midpoint is used to recenter the origin on
# the puck's own center so placing the object by `.location` matches
# PyBullet's cylinder-center convention, with the handle then hanging
# correctly above it.
GLB_PATH = WORKSPACE_DIR / "assets" / "models" / "curling_stone.glb"
STONE_SCAN_SCALE = 0.292 / 40.0
STONE_SCAN_PUCK_CENTER_Z = -0.78


def import_curling_stone_master() -> bpy.types.Object:
    if not GLB_PATH.exists():
        raise FileNotFoundError(f"Curling stone model not found: {GLB_PATH}")

    existing = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(GLB_PATH))
    imported = [obj for obj in bpy.context.scene.objects if obj not in existing]
    imported_names = [obj.name for obj in imported]
    mesh_objs = [obj for obj in imported if obj.type == "MESH"]
    if not mesh_objs:
        raise RuntimeError(f"No mesh found in {GLB_PATH}")
    stone_mesh = mesh_objs[0]

    bpy.ops.object.select_all(action="DESELECT")
    stone_mesh.select_set(True)
    bpy.context.view_layer.objects.active = stone_mesh
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    bpy.context.scene.cursor.location = (0.0, 0.0, STONE_SCAN_PUCK_CENTER_Z)
    bpy.ops.object.origin_set(type="ORIGIN_CURSOR")

    stone_mesh.scale = (STONE_SCAN_SCALE, STONE_SCAN_SCALE, STONE_SCAN_SCALE)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    stone_mesh.location = (0.0, 0.0, 0.0)
    stone_mesh.rotation_mode = "QUATERNION"
    stone_mesh.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
    stone_mesh.name = "curling_stone_master"

    for name in imported_names:
        if name == stone_mesh.name or name not in bpy.data.objects:
            continue
        obj = bpy.data.objects.get(name)
        if obj is not None and obj is not stone_mesh:
            bpy.data.objects.remove(obj, do_unlink=True)

    return stone_mesh


def recolor_stone_handle(obj: bpy.types.Object, hue: float) -> None:
    """Give this stone an independent copy of the scanned material with its
    base-color texture hue-shifted, so the two stones read as different
    teams (same granite body, different handle color) instead of two
    identical copies of the same asset. The granite body is low-saturation
    grey, so a hue rotation barely affects it -- almost all the visible
    change lands on the saturated red top disc / handle.
    `hue` is Blender's 0-1 wrapped hue value (0.5 = unchanged, +/-1.0 = full
    360 degree rotation back to the start).

    `obj` shares its mesh data (and hence material slots) with the other
    stone via linked-duplicate `.copy()`, so writing `obj.data.materials[0]`
    would recolor both stones at once. Setting the slot link to 'OBJECT'
    first gives this specific object its own material override instead.
    """
    if not obj.material_slots:
        return
    slot = obj.material_slots[0]
    original = slot.material
    if original is None:
        return
    new_mat = original.copy()
    slot.link = "OBJECT"
    slot.material = new_mat
    node_tree = new_mat.node_tree
    if node_tree is None:
        return
    bsdf = node_tree.nodes.get("Principled BSDF")
    base_tex = node_tree.nodes.get("Image Texture")
    if bsdf is None or base_tex is None:
        return
    hue_node = node_tree.nodes.new(type="ShaderNodeHueSaturation")
    hue_node.inputs["Hue"].default_value = hue
    node_tree.links.new(base_tex.outputs["Color"], hue_node.inputs["Color"])
    node_tree.links.new(hue_node.outputs["Color"], bsdf.inputs["Base Color"])


def place_curling_stone(
    name: str,
    location: tuple[float, float, float],
    obj: bpy.types.Object,
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
) -> bpy.types.Object:
    bpy.ops.object.light_add(type="AREA", location=location)
    light = bpy.context.object
    light.name = name
    if hasattr(light.data, "power"):
        light.data.power = float(power)
    else:
        light.data.energy = float(power)
    light.data.size = float(size)
    light.visible_camera = False
    look_at(light, target)
    return light


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
            },
            "camera": {
                "location": list(CAMERA_LOCATION),
                "target": list(CAMERA_TARGET),
                "lens_mm": 45.0,
            },
            "physics": {
                "stone_radius": STONE_RADIUS,
                "stone_height": STONE_HEIGHT,
                "stone_mass": 20.0,
                "stone_2_mass": 20.0,
                "stone_friction": 0.15,
                "stone_restitution": 0.0,
                "ice_friction": 0.015,
                "launch_speed": 1.5,
                "start_separation": 5.0,
                "gravity": [0.0, 0.0, -9.8],
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
    physics_python = WORKSPACE_DIR.parent / "miniconda" / "envs" / "physics" / "bin" / "python"
    if physics_python.exists():
        python = str(physics_python)
    else:
        python = shutil.which("python3") or shutil.which("python")
    if not python:
        raise RuntimeError("Cannot find python3/python for the PyBullet physics simulation.")

    physics = scenario["physics"]
    assert isinstance(physics, dict)

    script_path = Path(__file__).with_name("simulate_curling_collision.py")
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
            "--stone-radius",
            str(float(physics["stone_radius"])),
            "--stone-height",
            str(float(physics["stone_height"])),
            "--stone-mass",
            str(float(physics["stone_mass"])),
            "--stone-2-mass",
            str(float(physics["stone_2_mass"])),
            "--stone-friction",
            str(float(physics["stone_friction"])),
            "--stone-restitution",
            str(float(physics["stone_restitution"])),
            "--ice-friction",
            str(float(physics["ice_friction"])),
            "--launch-speed",
            str(float(physics["launch_speed"])),
            "--stone-1-launch-speed",
            str(float(physics.get("stone_1_launch_speed", physics["launch_speed"]))),
            "--stone-2-launch-speed",
            str(float(physics.get("stone_2_launch_speed", physics["launch_speed"]))),
            "--start-separation",
            str(float(physics["start_separation"])),
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


def apply_physics_animation(stones: list[bpy.types.Object], physics: dict) -> None:
    for obj in stones:
        obj.rotation_mode = "QUATERNION"
    for frame_record in physics["frames"]:
        frame = int(frame_record["frame_index"])
        for idx, stone_obj in enumerate(stones):
            stone_data = frame_record["stones"][idx]
            quat = stone_data["quaternion_xyzw"]
            stone_obj.location = stone_data["location"]
            stone_obj.rotation_quaternion = (quat[3], quat[0], quat[1], quat[2])
            stone_obj.keyframe_insert(data_path="location", frame=frame)
            stone_obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)
    set_linear_keyframes(stones)


def export_ground_truth(
    out_dir: Path,
    stones: list[bpy.types.Object],
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
            "stones": [
                {"object_name": stone.name, "index": idx}
                for idx, stone in enumerate(stones)
            ],
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
        "scenario": {"seed": int(scenario["seed"])},
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
                "stones": [
                    {
                        "matrix_world": [[float(v) for v in row] for row in stone.matrix_world],
                        "location": [float(v) for v in stone.location],
                        "linear_velocity": stone_data["linear_velocity"],
                        "angular_velocity": stone_data["angular_velocity"],
                    }
                    for stone, stone_data in zip(stones, physics_frame["stones"])
                ],
                "camera_matrix_world": [[float(v) for v in row] for row in camera.matrix_world],
                "camera_world_to_camera_matrix": [
                    [float(v) for v in row] for row in camera.matrix_world.inverted()
                ],
            }
        )
    output_path(out_dir, GROUND_TRUTH_NAME).write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )


def build_scene(args: argparse.Namespace, scenario: dict[str, object]) -> tuple[list[bpy.types.Object], bpy.types.Object]:
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

    env_tex = world_nodes.new(type="ShaderNodeTexEnvironment")
    env_tex.location = (-300, 0)
    hdri_path = POLYHAVEN_DIR / "outdoor" / "netball_court_2k.hdr"
    if hdri_path.exists():
        hdri_img = bpy.data.images.load(str(hdri_path), check_existing=True)
        env_tex.image = hdri_img
        print("[INFO] Using HDRI background: netball_court (indoor sports hall)")
    else:
        print(f"[WARN] HDRI not found at {hdri_path}, using solid fallback")
        bg_node = world_nodes.new(type="ShaderNodeBackground")
        bg_node.inputs["Color"].default_value = (0.15, 0.17, 0.2, 1.0)
        bg_node.inputs["Strength"].default_value = 1.0
        env_tex = bg_node

    mapping_node = world_nodes.new(type="ShaderNodeMapping")
    mapping_node.location = (-550, 0)
    mapping_node.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(90))
    tex_coord_node = world_nodes.new(type="ShaderNodeTexCoord")
    tex_coord_node.location = (-750, 0)
    output_node = world_nodes.new(type="ShaderNodeOutputWorld")

    if isinstance(env_tex, bpy.types.ShaderNodeTexEnvironment):
        world_links.new(tex_coord_node.outputs["Generated"], mapping_node.inputs["Vector"])
        world_links.new(mapping_node.outputs["Vector"], env_tex.inputs["Vector"])
        bg_node = world_nodes.new(type="ShaderNodeBackground")
        bg_node.location = (0, 0)
        bg_node.inputs["Strength"].default_value = 0.6
        world_links.new(env_tex.outputs["Color"], bg_node.inputs["Color"])
        world_links.new(bg_node.outputs["Background"], output_node.inputs["Surface"])
    else:
        world_links.new(env_tex.outputs["Background"], output_node.inputs["Surface"])

    bpy.ops.object.camera_add(location=CAMERA_LOCATION)
    camera = bpy.context.object
    camera.data.lens = 45
    scene.camera = camera
    target_obj = bpy.data.objects.new("camera_target", None)
    scene.collection.objects.link(target_obj)
    target_obj.location = CAMERA_TARGET
    constraint = camera.constraints.new("TRACK_TO")
    constraint.target = target_obj
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"

    bpy.ops.object.light_add(type="SUN", location=(1.5, -2.0, 3.0))
    sun = bpy.context.object
    sun.data.energy = 2.2
    sun.rotation_euler = (math.radians(55), math.radians(10), math.radians(35))

    add_area_light(
        "fill_light", location=(-2.5, -2.0, 1.5), power=60, size=1.0, target=LIGHT_TARGET,
    )
    add_area_light(
        "rim_light", location=(2.3, 1.5, 1.2), power=35, size=0.8, target=LIGHT_TARGET,
    )

    ice_mat = create_ice_material()
    bpy.ops.mesh.primitive_cube_add(
        size=1.0, location=(0.0, 0.0, FLOOR_Z - 0.01),
    )
    ice = bpy.context.object
    ice.name = "ice_sheet"
    ice.dimensions = (2 * ICE_HALF_LENGTH, 2 * ICE_HALF_WIDTH, 0.02)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    ice.data.materials.append(ice_mat)

    physics = scenario["physics"]
    assert isinstance(physics, dict)
    height = float(physics["stone_height"])
    separation = float(physics["start_separation"])

    master = import_curling_stone_master()
    copy_obj = master.copy()
    bpy.context.scene.collection.objects.link(copy_obj)

    stone_0 = place_curling_stone(
        "stone_0", (-separation / 2.0, 0.0, FLOOR_Z + height / 2.0), master,
    )
    stone_1 = place_curling_stone(
        "stone_1", (separation / 2.0, 0.0, FLOOR_Z + height / 2.0), copy_obj,
    )
    recolor_stone_handle(stone_1, hue=0.667)  # shift red -> yellow/gold
    stones = [stone_0, stone_1]

    frame_end = max(2, int(round(float(args.duration_sec) * int(args.fps))))
    scene.frame_start = 1
    scene.frame_end = frame_end

    return stones, camera


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

    stones, camera = build_scene(args, scenario)
    physics = run_physics_simulation(args, scenario)
    apply_physics_animation(stones, physics)
    export_ground_truth(
        out_dir, stones, camera, bpy.context.scene.frame_end, int(args.fps), physics, scenario,
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
