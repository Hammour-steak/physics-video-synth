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

OUTPUT_STEM = "toy_car_ball"
DIRECT_MP4_NAME = f"{OUTPUT_STEM}.mp4"
BLEND_NAME = f"{OUTPUT_STEM}.blend"
GROUND_TRUTH_NAME = "ground_truth_transforms.json"
PHYSICS_TEMP_NAME = "physics_transforms.json"
SCENARIO_METADATA_NAME = "scenario_metadata.json"

# Geometry matches simulate_toy_car_ball.py. The car and ball travel along
# world X, parallel to the backdrop wall (matching the reference photo's
# left-to-right framing), so the shelf is wide in X (room to travel) and
# shallow in Y (a real wall shelf's depth) rather than the other way round.
CAR_HALF_WIDTH = 0.0504
CAR_HALF_LENGTH = 0.11685
CAR_HALF_HEIGHT = 0.03255
BALL_RADIUS = 0.06

# -90 degree rotation about Z, matching simulate_toy_car_ball.py: the car's
# local -Y (nose) ends up pointing toward world -X, its direction of travel.
CAR_INITIAL_QUAT_XYZW = (0.0, 0.0, -0.70710678, 0.70710678)

TABLE_Z = 0.75
TABLE_HALF_X = 0.5
TABLE_HALF_Y = 0.15
TABLE_THICKNESS = 0.03
SHELF_DEPTH_Y = 0.0

FLOOR_HALF_X = 3.5
FLOOR_HALF_Y = 1.2
FLOOR_THICKNESS = 0.05

# The shelf is wall-mounted (no legs): a dark backdrop wall sits flush behind
# its back edge (the +y side, where the car starts), matching a plain
# studio-style physics test-bench shot -- white floating shelf, dark wall,
# a sliver of light wood floor at the very bottom of frame.
WALL_GAP = 0.0
WALL_HALF_X = 3.5
WALL_THICKNESS = 0.05
WALL_HEIGHT = 3.5

# A baseboard/skirting board at the wall-floor junction -- without it the
# dark wall meets the light floor in a hard, unnaturally sharp seam.
BASEBOARD_HEIGHT = 0.08
BASEBOARD_DEPTH = 0.02

CAMERA_LOCATION = (-0.5, -2.3, 0.95)
CAMERA_TARGET = (-0.5, 0.0, 0.45)
CAMERA_LENS_MM = 40.0
LIGHT_TARGET = (0.0, -0.1, TABLE_Z)


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
    parser.add_argument("--duration-sec", type=float, default=3.5)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--preview-frame", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cpu"), default="cpu")
    parser.add_argument("--seed", type=int, default=21)
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


def create_pbr_material(
    name: str,
    asset_name: str,
    *,
    scale: float = 4.0,
    roughness_boost: float = 0.0,
    value_mult: float = 1.0,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        return mat

    set_input_default(bsdf, "Roughness", 0.5)
    set_input_default(bsdf, "Metallic", 0.0)

    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    mapping = nodes.new(type="ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (scale, scale, scale)
    links.new(tex_coord.outputs["UV"], mapping.inputs["Vector"])

    try:
        diff_path = require_polyhaven_path(asset_name, f"{asset_name}_diff_4k.jpg")
        diff_img = load_image(diff_path, "sRGB")
        diff_img.colorspace_settings.name = "sRGB"
        diff_tex = nodes.new(type="ShaderNodeTexImage")
        diff_tex.image = diff_img
        links.new(mapping.outputs["Vector"], diff_tex.inputs["Vector"])
        links.new(diff_tex.outputs["Color"], bsdf.inputs["Base Color"])

        rough_path = require_polyhaven_path(asset_name, f"{asset_name}_rough_4k.jpg")
        rough_img = load_image(rough_path, "Non-Color")
        rough_img.colorspace_settings.name = "Non-Color"
        rough_tex = nodes.new(type="ShaderNodeTexImage")
        rough_tex.image = rough_img
        links.new(mapping.outputs["Vector"], rough_tex.inputs["Vector"])
        if roughness_boost:
            add_node = nodes.new(type="ShaderNodeMath")
            add_node.operation = "ADD"
            add_node.inputs[1].default_value = roughness_boost
            links.new(rough_tex.outputs["Color"], add_node.inputs[0])
            links.new(add_node.outputs["Value"], bsdf.inputs["Roughness"])
        else:
            links.new(rough_tex.outputs["Color"], bsdf.inputs["Roughness"])

        nor_path = require_polyhaven_path(asset_name, f"{asset_name}_nor_gl_4k.jpg")
        nor_img = load_image(nor_path, "Non-Color")
        nor_img.colorspace_settings.name = "Non-Color"
        nor_tex = nodes.new(type="ShaderNodeTexImage")
        nor_tex.image = nor_img
        nor_map = nodes.new(type="ShaderNodeNormalMap")
        nor_map.inputs["Strength"].default_value = 0.5
        links.new(mapping.outputs["Vector"], nor_tex.inputs["Vector"])
        links.new(nor_tex.outputs["Color"], nor_map.inputs["Color"])
        links.new(nor_map.outputs["Normal"], bsdf.inputs["Normal"])

        ao_path = require_polyhaven_path(asset_name, f"{asset_name}_ao_4k.jpg")
        ao_img = load_image(ao_path, "Non-Color")
        ao_img.colorspace_settings.name = "Non-Color"
        ao_tex = nodes.new(type="ShaderNodeTexImage")
        ao_tex.image = ao_img
        links.new(mapping.outputs["Vector"], ao_tex.inputs["Vector"])
        mix_ao = nodes.new(type="ShaderNodeMixRGB")
        mix_ao.blend_type = "MULTIPLY"
        mix_ao.inputs["Fac"].default_value = 0.6
        links.new(diff_tex.outputs["Color"], mix_ao.inputs["Color1"])
        links.new(ao_tex.outputs["Color"], mix_ao.inputs["Color2"])

        if value_mult != 1.0:
            # Distinguishes the room floor from the tabletop even though both
            # are wood -- without this the two read as one continuous stacked
            # surface when both are visible in the same close-up frame.
            hue_sat = nodes.new(type="ShaderNodeHueSaturation")
            hue_sat.inputs["Value"].default_value = value_mult
            links.new(mix_ao.outputs["Color"], hue_sat.inputs["Color"])
            links.new(hue_sat.outputs["Color"], bsdf.inputs["Base Color"])
        else:
            links.new(mix_ao.outputs["Color"], bsdf.inputs["Base Color"])
    except FileNotFoundError as e:
        print(f"[WARN] Texture not found for {asset_name}: {e}, using flat fallback color")

    return mat


def world_bbox(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    return Vector((min(xs), min(ys), min(zs))), Vector((max(xs), max(ys), max(zs)))


def bake_and_center(mesh_objs: list[bpy.types.Object], new_name: str) -> bpy.types.Object:
    """Bake every parent transform (including the glTF axis conversion) into
    each mesh's own vertex data, join multi-part assets into one object, then
    recenter the origin on the joined mesh's own bounding-box center so the
    object's `.location` matches PyBullet's box/sphere center-of-mass
    convention instead of whatever pivot the original asset happened to use.
    """
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


# assets/models/toy_car.glb (Gabriel Solon, CC-BY): a 10-part toy race car
# (body, grills, tires, rims, spoiler, skids, headlights, windshield,
# taillights) each parented under its own "<part>_mesh.ld" empty. Baking and
# joining all parts gives real-world dimensions directly (no extra scale
# factor needed, unlike the ball below): width(x) 0.1008m, length(y)
# 0.2337m, height(z) 0.0651m -- a plausible ~23cm plastic toy car. The
# model's nose (headlights/grille) already faces -Y and its tail (spoiler/
# taillights) faces +Y, matching this scene's -Y direction of travel with no
# extra rotation needed.
CAR_GLB_PATH = WORKSPACE_DIR / "assets" / "models" / "toy_car.glb"

# assets/models/pixar_ball.glb (Maggatron, CC-BY): a clean hand-modeled
# sphere authored in unitless local coordinates (raw radius exactly 1.0, not
# meters), so -- unlike the car -- an explicit scale factor is required. A
# 0.06m radius (12cm diameter) toy ball is roughly double the car's height,
# large enough for a dramatic collision but not so large the car only grazes
# its lower half.
BALL_GLB_PATH = WORKSPACE_DIR / "assets" / "models" / "pixar_ball.glb"
BALL_TARGET_RADIUS = 0.06

# assets/models/potted_plant.glb (propsworld.3d, CC-BY): a 4-part scanned
# succulent-in-a-pot, also authored at an arbitrary unitless scale (raw
# height 1.0, not meters) like the ball. Purely a static decoration -- not
# driven by physics -- so unlike the car/ball it's recentered on its own
# base rather than its volumetric center, so placing it by `.location` puts
# its pot flush on the shelf surface. Target height 0.3m (30cm) is a real
# small-houseplant scale for an actual room-scale shelf, deliberately
# larger than the toy car/ball next to it rather than matching their
# miniature scale.
PLANT_GLB_PATH = WORKSPACE_DIR / "assets" / "models" / "potted_plant.glb"
PLANT_TARGET_HEIGHT = 0.3


def import_toy_car_master() -> bpy.types.Object:
    mesh_objs, imported_names = import_glb_meshes(CAR_GLB_PATH)
    merged = bake_and_center(mesh_objs, "toy_car_master")
    for name in imported_names:
        obj = bpy.data.objects.get(name)
        if obj is not None and obj is not merged:
            bpy.data.objects.remove(obj, do_unlink=True)
    return merged


def import_toy_ball_master() -> bpy.types.Object:
    mesh_objs, imported_names = import_glb_meshes(BALL_GLB_PATH)
    merged = bake_and_center(mesh_objs, "toy_ball_master")

    bbox_min, bbox_max = world_bbox(merged)
    raw_radius = max((bbox_max - bbox_min)[axis] for axis in range(3)) / 2.0
    scale = BALL_TARGET_RADIUS / raw_radius
    merged.scale = (scale, scale, scale)
    bpy.ops.object.select_all(action="DESELECT")
    merged.select_set(True)
    bpy.context.view_layer.objects.active = merged
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    for name in imported_names:
        obj = bpy.data.objects.get(name)
        if obj is not None and obj is not merged:
            bpy.data.objects.remove(obj, do_unlink=True)
    return merged


def import_potted_plant_master() -> bpy.types.Object:
    mesh_objs, imported_names = import_glb_meshes(PLANT_GLB_PATH)
    merged = bake_and_center(mesh_objs, "potted_plant_master")

    bbox_min, bbox_max = world_bbox(merged)
    raw_height = (bbox_max - bbox_min).z
    scale = PLANT_TARGET_HEIGHT / raw_height
    merged.scale = (scale, scale, scale)
    bpy.ops.object.select_all(action="DESELECT")
    merged.select_set(True)
    bpy.context.view_layer.objects.active = merged
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    # Recenter on the base (not the volumetric center like the car/ball) so
    # placing it by `.location` sits the pot flush on the shelf surface.
    bbox_min, bbox_max = world_bbox(merged)
    bpy.context.scene.cursor.location = (0.0, 0.0, bbox_min.z)
    bpy.ops.object.origin_set(type="ORIGIN_CURSOR")
    merged.location = (0.0, 0.0, 0.0)

    for name in imported_names:
        obj = bpy.data.objects.get(name)
        if obj is not None and obj is not merged:
            bpy.data.objects.remove(obj, do_unlink=True)
    return merged


def place_object(
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
                "lens_mm": CAMERA_LENS_MM,
            },
            "physics": {
                "car_mass": 0.35,
                "car_friction": 0.22,
                "car_restitution": 0.1,
                "ball_mass": 0.05,
                "ball_friction": 0.3,
                "ball_restitution": 0.6,
                "table_friction": 0.2,
                "floor_friction": 0.8,
                "launch_speed": 0.6,
                "car_start_x": 0.1,
                "ball_start_x": -(TABLE_HALF_X) + BALL_RADIUS * 1.2,
                "gravity": [0.0, 0.0, -9.8],
                # Two-slot presence list (car, ball). A PCVE DELETE edit
                # writes 0 at the object's slot; only slot 1 (ball) is
                # actually removable in the sim -- the car drives the
                # whole scene so it is always kept.
                "active": [1, 1],
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

    script_path = Path(__file__).with_name("simulate_toy_car_ball.py")
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
            "--car-friction",
            str(float(physics["car_friction"])),
            "--car-restitution",
            str(float(physics["car_restitution"])),
            "--ball-mass",
            str(float(physics["ball_mass"])),
            "--ball-friction",
            str(float(physics["ball_friction"])),
            "--ball-restitution",
            str(float(physics["ball_restitution"])),
            "--table-friction",
            str(float(physics["table_friction"])),
            "--floor-friction",
            str(float(physics["floor_friction"])),
            "--launch-speed",
            str(float(physics["launch_speed"])),
            "--car-start-x",
            str(float(physics["car_start_x"])),
            "--ball-start-x",
            str(float(physics["ball_start_x"])),
            "--gravity-z",
            str(float(physics["gravity"][2])),
            "--ball-active",
            str(int(physics.get("active", [1, 1])[1])),
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
    car: bpy.types.Object, ball: bpy.types.Object, physics: dict,
) -> None:
    # A DELETE edit hid the ball in build_scene; skip keyframing it.
    animate_ball = not bool(ball.hide_render)
    car.rotation_mode = "QUATERNION"
    if animate_ball:
        ball.rotation_mode = "QUATERNION"
    for frame_record in physics["frames"]:
        frame = int(frame_record["frame_index"])
        pairs = [(car, "car")]
        if animate_ball:
            pairs.append((ball, "ball"))
        for obj, key in pairs:
            data = frame_record[key]
            quat = data["quaternion_xyzw"]
            obj.location = data["location"]
            obj.rotation_quaternion = (quat[3], quat[0], quat[1], quat[2])
            obj.keyframe_insert(data_path="location", frame=frame)
            obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)
    set_linear_keyframes([car] + ([ball] if animate_ball else []))


def export_ground_truth(
    out_dir: Path,
    car: bpy.types.Object,
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
            "car": {"object_name": car.name},
            "ball": {"present": not bool(ball.hide_render), "object_name": ball.name},
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
                "car": {
                    "matrix_world": [[float(v) for v in row] for row in car.matrix_world],
                    "location": [float(v) for v in car.location],
                    "linear_velocity": physics_frame["car"]["linear_velocity"],
                    "angular_velocity": physics_frame["car"]["angular_velocity"],
                },
                "ball": (
                    {"present": False}
                    if bool(ball.hide_render)
                    else {
                        "matrix_world": [[float(v) for v in row] for row in ball.matrix_world],
                        "location": [float(v) for v in ball.location],
                        "linear_velocity": physics_frame["ball"]["linear_velocity"],
                        "angular_velocity": physics_frame["ball"]["angular_velocity"],
                    }
                ),
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


def build_scene(
    args: argparse.Namespace, scenario: dict[str, object],
) -> tuple[bpy.types.Object, bpy.types.Object, bpy.types.Object]:
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

    # A plain, warm-toned studio-style world background -- the backdrop wall
    # geometry (built below) is what's actually visible behind the shelf in
    # frame; this only provides soft, warm fill/reflections instead of an
    # indoor-room HDRI, matching the plain physics-test-bench look of the
    # reference photo (lit for a cozy apartment feel) rather than a
    # neutral/cold studio.
    bg_node = world_nodes.new(type="ShaderNodeBackground")
    bg_node.inputs["Color"].default_value = (0.13, 0.09, 0.06, 1.0)
    bg_node.inputs["Strength"].default_value = 0.9
    output_node = world_nodes.new(type="ShaderNodeOutputWorld")
    world_links.new(bg_node.outputs["Background"], output_node.inputs["Surface"])

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

    bpy.ops.object.light_add(type="SUN", location=(1.0, -1.5, 2.5))
    sun = bpy.context.object
    sun.data.energy = 3.5
    sun.data.color = (1.0, 0.87, 0.68)
    sun.rotation_euler = (math.radians(50), math.radians(5), math.radians(35))

    add_area_light(
        "fill_light", location=(-1.2, -0.6, TABLE_Z + 0.6), power=90, size=1.4,
        target=LIGHT_TARGET, color=(1.0, 0.83, 0.62),
    )
    add_area_light(
        "rim_light", location=(1.2, 0.6, TABLE_Z + 0.5), power=55, size=1.2,
        target=LIGHT_TARGET, color=(1.0, 0.85, 0.68),
    )
    add_area_light(
        "wall_wash_light", location=(-0.7, 0.6, TABLE_Z + 1.2), power=60, size=2.0,
        target=(-0.7, TABLE_HALF_Y, TABLE_Z + 0.5), color=(1.0, 0.84, 0.65),
    )

    # Dark backdrop wall, flush behind the shelf's back edge (+y side) --
    # built first so the shelf and objects render in front of it.
    wall_mat = bpy.data.materials.new("backdrop_wall")
    wall_mat.use_nodes = True
    wall_bsdf = wall_mat.node_tree.nodes.get("Principled BSDF")
    if wall_bsdf is not None:
        set_input_default(wall_bsdf, "Base Color", (0.18, 0.115, 0.078, 1.0))
        set_input_default(wall_bsdf, "Roughness", 0.75)
        set_input_default(wall_bsdf, "Metallic", 0.0)
    wall_y = TABLE_HALF_Y + WALL_GAP + WALL_THICKNESS / 2.0
    bpy.ops.mesh.primitive_cube_add(
        size=1.0, location=(0.0, wall_y, WALL_HEIGHT / 2.0),
    )
    wall = bpy.context.object
    wall.name = "backdrop_wall"
    wall.dimensions = (2 * WALL_HALF_X, WALL_THICKNESS, WALL_HEIGHT)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    wall.data.materials.append(wall_mat)

    # Baseboard along the wall-floor junction, protruding slightly in front
    # of the wall face so it doesn't z-fight with it.
    baseboard_mat = bpy.data.materials.new("baseboard_white")
    baseboard_mat.use_nodes = True
    baseboard_bsdf = baseboard_mat.node_tree.nodes.get("Principled BSDF")
    if baseboard_bsdf is not None:
        set_input_default(baseboard_bsdf, "Base Color", (0.85, 0.81, 0.73, 1.0))
        set_input_default(baseboard_bsdf, "Roughness", 0.5)
        set_input_default(baseboard_bsdf, "Metallic", 0.0)
    wall_front_y = wall_y - WALL_THICKNESS / 2.0
    bpy.ops.mesh.primitive_cube_add(
        size=1.0,
        location=(0.0, wall_front_y - BASEBOARD_DEPTH / 2.0, BASEBOARD_HEIGHT / 2.0),
    )
    baseboard = bpy.context.object
    baseboard.name = "baseboard"
    baseboard.dimensions = (2 * WALL_HALF_X, BASEBOARD_DEPTH, BASEBOARD_HEIGHT)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    baseboard.data.materials.append(baseboard_mat)

    # White wall-mounted floating shelf (no legs -- it reads as attached to
    # the wall directly behind it, matching the reference photo).
    shelf_mat = bpy.data.materials.new("shelf_white")
    shelf_mat.use_nodes = True
    shelf_bsdf = shelf_mat.node_tree.nodes.get("Principled BSDF")
    if shelf_bsdf is not None:
        set_input_default(shelf_bsdf, "Base Color", (0.92, 0.87, 0.78, 1.0))
        set_input_default(shelf_bsdf, "Roughness", 0.4)
        set_input_default(shelf_bsdf, "Metallic", 0.0)
    bpy.ops.mesh.primitive_cube_add(
        size=1.0, location=(0.0, 0.0, TABLE_Z - TABLE_THICKNESS / 2.0),
    )
    table = bpy.context.object
    table.name = "shelf"
    table.dimensions = (2 * TABLE_HALF_X, 2 * TABLE_HALF_Y, TABLE_THICKNESS)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    table.data.materials.append(shelf_mat)

    floor_mat = create_pbr_material(
        "room_floor_wood", "wood_floor_worn", scale=10.0,
    )
    bpy.ops.mesh.primitive_cube_add(
        size=1.0, location=(0.0, 0.0, -FLOOR_THICKNESS / 2.0),
    )
    floor = bpy.context.object
    floor.name = "room_floor"
    floor.dimensions = (2 * FLOOR_HALF_X, 2 * FLOOR_HALF_Y, FLOOR_THICKNESS)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    floor.data.materials.append(floor_mat)

    physics = scenario["physics"]
    assert isinstance(physics, dict)

    car_master = import_toy_car_master()
    car = place_object(
        "toy_car",
        (float(physics["car_start_x"]), SHELF_DEPTH_Y, TABLE_Z + CAR_HALF_HEIGHT),
        car_master,
    )

    ball_master = import_toy_ball_master()
    ball = place_object(
        "toy_ball",
        (float(physics["ball_start_x"]), SHELF_DEPTH_Y, TABLE_Z + BALL_RADIUS),
        ball_master,
    )
    # DELETE edit: hide the ball from render + viewport. apply_physics_animation
    # reads hide_render and skips keyframing it.
    if not int(physics.get("active", [1, 1])[1]):
        ball.hide_viewport = True
        ball.hide_render = True

    # Static decoration, not physics-driven: a small potted plant to the
    # right of the car's start position, set back toward the wall like a
    # real shelf plant rather than out in the car's path.
    plant_master = import_potted_plant_master()
    place_object(
        "potted_plant",
        (float(physics["car_start_x"]) + 0.25, -0.05, TABLE_Z),
        plant_master,
    )

    frame_end = max(2, int(round(float(args.duration_sec) * int(args.fps))))
    scene.frame_start = 1
    scene.frame_end = frame_end

    return car, ball, camera


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

    car, ball, camera = build_scene(args, scenario)
    physics = run_physics_simulation(args, scenario)
    apply_physics_animation(car, ball, physics)
    export_ground_truth(
        out_dir, car, ball, camera, bpy.context.scene.frame_end, int(args.fps), physics, scenario,
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
