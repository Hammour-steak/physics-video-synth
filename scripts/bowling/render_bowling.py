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
GLB_PATH = WORKSPACE_DIR / "assets" / "models" / "bowling_club.glb"

OUTPUT_STEM = "bowling"
DIRECT_MP4_NAME = f"{OUTPUT_STEM}.mp4"
BLEND_NAME = f"{OUTPUT_STEM}.blend"
GROUND_TRUTH_NAME = "ground_truth_transforms.json"
PHYSICS_TEMP_NAME = "physics_transforms.json"
SCENARIO_METADATA_NAME = "scenario_metadata.json"

FLOOR_SIZE = 10.0
FLOOR_THICKNESS = 0.1
FLOOR_Z = 0.05
BALL_RADIUS = 0.12
PIN_RADIUS = 0.075
PIN_HEIGHT = 0.495

# Camera at the bowler end (x≈28) looking down the lane toward the pin deck
# at x≈-5 where the scoreboard and open white door are.
CAMERA_LOCATION = (12.0, 26.3, 1.5)
CAMERA_TARGET = (-5.0, 26.3, 0.5)
LIGHT_TARGET = (-5.0, 26.3, 0.5)


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
    parser.add_argument("--preview-frame", type=int, default=16)
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
            noise = nodes.new(type="ShaderNodeTexNoise")
            noise.inputs["Scale"].default_value = 55.0
            noise.inputs["Detail"].default_value = 9.0
            noise.inputs["Roughness"].default_value = 0.62
            bump = nodes.new(type="ShaderNodeBump")
            bump.inputs["Strength"].default_value = float(noise_bump)
            bump.inputs["Distance"].default_value = 0.055
            links = mat.node_tree.links
            links.new(noise.outputs["Fac"], bump.inputs["Height"])
            links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def create_wall_material(scenario: dict[str, object]) -> bpy.types.Material:
    return create_principled_material(
        "slightly uneven painted plaster",
        (0.70, 0.66, 0.60, 1.0),
        roughness=0.88,
        noise_bump=0.014,
    )


def create_baseboard_material(scenario: dict[str, object]) -> bpy.types.Material:
    return create_principled_material(
        "painted off white baseboard",
        (0.78, 0.74, 0.68, 1.0),
        roughness=0.66,
        noise_bump=0.006,
    )


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


def add_room_shell(scenario: dict[str, object]) -> None:
    wall_mat = create_wall_material(scenario)
    baseboard_mat = create_baseboard_material(scenario)
    add_box("matte back plaster wall", (0.0, 3.05, 1.45), (8.6, 0.08, 2.90), wall_mat)
    add_box("matte right plaster wall", (4.20, -0.55, 1.45), (0.08, 7.2, 2.90), wall_mat)
    add_box("back wall baseboard", (0.0, 2.995, 0.13), (8.5, 0.09, 0.16), baseboard_mat, bevel_width=0.006)
    add_box("right wall baseboard", (4.145, -0.55, 0.13), (0.09, 7.1, 0.16), baseboard_mat, bevel_width=0.006)


def import_bowling_assets() -> tuple[bpy.types.Object, list[bpy.types.Object]]:
    """Import the club GLB, keep one ball plus three pins, and keep the rest of
    the club as a static background.  All other decorative balls/pins are
    removed so they do not collide visually with the animated ones."""
    if not GLB_PATH.exists():
        raise FileNotFoundError(f"Bowling club model not found: {GLB_PATH}")

    existing = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(GLB_PATH))
    imported = [obj for obj in bpy.context.scene.objects if obj not in existing]
    imported_names = [obj.name for obj in imported]
    print(f"[INFO] Imported {len(imported)} objects from {GLB_PATH.name}")

    ball_candidates = [
        obj for obj in imported
        if obj.type == "MESH" and obj.name.startswith("Bowling_ball")
    ]
    if not ball_candidates:
        raise RuntimeError(f"No bowling ball mesh found in {GLB_PATH}")
    ball = ball_candidates[0]

    pin_objs = [
        obj for obj in imported
        if obj.type == "MESH" and obj.name.startswith("Bowling_pin")
    ]
    prefix_map: dict[str, list[bpy.types.Object]] = {}
    for obj in pin_objs:
        # Name format: "Bowling_pin001_Material #4304_0"
        parts = obj.name.split("_")
        prefix = f"{parts[0]}_{parts[1]}"
        prefix_map.setdefault(prefix, []).append(obj)

    if len(prefix_map) < 3:
        raise RuntimeError(f"Need at least 3 bowling pins, found {len(prefix_map)}")

    # Detach kept meshes from the imported hierarchy so deleting the parent
    # empties does not move them.
    selected_prefixes = sorted(prefix_map.keys())[:3]
    kept_parts = {ball}
    for prefix in selected_prefixes:
        kept_parts.update(prefix_map[prefix])
    bpy.ops.object.select_all(action="DESELECT")
    for obj in kept_parts:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = ball
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")

    # Join each pin's material parts into one mesh.
    # Apply transforms to each part first so the joined result has consistent scale.
    pins: list[bpy.types.Object] = []
    for idx, prefix in enumerate(selected_prefixes):
        parts = prefix_map[prefix]
        for part in parts:
            bpy.context.view_layer.objects.active = part
            bpy.ops.object.select_all(action="DESELECT")
            part.select_set(True)
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        bpy.ops.object.select_all(action="DESELECT")
        for part in parts:
            part.select_set(True)
        bpy.context.view_layer.objects.active = parts[0]
        bpy.ops.object.join()
        pin = bpy.context.active_object
        pin.name = f"bowling_pin_{idx}"
        pins.append(pin)

    # Remove every other decorative bowling ball or pin.  Keep the rest of the
    # club (lanes, walls, ceiling, ball stands, screens, etc.) as background.
    kept_names = {ball.name} | {pin.name for pin in pins}
    for obj_name in imported_names:
        if obj_name in kept_names:
            continue
        obj = bpy.data.objects.get(obj_name)
        if obj is None:
            continue
        if obj.type == "MESH" and (
            obj.name.startswith("Bowling_ball") or obj.name.startswith("Bowling_pin")
        ):
            bpy.data.objects.remove(obj, do_unlink=True)

    # Center origins and reset transforms so object coordinates match the
    # physics simulation (sphere/pin cylinder centers).
    ball.name = "bowling_ball"
    for obj in [ball, *pins]:
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        bpy.ops.object.origin_set(type="GEOMETRY_ORIGIN", center="BOUNDS")
        obj.location = (0.0, 0.0, 0.0)
        obj.rotation_euler = (0.0, 0.0, 0.0)
        obj.scale = (1.0, 1.0, 1.0)

    return ball, pins


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
            },
            "camera": {
                "location": list(CAMERA_LOCATION),
                "target": list(CAMERA_TARGET),
                "lens_mm": 35.0,
            },
            "physics": {
                "ball_radius": BALL_RADIUS,
                "ball_mass": 3.0,
                "ball_friction": 0.4,
                "ball_restitution": 0.5,
                "pin_radius": PIN_RADIUS,
                "pin_height": PIN_HEIGHT,
                "pin_mass": 0.8,
                "pin_friction": 0.4,
                "pin_restitution": 0.3,
                "floor_friction": 0.5,
                "floor_z": FLOOR_Z,
                "ball_initial_location": [10.0, 0.0, 0.18],
                "ball_initial_velocity": [-8.0, 0.0, 0.0],
                "pin_spacing": 0.28,
                "gravity": [0.0, 0.0, -9.81],
                "scene_offset_x": -5.0,
                "scene_offset_y": 26.3,
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
    # Prefer the project's physics conda environment, which has PyBullet installed.
    physics_python = WORKSPACE_DIR.parent / "miniconda" / "envs" / "physics" / "bin" / "python"
    if physics_python.exists():
        python = str(physics_python)
    else:
        python = shutil.which("python3") or shutil.which("python")
    if not python:
        raise RuntimeError("Cannot find python3/python for the PyBullet physics simulation.")

    physics = scenario["physics"]
    assert isinstance(physics, dict)

    script_path = Path(__file__).with_name("simulate_bowling.py")
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
            "--ball-friction",
            str(float(physics["ball_friction"])),
            "--ball-restitution",
            str(float(physics["ball_restitution"])),
            "--pin-radius",
            str(float(physics["pin_radius"])),
            "--pin-height",
            str(float(physics["pin_height"])),
            "--pin-mass",
            str(float(physics["pin_mass"])),
            "--pin-friction",
            str(float(physics["pin_friction"])),
            "--pin-restitution",
            str(float(physics["pin_restitution"])),
            "--floor-friction",
            str(float(physics["floor_friction"])),
            "--floor-z",
            str(float(physics["floor_z"])),
            "--scene-offset-x",
            str(float(physics["scene_offset_x"])),
            "--scene-offset-y",
            str(float(physics["scene_offset_y"])),
            "--ball-initial-location",
            *[str(float(v)) for v in physics["ball_initial_location"]],
            "--ball-initial-velocity",
            *[str(float(v)) for v in physics["ball_initial_velocity"]],
            "--pin-spacing",
            str(float(physics["pin_spacing"])),
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
    pins: list[bpy.types.Object],
    physics: dict,
) -> None:
    for obj in [ball, *pins]:
        obj.rotation_mode = "QUATERNION"

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

        for idx, pin_obj in enumerate(pins):
            pin_data = frame_record["pins"][idx]
            pquat = pin_data["quaternion_xyzw"]
            pin_obj.location = pin_data["location"]
            pin_obj.rotation_quaternion = (
                pquat[3],
                pquat[0],
                pquat[1],
                pquat[2],
            )
            pin_obj.keyframe_insert(data_path="location", frame=frame)
            pin_obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)

    set_linear_keyframes([ball, *pins])


def export_ground_truth(
    out_dir: Path,
    ball: bpy.types.Object,
    pins: list[bpy.types.Object],
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
                "radius_m_scene_units": BALL_RADIUS,
            },
            "pins": [
                {
                    "object_name": pin.name,
                    "radius_m_scene_units": PIN_RADIUS,
                    "height_m_scene_units": PIN_HEIGHT,
                    "index": idx,
                }
                for idx, pin in enumerate(pins)
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
                "pins": [
                    {
                        "matrix_world": [[float(v) for v in row] for row in pin.matrix_world],
                        "location": [float(v) for v in pin.location],
                        "linear_velocity": pin_data["linear_velocity"],
                        "angular_velocity": pin_data["angular_velocity"],
                        "gap_to_ball": pin_data["gap_to_ball"],
                    }
                    for pin, pin_data in zip(pins, physics_frame["pins"])
                ],
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


def build_scene(args: argparse.Namespace, scenario: dict[str, object]) -> tuple[bpy.types.Object, list[bpy.types.Object], bpy.types.Object]:
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

    env_tex = world_nodes.new(type="ShaderNodeTexEnvironment")
    env_tex.location = (-300, 0)

    hdri_path = POLYHAVEN_DIR / "brown_photostudio_05" / "brown_photostudio_05_2k.hdr"
    if hdri_path.exists():
        hdri_img = bpy.data.images.load(str(hdri_path), check_existing=True)
        env_tex.image = hdri_img
        print(f"[INFO] Using HDRI background: brown_photostudio_05")
    else:
        print(f"[WARN] HDRI not found at {hdri_path}, using solid fallback")
        bg_node = world_nodes.new(type="ShaderNodeBackground")
        bg_node.inputs["Color"].default_value = (0.35, 0.32, 0.30, 1.0)
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
    camera.data.lens = 35
    scene.camera = camera

    target = bpy.data.objects.new("camera_target", None)
    scene.collection.objects.link(target)
    target.location = CAMERA_TARGET
    constraint = camera.constraints.new("TRACK_TO")
    constraint.target = target
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"

    bpy.ops.object.light_add(type="SUN", location=(15.0, 27.0, 5.0))
    sun = bpy.context.object
    sun.data.energy = 0.5
    sun.rotation_euler = (math.radians(45), math.radians(15), math.radians(30))

    fill_light = add_area_light(
        "fill_light",
        location=(10.0, 26.3, 2.5),
        power=55,
        size=2.5,
        target=LIGHT_TARGET,
    )
    rim_light = add_area_light(
        "rim_light",
        location=(12.0, 24.5, 2.0),
        power=45,
        size=2.0,
        target=LIGHT_TARGET,
    )

    ball, pins = import_bowling_assets()

    # Position visual objects at their physics initial locations so the first
    # keyframe does not cause a jump.
    physics = scenario["physics"]
    assert isinstance(physics, dict)
    offset_x = float(physics.get("scene_offset_x", 0.0))
    offset_y = float(physics.get("scene_offset_y", 0.0))
    floor_z = float(physics.get("floor_z", FLOOR_Z))

    ball_init = tuple(float(v) for v in physics["ball_initial_location"])
    ball.location = (ball_init[0] + offset_x, ball_init[1] + offset_y, ball_init[2] + floor_z)
    ball.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)

    pin_spacing = float(physics["pin_spacing"])
    pin_base_positions = [
        (0.0, 0.0),
        (pin_spacing * 0.866, pin_spacing * 0.5),
        (pin_spacing * 0.866, -pin_spacing * 0.5),
    ]
    for pin, (x, y) in zip(pins, pin_base_positions):
        pin.location = (x + offset_x, y + offset_y, floor_z + PIN_HEIGHT / 2.0)
        pin.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)

    frame_end = max(2, int(round(float(args.duration_sec) * int(args.fps))))
    scene.frame_start = 1
    scene.frame_end = frame_end

    return ball, pins, camera


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

    ball, pins, camera = build_scene(args, scenario)
    physics = run_physics_simulation(args, scenario)
    apply_physics_animation(ball, pins, physics)
    export_ground_truth(
        out_dir,
        ball,
        pins,
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

    bpy.ops.wm.save_as_mainfile(filepath=str(output_path(out_dir, BLEND_NAME)))
    print(f"[INFO] Render complete. Output: {out_dir}")


if __name__ == "__main__":
    main()
