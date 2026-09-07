from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import bpy
import mathutils


WORKSPACE_DIR = Path(__file__).resolve().parents[2]
ASSETS_DIR = WORKSPACE_DIR / "assets"
POLYHAVEN_DIR = ASSETS_DIR / "polyhaven"
POOL_TABLE_PATH = ASSETS_DIR / "models" / "pool_table.glb"

OUTPUT_STEM = "pool_collision"
DIRECT_MP4_NAME = f"{OUTPUT_STEM}.mp4"
BLEND_NAME = f"{OUTPUT_STEM}.blend"
GROUND_TRUTH_NAME = "ground_truth_transforms.json"
PHYSICS_TEMP_NAME = "physics_transforms.json"
SCENARIO_METADATA_NAME = "scenario_metadata.json"

SCENE_SCALE = 1.0 / 3.0

GREEN_MESH_NAME = "green_pool_grass_text_0"
CUE_BALL_MESH_NAME = "pool_ball_16_pool_ball_white_text_0"
TARGET_BALL_MESH_NAME = "pool_ball_8_pool_ball_8_text_0"

CAMERA_LOCATION = (1.1, -1.85, 1.4)
CAMERA_TARGET_OFFSET = (0.0, 0.0, 0.01)
CAMERA_LENS_MM = 35.0


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
    parser.add_argument("--duration-sec", type=float, default=2.5)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--preview-frame", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cpu"), default="cpu")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--hdri-rotation", type=float, default=230.0)
    parser.add_argument("--scene-lower-z", type=float, default=0.25)
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


def load_image(path: Path, color_space: str) -> bpy.types.Image:
    name = f"{path.stem}_{color_space}"
    existing = bpy.data.images.get(name)
    if existing is not None and existing.filepath == str(path):
        return existing
    return bpy.data.images.load(str(path), check_existing=False)


def set_linear_keyframes(objects) -> None:
    for obj in objects:
        if obj.animation_data and obj.animation_data.action:
            for fcurve in obj.animation_data.action.fcurves:
                for key in fcurve.keyframe_points:
                    key.interpolation = "LINEAR"


def object_world_bounding_box_top(obj: bpy.types.Object) -> float:
    bpy.context.view_layer.update()
    return max((obj.matrix_world @ mathutils.Vector(c)).z for c in obj.bound_box)


def object_world_bounding_box_center(obj: bpy.types.Object) -> mathutils.Vector:
    bpy.context.view_layer.update()
    corners = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    return sum(corners, mathutils.Vector()) / len(corners)


def prepare_active_ball(obj: bpy.types.Object) -> None:
    """Detach ball from its parent and bake its world transform into the mesh."""
    mw = obj.matrix_world.copy()
    obj.parent = None
    obj.matrix_world = mw

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    obj.select_set(False)


def import_pool_table(scene_lower_z: float = 0.0) -> tuple[bpy.types.Object, bpy.types.Object, bpy.types.Object]:
    bpy.ops.wm.read_factory_settings(use_empty=True)

    for name in ("Cube", "Light", "Camera"):
        obj = bpy.data.objects.get(name)
        if obj is not None:
            bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.import_scene.gltf(filepath=str(POOL_TABLE_PATH))

    # Apply uniform scene scale and optional vertical offset to all root-level objects.
    for obj in bpy.data.objects:
        if obj.parent is None:
            obj.scale = (SCENE_SCALE, SCENE_SCALE, SCENE_SCALE)
            if scene_lower_z != 0.0:
                obj.location.z -= scene_lower_z

    bpy.context.view_layer.update()

    green_obj = bpy.data.objects.get(GREEN_MESH_NAME)
    cue_obj = bpy.data.objects.get(CUE_BALL_MESH_NAME)
    target_obj = bpy.data.objects.get(TARGET_BALL_MESH_NAME)

    if green_obj is None:
        raise RuntimeError(f"Green surface mesh not found: {GREEN_MESH_NAME}")
    if cue_obj is None:
        raise RuntimeError(f"Cue ball mesh not found: {CUE_BALL_MESH_NAME}")
    if target_obj is None:
        raise RuntimeError(f"Target ball mesh not found: {TARGET_BALL_MESH_NAME}")

    return green_obj, cue_obj, target_obj


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
                "target_offset": list(CAMERA_TARGET_OFFSET),
                "lens_mm": CAMERA_LENS_MM,
            },
            "physics": {
                "ball_radius": 0.05715,
                # Per-ball fields (the PCVE edit surface). Two identical
                # billiard balls at baseline; an edit names one of them and
                # writes at its slot. The globals (ball_*) below stay as
                # fallbacks for callers that do not care about per-ball
                # control -- if the per-ball value is None the sim uses the
                # global.
                "cue_mass": 0.17,
                "cue_friction": 0.15,
                "cue_restitution": 0.90,
                "cue_rolling_friction": 0.02,
                "cue_spinning_friction": 0.02,
                "target_mass": 0.17,
                "target_friction": 0.15,
                "target_restitution": 0.90,
                "target_rolling_friction": 0.02,
                "target_spinning_friction": 0.02,
                # Two-slot presence list, in fixed order (cue, target). A
                # PCVE DELETE edit writes 0 at the ball's slot.
                "active": [1, 1],
                "ball_mass": 0.17,
                "ball_friction": 0.15,
                "ball_restitution": 0.90,
                "ball_rolling_friction": 0.02,
                "ball_spinning_friction": 0.02,
                "table_friction": 0.08,
                "table_restitution": 0.10,
                "gravity": [0.0, 0.0, -9.81],
                "cue_initial_location": [0.0, -0.6, 0.0],
                "target_initial_location": [0.0, 0.0, 0.0],
                "cue_initial_velocity": [0.0, 1.0, 0.0],
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


def run_physics_simulation(
    args: argparse.Namespace,
    scenario: dict[str, object],
    surface_z: float,
    ball_radius: float,
) -> dict[str, Any]:
    # Prefer the project's conda environment where PyBullet is installed.
    physics_python_candidates = [
        WORKSPACE_DIR.parent / "miniconda3" / "envs" / "physics" / "bin" / "python",
        WORKSPACE_DIR.parent / "miniconda" / "envs" / "physics" / "bin" / "python",
        Path.home() / "miniconda3" / "envs" / "physics" / "bin" / "python",
        Path.home() / "miniconda" / "envs" / "physics" / "bin" / "python",
    ]
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        physics_python_candidates.append(Path(conda_prefix) / "bin" / "python")

    python = None
    for candidate in physics_python_candidates:
        if candidate.exists():
            python = str(candidate)
            break
    if python is None:
        python = shutil.which("python3") or shutil.which("python")
    if not python:
        raise RuntimeError("Cannot find python3/python for the PyBullet physics simulation.")

    physics = scenario["physics"]
    assert isinstance(physics, dict)

    script_path = Path(__file__).with_name("simulate_pool_collision.py")
    physics_path = args.out_dir / PHYSICS_TEMP_NAME

    def vec3(name: str) -> list[float]:
        value = physics.get(name)
        if isinstance(value, (list, tuple)) and len(value) == 3:
            return [float(v) for v in value]
        return [0.0, 0.0, 0.0]

    cue_loc = vec3("cue_initial_location")
    target_loc = vec3("target_initial_location")
    cue_vel = vec3("cue_initial_velocity")
    gravity = vec3("gravity")

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
            str(ball_radius),
            "--ball-mass",
            str(float(physics["ball_mass"])),
            "--ball-friction",
            str(float(physics["ball_friction"])),
            "--ball-restitution",
            str(float(physics["ball_restitution"])),
            "--ball-rolling-friction",
            str(float(physics["ball_rolling_friction"])),
            "--ball-spinning-friction",
            str(float(physics["ball_spinning_friction"])),
            "--table-friction",
            str(float(physics["table_friction"])),
            "--table-restitution",
            str(float(physics["table_restitution"])),
            "--gravity-z",
            str(gravity[2]),
            "--surface-z",
            str(surface_z),
            "--cue-x",
            str(cue_loc[0]),
            "--cue-y",
            str(cue_loc[1]),
            "--cue-z",
            str(cue_loc[2]),
            "--target-x",
            str(target_loc[0]),
            "--target-y",
            str(target_loc[1]),
            "--target-z",
            str(target_loc[2]),
            "--cue-vx",
            str(cue_vel[0]),
            "--cue-vy",
            str(cue_vel[1]),
            "--cue-vz",
            str(cue_vel[2]),
            "--cue-mass",           str(float(physics["cue_mass"])),
            "--cue-friction",       str(float(physics["cue_friction"])),
            "--cue-restitution",    str(float(physics["cue_restitution"])),
            "--cue-rolling-friction",   str(float(physics["cue_rolling_friction"])),
            "--cue-spinning-friction",  str(float(physics["cue_spinning_friction"])),
            "--target-mass",        str(float(physics["target_mass"])),
            "--target-friction",    str(float(physics["target_friction"])),
            "--target-restitution", str(float(physics["target_restitution"])),
            "--target-rolling-friction",  str(float(physics["target_rolling_friction"])),
            "--target-spinning-friction", str(float(physics["target_spinning_friction"])),
            "--cue-active",    str(int(physics["active"][0])),
            "--target-active", str(int(physics["active"][1])),
        ],
        check=True,
    )
    records = json.loads(physics_path.read_text(encoding="utf-8"))
    physics_path.unlink(missing_ok=True)
    return records


def setup_world_and_lights(scenario: dict[str, object]) -> None:
    scene = bpy.context.scene
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

    hdri_path = POLYHAVEN_DIR / "empty_room" / "small_empty_room_3_2k.hdr"
    if hdri_path.exists():
        hdri_img = load_image(hdri_path, "Non-Color")
        hdri_img.colorspace_settings.name = "Linear"
        env_tex.image = hdri_img
        print(f"[INFO] Using HDRI background: {hdri_path.name}")
    else:
        print(f"[WARN] HDRI not found at {hdri_path}, using solid fallback")
        bg_node = world_nodes.new(type="ShaderNodeBackground")
        bg_node.inputs["Color"].default_value = (0.30, 0.35, 0.40, 1.0)
        bg_node.inputs["Strength"].default_value = 1.0
        env_tex = bg_node

    hdri_cfg = scenario.get("hdri", {})
    rotation_z = float(hdri_cfg.get("rotation_z", 0.0))

    mapping_node = world_nodes.new(type="ShaderNodeMapping")
    mapping_node.location = (-550, 0)
    mapping_node.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(rotation_z))

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

    camera_cfg = scenario["camera"]
    assert isinstance(camera_cfg, dict)
    camera_location = tuple(camera_cfg.get("location", CAMERA_LOCATION))
    target_offset = tuple(camera_cfg.get("target_offset", CAMERA_TARGET_OFFSET))

    bpy.ops.object.camera_add(location=camera_location)
    camera = bpy.context.object
    camera.name = "render_camera"
    camera.data.lens = float(camera_cfg.get("lens_mm", CAMERA_LENS_MM))
    scene.camera = camera

    target = bpy.data.objects.new("camera_target", None)
    scene.collection.objects.link(target)
    target.location = target_offset
    constraint = camera.constraints.new("TRACK_TO")
    constraint.target = target
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"

    bpy.ops.object.light_add(type="SUN", location=(2.0, -2.0, 4.0))
    sun = bpy.context.object
    sun.data.energy = 0.5
    sun.rotation_euler = (math.radians(45), math.radians(15), math.radians(30))

    bpy.ops.object.light_add(type="AREA", location=(-1.5, 1.0, 2.5))
    fill_light = bpy.context.object
    fill_light.data.energy = 150
    fill_light.data.size = 3.0

    bpy.ops.object.light_add(type="AREA", location=(0.5, -0.5, 2.0))
    rim_light = bpy.context.object
    rim_light.data.energy = 80
    rim_light.data.size = 2.0


def build_scene(
    args: argparse.Namespace,
    scenario: dict[str, object],
) -> tuple[bpy.types.Object, bpy.types.Object, bpy.types.Object]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_lower_z = float(scenario.get("scene_lower_z", 0.0))
    green_obj, cue_obj, target_obj = import_pool_table(scene_lower_z)

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

    setup_world_and_lights(scenario)

    prepare_active_ball(cue_obj)
    prepare_active_ball(target_obj)

    # Hide the remaining balls and the cue sticks so they do not obstruct the shot.
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        parent_name = obj.parent.name if obj.parent else ""
        is_other_ball = obj.name.startswith("pool_ball_") and obj not in (cue_obj, target_obj)
        is_cue_stick = "_stick_" in obj.name.lower() or "pool_stick" in obj.name.lower()
        if is_other_ball or is_cue_stick:
            obj.hide_viewport = True
            obj.hide_render = True

    # Hide small decorative rail markers (diamonds and copper tacks) that read as
    # scattered yellow/brown dots in the render.
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        is_diamond = (
            obj.name.startswith("pSphere")
            and "pool_diamond" in obj.name.lower()
        )
        is_small_copper = (
            max(obj.dimensions) < 0.05
            and any("copper_text" in (m.name if m else "") for m in obj.data.materials)
        )
        if is_diamond or is_small_copper:
            obj.hide_viewport = True
            obj.hide_render = True

    surface_z = object_world_bounding_box_top(green_obj)
    ball_radius = max(cue_obj.dimensions) / 2.0
    print(f"[INFO] Pool table surface_z={surface_z:.4f}, ball_radius={ball_radius:.4f}")

    # Aim the camera at the table surface, not the floor.
    camera_target = bpy.data.objects.get("camera_target")
    if camera_target is not None:
        camera_target.location = (0.0, 0.0, surface_z + 0.01)

    physics = scenario["physics"]
    assert isinstance(physics, dict)

    cue_loc = physics.get("cue_initial_location", [0.0, -0.6, 0.0])
    target_loc = physics.get("target_initial_location", [0.0, 0.0, 0.0])

    cue_obj.location = (float(cue_loc[0]), float(cue_loc[1]), surface_z + ball_radius + float(cue_loc[2]))
    target_obj.location = (float(target_loc[0]), float(target_loc[1]), surface_z + ball_radius + float(target_loc[2]))

    # DELETE edit: hide the removed ball from render and viewport.
    active = physics.get("active", [1, 1])
    if not int(active[0]):
        cue_obj.hide_viewport = True
        cue_obj.hide_render = True
    if not int(active[1]):
        target_obj.hide_viewport = True
        target_obj.hide_render = True

    # Update scenario with the actual values used for physics and rendering.
    physics["ball_radius"] = ball_radius
    physics["surface_z"] = surface_z

    frame_end = max(2, int(round(float(args.duration_sec) * int(args.fps))))
    scene.frame_start = 1
    scene.frame_end = frame_end

    camera = bpy.data.objects.get("render_camera")
    if camera is None:
        raise RuntimeError("Render camera was not created")

    return cue_obj, target_obj, camera


def apply_physics_animation(
    cue_ball: bpy.types.Object,
    target_ball: bpy.types.Object,
    physics: dict,
) -> None:
    # A hidden ball (DELETE edit) does not need keyframes -- its frozen
    # position was already set in build_scene. Skip animating it.
    animate_cue = not bool(cue_ball.hide_render)
    animate_target = not bool(target_ball.hide_render)

    if animate_cue:
        cue_ball.rotation_mode = "QUATERNION"
    if animate_target:
        target_ball.rotation_mode = "QUATERNION"

    for frame_record in physics["frames"]:
        frame = int(frame_record["frame_index"])

        if animate_cue:
            cue_quat = frame_record["cue_ball_quaternion_xyzw"]
            cue_ball.location = frame_record["cue_ball_location"]
            cue_ball.rotation_quaternion = (
                cue_quat[3],
                cue_quat[0],
                cue_quat[1],
                cue_quat[2],
            )
            cue_ball.keyframe_insert(data_path="location", frame=frame)
            cue_ball.keyframe_insert(data_path="rotation_quaternion", frame=frame)

        if animate_target:
            target_quat = frame_record["target_ball_quaternion_xyzw"]
            target_ball.location = frame_record["target_ball_location"]
            target_ball.rotation_quaternion = (
                target_quat[3],
                target_quat[0],
                target_quat[1],
                target_quat[2],
            )
            target_ball.keyframe_insert(data_path="location", frame=frame)
            target_ball.keyframe_insert(data_path="rotation_quaternion", frame=frame)

    animated = []
    if animate_cue:
        animated.append(cue_ball)
    if animate_target:
        animated.append(target_ball)
    set_linear_keyframes(animated)


def export_ground_truth(
    out_dir: Path,
    cue_ball: bpy.types.Object,
    target_ball: bpy.types.Object,
    camera: bpy.types.Object,
    frame_end: int,
    fps: int,
    physics: dict,
    scenario: dict[str, object],
) -> None:
    scene = bpy.context.scene
    physics_info = scenario["physics"]
    assert isinstance(physics_info, dict)
    ball_radius = float(physics_info["ball_radius"])

    records = {
        "schema_version": 1,
        "fps": int(fps),
        "frame_start": 1,
        "frame_end": int(frame_end),
        "scenario_metadata_path": str(output_path(out_dir, SCENARIO_METADATA_NAME)),
        "physics": {key: value for key, value in physics.items() if key != "frames"},
        "objects": {
            "cue_ball": {
                "present": not bool(cue_ball.hide_render),
                "object_name": cue_ball.name,
                "radius_m_scene_units": ball_radius,
            },
            "target_ball": {
                "present": not bool(target_ball.hide_render),
                "object_name": target_ball.name,
                "radius_m_scene_units": ball_radius,
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
                "cue_ball_matrix_world": [[float(v) for v in row] for row in cue_ball.matrix_world],
                "cue_ball_location": [float(v) for v in cue_ball.location],
                "cue_ball_linear_velocity": physics_frame["cue_ball_linear_velocity"],
                "cue_ball_angular_velocity": physics_frame["cue_ball_angular_velocity"],
                "cue_ball_table_gap": physics_frame["cue_ball_table_gap"],
                "target_ball_matrix_world": [[float(v) for v in row] for row in target_ball.matrix_world],
                "target_ball_location": [float(v) for v in target_ball.location],
                "target_ball_linear_velocity": physics_frame["target_ball_linear_velocity"],
                "target_ball_angular_velocity": physics_frame["target_ball_angular_velocity"],
                "target_ball_table_gap": physics_frame["target_ball_table_gap"],
                "ball_ball_gap": physics_frame["ball_ball_gap"],
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
    scenario.setdefault("hdri", {})["rotation_z"] = float(args.hdri_rotation)
    scenario["scene_lower_z"] = float(args.scene_lower_z)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_scenario_metadata(out_dir, scenario)

    cue_ball, target_ball, camera = build_scene(args, scenario)
    physics = run_physics_simulation(args, scenario, scenario["physics"]["surface_z"], scenario["physics"]["ball_radius"])
    apply_physics_animation(cue_ball, target_ball, physics)
    export_ground_truth(
        out_dir,
        cue_ball,
        target_ball,
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
