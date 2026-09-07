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
TIMED_EDITS_TEMP_NAME = "timed_edits.json"
SCENARIO_METADATA_NAME = "scenario_metadata.json"

SCENE_SCALE = 1.0 / 3.0

GREEN_MESH_NAME = "green_pool_grass_text_0"
CUE_BALL_MESH_NAME = "pool_ball_16_pool_ball_white_text_0"
TARGET_BALL_MESH_NAME = "pool_ball_8_pool_ball_8_text_0"
# The ball a PCVE ADD edit puts on the felt. The table model ships the whole
# rack and this renderer hides the balls the shot does not use, so "adding" a
# ball is un-hiding one that was always in the file -- it arrives with its own
# number and colour, and reads at a glance against both the white cue and the
# black eight.
EXTRA_BALL_MESH_NAME = "pool_ball_1_pool_ball_1_text_0"

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


def import_pool_table(
    scene_lower_z: float = 0.0,
) -> tuple[bpy.types.Object, bpy.types.Object, bpy.types.Object, bpy.types.Object]:
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
    extra_obj = bpy.data.objects.get(EXTRA_BALL_MESH_NAME)

    if green_obj is None:
        raise RuntimeError(f"Green surface mesh not found: {GREEN_MESH_NAME}")
    if cue_obj is None:
        raise RuntimeError(f"Cue ball mesh not found: {CUE_BALL_MESH_NAME}")
    if target_obj is None:
        raise RuntimeError(f"Target ball mesh not found: {TARGET_BALL_MESH_NAME}")
    if extra_obj is None:
        raise RuntimeError(f"Extra ball mesh not found: {EXTRA_BALL_MESH_NAME}")

    return green_obj, cue_obj, target_obj, extra_obj


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
                # Measured off the table model, and the same number the
                # renderer writes back over this key in build_scene before the
                # simulator ever sees it -- a regulation 57.15 mm ball, so
                # 28.8 mm of radius. It read 0.05715 here until the ADD edits
                # went in: harmless while nothing consumed the placeholder,
                # but the PCVE geometry anchors turn "3 radii" into metres
                # with it, and at the ball's diameter every ADD would have
                # landed at twice the distance its prompt claims.
                "ball_radius": 0.0288317501544952,
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
                # The yellow one-ball: absent from the baseline shot, so its
                # physics sits here unused until a PCVE ADD edit turns slot 2
                # of `active` on and overwrites its location.
                "extra_mass": 0.17,
                "extra_friction": 0.15,
                "extra_restitution": 0.90,
                "extra_rolling_friction": 0.02,
                "extra_spinning_friction": 0.02,
                "extra_initial_location": [0.0, -0.3, 0.0],
                # Three-slot presence list, in fixed order (cue, target,
                # yellow). A PCVE DELETE edit writes 0 at a ball's slot; an
                # ADD edit writes 1 at the yellow ball's, which is the one
                # slot that reads 0 at baseline.
                "active": [1, 1, 0],
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
    # Edits that land partway through the clip travel to the simulator as a
    # file rather than as flags: each entry is a whole parameter dict, and the
    # sim applies it at the top of its frame.
    timed_edits = physics.get("timed_edits") or []
    timed_edits_path = args.out_dir / TIMED_EDITS_TEMP_NAME
    if timed_edits:
        timed_edits_path.parent.mkdir(parents=True, exist_ok=True)
        timed_edits_path.write_text(json.dumps(timed_edits, indent=2), encoding="utf-8")

    def vec3(name: str) -> list[float]:
        value = physics.get(name)
        if isinstance(value, (list, tuple)) and len(value) == 3:
            return [float(v) for v in value]
        return [0.0, 0.0, 0.0]

    cue_loc = vec3("cue_initial_location")
    target_loc = vec3("target_initial_location")
    extra_loc = vec3("extra_initial_location")
    cue_vel = vec3("cue_initial_velocity")
    gravity = vec3("gravity")
    # Scenarios written before the yellow ball existed carry a two-slot
    # `active`; read the third slot defensively so those still replay.
    active = list(physics.get("active", [1, 1, 0]))
    while len(active) < 3:
        active.append(0)

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
            "--extra-mass",         str(float(physics.get("extra_mass", physics["ball_mass"]))),
            "--extra-friction",     str(float(physics.get("extra_friction", physics["ball_friction"]))),
            "--extra-restitution",  str(float(physics.get("extra_restitution", physics["ball_restitution"]))),
            "--extra-rolling-friction",
            str(float(physics.get("extra_rolling_friction", physics["ball_rolling_friction"]))),
            "--extra-spinning-friction",
            str(float(physics.get("extra_spinning_friction", physics["ball_spinning_friction"]))),
            "--extra-x", str(extra_loc[0]),
            "--extra-y", str(extra_loc[1]),
            "--extra-z", str(extra_loc[2]),
            "--cue-active",    str(int(active[0])),
            "--target-active", str(int(active[1])),
            "--extra-active",  str(int(active[2])),
        ]
        + (["--timed-edits-json", str(timed_edits_path)] if timed_edits else []),
        check=True,
    )
    records = json.loads(physics_path.read_text(encoding="utf-8"))
    physics_path.unlink(missing_ok=True)
    timed_edits_path.unlink(missing_ok=True)
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
) -> tuple[bpy.types.Object, bpy.types.Object, bpy.types.Object, bpy.types.Object]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_lower_z = float(scenario.get("scene_lower_z", 0.0))
    green_obj, cue_obj, target_obj, extra_obj = import_pool_table(scene_lower_z)

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
    prepare_active_ball(extra_obj)

    # Hide the remaining balls and the cue sticks so they do not obstruct the
    # shot. The yellow ball is spared here and hidden below only if the
    # scenario leaves its slot off, which is what makes it "addable".
    balls_in_play = (cue_obj, target_obj, extra_obj)
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        parent_name = obj.parent.name if obj.parent else ""
        is_other_ball = obj.name.startswith("pool_ball_") and obj not in balls_in_play
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
    extra_loc = physics.get("extra_initial_location", [0.0, -0.3, 0.0])

    cue_obj.location = (float(cue_loc[0]), float(cue_loc[1]), surface_z + ball_radius + float(cue_loc[2]))
    target_obj.location = (float(target_loc[0]), float(target_loc[1]), surface_z + ball_radius + float(target_loc[2]))
    extra_obj.location = (float(extra_loc[0]), float(extra_loc[1]), surface_z + ball_radius + float(extra_loc[2]))

    # Presence: a DELETE edit hides the ball it removed, and the yellow ball
    # is hidden unless an ADD edit turned its slot on. Same mechanism read
    # from the same list.
    active = list(physics.get("active", [1, 1, 0]))
    while len(active) < 3:
        active.append(0)
    for obj, slot in zip((cue_obj, target_obj, extra_obj), active):
        if not int(slot):
            obj.hide_viewport = True
            obj.hide_render = True

    # Update scenario with the actual values used for physics and rendering.
    physics["ball_radius"] = ball_radius
    physics["surface_z"] = surface_z

    frame_end = max(2, int(round(float(args.duration_sec) * int(args.fps))))
    scene.frame_start = 1
    scene.frame_end = frame_end

    camera = bpy.data.objects.get("render_camera")
    if camera is None:
        raise RuntimeError("Render camera was not created")

    return cue_obj, target_obj, extra_obj, camera


def apply_physics_animation(
    balls: "list[tuple[bpy.types.Object, str]]",
    physics: dict,
) -> None:
    """Keyframe each ball from its per-frame sim record.

    ``balls`` pairs each Blender object with the prefix the simulator writes
    it under (``cue_ball``, ``target_ball``, ``extra_ball``). A hidden ball --
    one a DELETE edit removed, or the yellow ball when no ADD edit placed it --
    needs no keyframes: build_scene already parked it at its frozen position.
    """
    animated = []
    for obj, prefix in balls:
        if obj.hide_render:
            continue
        obj.rotation_mode = "QUATERNION"
        animated.append((obj, prefix))

    for frame_record in physics["frames"]:
        frame = int(frame_record["frame_index"])
        for obj, prefix in animated:
            quat = frame_record[f"{prefix}_quaternion_xyzw"]
            obj.location = frame_record[f"{prefix}_location"]
            # Blender wants w first, the simulator hands over w last.
            obj.rotation_quaternion = (quat[3], quat[0], quat[1], quat[2])
            obj.keyframe_insert(data_path="location", frame=frame)
            obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)

    set_linear_keyframes([obj for obj, _ in animated])
    apply_disappearances(animated, physics)


def removal_frame(physics: dict, prefix: str) -> int | None:
    """The first frame ``prefix`` is absent on, if it starts out present."""
    frames = physics["frames"]
    if not frames or not frames[0][f"{prefix}_present"]:
        return None
    return next((int(f["frame_index"]) for f in frames
                 if not f[f"{prefix}_present"]), None)


def apply_disappearances(
    animated: "list[tuple[bpy.types.Object, str]]", physics: dict
) -> None:
    """Make a ball the simulation removed mid-run leave the picture.

    A whole-clip delete parks the ball hidden before any keyframe is written;
    this is the other kind, where it rolls through the frames it has in the
    source and then is gone. CONSTANT interpolation so it vanishes between two
    frames rather than fading across them.
    """
    for obj, prefix in animated:
        gone_at = removal_frame(physics, prefix)
        if gone_at is None or gone_at <= 1:
            continue
        for path in ("hide_viewport", "hide_render"):
            setattr(obj, path, False)
            obj.keyframe_insert(data_path=path, frame=gone_at - 1)
            setattr(obj, path, True)
            obj.keyframe_insert(data_path=path, frame=gone_at)
        for fcurve in obj.animation_data.action.fcurves:
            if fcurve.data_path in ("hide_viewport", "hide_render"):
                for key in fcurve.keyframe_points:
                    key.interpolation = "CONSTANT"


def export_ground_truth(
    out_dir: Path,
    balls: "list[tuple[bpy.types.Object, str]]",
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
            prefix: {
                "present": bool(physics["frames"][0][f"{prefix}_present"]),
                "object_name": obj.name,
                "radius_m_scene_units": ball_radius,
                # The frame it stops being on screen, for a ball a timed edit
                # takes away partway through; None when it is there for the
                # whole clip (or was never there at all).
                "removed_at_frame": removal_frame(physics, prefix),
            }
            for obj, prefix in balls
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
        record = {
            "frame_index": frame,
            "time_sec": (frame - 1) / float(fps),
        }
        for obj, prefix in balls:
            record[f"{prefix}_present"] = bool(physics_frame[f"{prefix}_present"])
            record[f"{prefix}_matrix_world"] = [
                [float(v) for v in row] for row in obj.matrix_world
            ]
            record[f"{prefix}_location"] = [float(v) for v in obj.location]
            for field in ("linear_velocity", "angular_velocity", "table_gap"):
                record[f"{prefix}_{field}"] = physics_frame[f"{prefix}_{field}"]
        # Pair separations, so a consumer can find the contact frames without
        # recomputing distances from the matrices.
        for gap in ("ball_ball_gap", "cue_extra_gap", "extra_target_gap"):
            if gap in physics_frame:
                record[gap] = physics_frame[gap]
        record["camera_matrix_world"] = [
            [float(v) for v in row] for row in camera.matrix_world
        ]
        record["camera_world_to_camera_matrix"] = [
            [float(v) for v in row] for row in camera.matrix_world.inverted()
        ]
        records["frames"].append(record)

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

    cue_ball, target_ball, extra_ball, camera = build_scene(args, scenario)
    physics = run_physics_simulation(args, scenario, scenario["physics"]["surface_z"], scenario["physics"]["ball_radius"])
    balls = [
        (cue_ball, "cue_ball"),
        (target_ball, "target_ball"),
        (extra_ball, "extra_ball"),
    ]
    apply_physics_animation(balls, physics)
    export_ground_truth(
        out_dir,
        balls,
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
