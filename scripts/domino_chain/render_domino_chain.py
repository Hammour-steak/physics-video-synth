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
GLB_PATH = WORKSPACE_DIR / "assets" / "models" / "domino_test.glb"

OUTPUT_STEM = "domino_chain"
DIRECT_MP4_NAME = f"{OUTPUT_STEM}.mp4"
BLEND_NAME = f"{OUTPUT_STEM}.blend"
GROUND_TRUTH_NAME = "ground_truth_transforms.json"
PHYSICS_TEMP_NAME = "physics_transforms.json"
SCENARIO_METADATA_NAME = "scenario_metadata.json"

# Geometry taken from assets/models/domino_test.glb: a single upright tile
# with local X = thickness (row axis), Y = width, Z = height.
FLOOR_Z = -0.0322
DOMINO_THICKNESS = 0.20
DOMINO_WIDTH = 0.70
DOMINO_HEIGHT = 1.30

# Side-on view of the row so the falling cascade reads left-to-right.
# Tuned for a 4-tile row (~2.4m); widen the distance if domino_count is
# increased via --scenario-overrides-json.
CAMERA_LOCATION = (0.3, -5.8, 2.1)
CAMERA_TARGET = (0.0, 0.0, FLOOR_Z + 0.55)
LIGHT_TARGET = (0.0, 0.0, FLOOR_Z + 0.55)


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
    look_at(light, target)
    return light


def import_domino_assets() -> tuple[bpy.types.Object, bpy.types.Object]:
    """Import the domino GLB, keep the tabletop plane and a single domino
    tile as a reusable master mesh, and discard every other pre-placed
    domino instance from the source file (we build our own straight row of
    duplicates in build_scene)."""
    if not GLB_PATH.exists():
        raise FileNotFoundError(f"Domino model not found: {GLB_PATH}")

    existing = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(GLB_PATH))
    imported = [obj for obj in bpy.context.scene.objects if obj not in existing]
    imported_names = [obj.name for obj in imported]
    print(f"[INFO] Imported {len(imported)} objects from {GLB_PATH.name}")

    desktop = bpy.data.objects.get("Plane_0")
    if desktop is None:
        raise RuntimeError(f"No tabletop mesh (Plane_0) found in {GLB_PATH}")

    master_parts = [bpy.data.objects.get("5-6_0"), bpy.data.objects.get("5-6_1")]
    master_parts = [obj for obj in master_parts if obj is not None]
    if not master_parts:
        raise RuntimeError(f"No domino tile mesh (5-6_0/5-6_1) found in {GLB_PATH}")

    # Detach the kept meshes from the imported empty hierarchy, baking the
    # parent transforms (including the glTF Y-up -> Z-up conversion) into
    # each object's own transform.
    bpy.ops.object.select_all(action="DESELECT")
    for obj in [desktop, *master_parts]:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = desktop
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")

    for obj in [desktop, *master_parts]:
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # Join the domino body and its inset face into a single tile mesh.
    bpy.ops.object.select_all(action="DESELECT")
    for obj in master_parts:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = master_parts[0]
    bpy.ops.object.join()
    domino_master = bpy.context.active_object
    domino_master.name = "domino_master"

    # Re-center the origin on the tile so placing it by `.location` positions
    # its geometric center, matching the PyBullet box body convention.
    bpy.ops.object.select_all(action="DESELECT")
    domino_master.select_set(True)
    bpy.context.view_layer.objects.active = domino_master
    bpy.ops.object.origin_set(type="GEOMETRY_ORIGIN", center="BOUNDS")
    domino_master.location = (0.0, 0.0, 0.0)
    domino_master.rotation_mode = "QUATERNION"
    domino_master.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
    domino_master.scale = (1.0, 1.0, 1.0)

    desktop.name = "table_surface"
    desktop.rotation_mode = "QUATERNION"

    # Remove every other pre-placed domino instance and now-empty parent
    # nodes left over from the imported hierarchy.
    kept = {desktop.name, domino_master.name}
    for name in imported_names:
        if name in kept:
            continue
        obj = bpy.data.objects.get(name)
        if obj is None:
            continue
        bpy.data.objects.remove(obj, do_unlink=True)

    return desktop, domino_master


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
                "lens_mm": 40.0,
            },
            "physics": {
                # Row size: baseline 4 tiles. The PCVE vocab hard-codes 4
                # named objects (domino_1 .. domino_4) that address per-tile
                # lists below; changing this drift-locks the vocab.
                "domino_count": 4,
                "domino_spacing": 0.8,
                "domino_thickness": DOMINO_THICKNESS,
                "domino_width": DOMINO_WIDTH,
                "domino_height": DOMINO_HEIGHT,
                # Per-tile lists in row order (index 0 = pushed tile). A
                # PCVE edit names one tile and writes at its index; the
                # scalars below stay as defaults for CLI callers that do
                # not care about per-tile control.
                "domino_masses": [0.12, 0.12, 0.12, 0.12],
                "domino_frictions": [0.6, 0.6, 0.6, 0.6],
                "domino_restitutions": [0.05, 0.05, 0.05, 0.05],
                "domino_active": [1, 1, 1, 1],
                "domino_mass": 0.12,
                "domino_friction": 0.6,
                "domino_restitution": 0.05,
                "floor_friction": 0.6,
                "push_angle_deg": 12.0,
                "floor_z": FLOOR_Z,
                "gravity": [0.0, 0.0, -9.81],
                "scene_offset_x": 0.0,
                "scene_offset_y": 0.0,
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

    script_path = Path(__file__).with_name("simulate_domino_chain.py")
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
            "--domino-count",
            str(int(physics["domino_count"])),
            "--domino-spacing",
            str(float(physics["domino_spacing"])),
            "--domino-thickness",
            str(float(physics["domino_thickness"])),
            "--domino-width",
            str(float(physics["domino_width"])),
            "--domino-height",
            str(float(physics["domino_height"])),
            "--domino-mass",
            str(float(physics["domino_mass"])),
            "--domino-friction",
            str(float(physics["domino_friction"])),
            "--domino-restitution",
            str(float(physics["domino_restitution"])),
            "--floor-friction",
            str(float(physics["floor_friction"])),
            "--push-angle-deg",
            str(float(physics["push_angle_deg"])),
            "--gravity-z",
            str(float(physics["gravity"][2])),
            "--floor-z",
            str(float(physics["floor_z"])),
            "--scene-offset-x",
            str(float(physics["scene_offset_x"])),
            "--scene-offset-y",
            str(float(physics["scene_offset_y"])),
            "--domino-masses", *[str(float(v)) for v in physics["domino_masses"]],
            "--domino-frictions-list", *[str(float(v)) for v in physics["domino_frictions"]],
            "--domino-restitutions-list", *[str(float(v)) for v in physics["domino_restitutions"]],
            "--domino-active", *[str(int(v)) for v in physics["domino_active"]],
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


def apply_physics_animation(dominoes: list, physics: dict) -> None:
    for obj in dominoes:
        if obj is None:
            continue
        obj.rotation_mode = "QUATERNION"

    for frame_record in physics["frames"]:
        frame = int(frame_record["frame_index"])
        for idx, domino_obj in enumerate(dominoes):
            if domino_obj is None:
                continue
            domino_data = frame_record["dominoes"][idx]
            dquat = domino_data["quaternion_xyzw"]
            domino_obj.location = domino_data["location"]
            domino_obj.rotation_quaternion = (
                dquat[3],
                dquat[0],
                dquat[1],
                dquat[2],
            )
            domino_obj.keyframe_insert(data_path="location", frame=frame)
            domino_obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)

    set_linear_keyframes([o for o in dominoes if o is not None])


def export_ground_truth(
    out_dir: Path,
    dominoes: list[bpy.types.Object],
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
            "dominoes": [
                (
                    {"present": False, "object_name": None, "index": idx}
                    if domino is None
                    else {"present": True, "object_name": domino.name, "index": idx}
                )
                for idx, domino in enumerate(dominoes)
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
                "dominoes": [
                    (
                        {"present": False}
                        if domino is None
                        else {
                            "present": True,
                            "matrix_world": [[float(v) for v in row] for row in domino.matrix_world],
                            "location": [float(v) for v in domino.location],
                            "linear_velocity": domino_data["linear_velocity"],
                            "angular_velocity": domino_data["angular_velocity"],
                            "tilt_deg": domino_data["tilt_deg"],
                        }
                    )
                    for domino, domino_data in zip(dominoes, physics_frame["dominoes"])
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


def build_scene(args: argparse.Namespace, scenario: dict[str, object]) -> tuple[list[bpy.types.Object], bpy.types.Object]:
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

    hdri_path = POLYHAVEN_DIR / "lythwood_room" / "lythwood_room_2k.hdr"
    if hdri_path.exists():
        hdri_img = bpy.data.images.load(str(hdri_path), check_existing=True)
        env_tex.image = hdri_img
        print(f"[INFO] Using HDRI background: lythwood_room")
    else:
        print(f"[WARN] HDRI not found at {hdri_path}, using solid fallback")
        bg_node = world_nodes.new(type="ShaderNodeBackground")
        bg_node.inputs["Color"].default_value = (0.30, 0.30, 0.32, 1.0)
        bg_node.inputs["Strength"].default_value = 1.0
        env_tex = bg_node

    mapping_node = world_nodes.new(type="ShaderNodeMapping")
    mapping_node.location = (-550, 0)
    mapping_node.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(270))

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
    camera.data.lens = 40
    scene.camera = camera

    target = bpy.data.objects.new("camera_target", None)
    scene.collection.objects.link(target)
    target.location = CAMERA_TARGET
    constraint = camera.constraints.new("TRACK_TO")
    constraint.target = target
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"

    bpy.ops.object.light_add(type="SUN", location=(3.0, -3.0, 4.0))
    sun = bpy.context.object
    sun.data.energy = 1.2
    sun.rotation_euler = (math.radians(45), math.radians(15), math.radians(30))

    add_area_light("fill_light", location=(-2.0, -3.5, 2.5), power=200, size=2.5, target=LIGHT_TARGET)
    add_area_light("rim_light", location=(1.5, 2.0, 2.0), power=120, size=2.0, target=LIGHT_TARGET)

    desktop, domino_master = import_domino_assets()

    physics = scenario["physics"]
    assert isinstance(physics, dict)
    count = int(physics["domino_count"])
    spacing = float(physics["domino_spacing"])
    offset_x = float(physics.get("scene_offset_x", 0.0))
    offset_y = float(physics.get("scene_offset_y", 0.0))
    floor_z = float(physics.get("floor_z", FLOOR_Z))
    height = float(physics["domino_height"])

    row_start = -(count - 1) * spacing / 2.0
    active_flags = physics.get("domino_active", [1] * count)
    dominoes: list = []
    first_present_index = next((i for i, a in enumerate(active_flags) if int(a)), None)
    for i in range(count):
        if not int(active_flags[i]):
            dominoes.append(None)
            continue
        # Use the imported master mesh for the first present tile, copy for
        # the rest -- otherwise the master ends up unlinked when index 0 is
        # removed by an edit.
        obj = domino_master if i == first_present_index else domino_master.copy()
        if i != first_present_index:
            scene.collection.objects.link(obj)
        obj.name = f"domino_{i:03d}"
        obj.rotation_mode = "QUATERNION"
        obj.location = (
            row_start + i * spacing + offset_x,
            offset_y,
            floor_z + height / 2.0,
        )
        obj.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        dominoes.append(obj)

    frame_end = max(2, int(round(float(args.duration_sec) * int(args.fps))))
    scene.frame_start = 1
    scene.frame_end = frame_end

    return dominoes, camera


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

    dominoes, camera = build_scene(args, scenario)
    physics = run_physics_simulation(args, scenario)
    apply_physics_animation(dominoes, physics)
    export_ground_truth(
        out_dir,
        dominoes,
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
