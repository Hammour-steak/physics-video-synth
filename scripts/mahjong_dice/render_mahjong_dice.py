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
GLB_PATH = WORKSPACE_DIR / "assets" / "models" / "riichi_mahjong.glb"

OUTPUT_STEM = "mahjong_dice"
DIRECT_MP4_NAME = f"{OUTPUT_STEM}.mp4"
BLEND_NAME = f"{OUTPUT_STEM}.blend"
GROUND_TRUTH_NAME = "ground_truth_transforms.json"
PHYSICS_TEMP_NAME = "physics_transforms.json"
SCENARIO_METADATA_NAME = "scenario_metadata.json"

# Geometry taken from assets/models/riichi_mahjong.glb: the two decorative
# dice in the table's center tray are 0.0833-unit cubes resting at world Z
# 0.9382 on a flat surface.
DIE_EDGE = 0.0833
TRAY_Z = 0.9382
FLOOR_Z = TRAY_Z - DIE_EDGE / 2.0
DIE_0_XY = (-0.1612, -0.0052)
DIE_1_XY = (-0.1991, 0.1425)

TRAY_CENTER = ((DIE_0_XY[0] + DIE_1_XY[0]) / 2.0, (DIE_0_XY[1] + DIE_1_XY[1]) / 2.0)
CAMERA_LOCATION = (TRAY_CENTER[0] + 1.75, TRAY_CENTER[1] - 1.85, FLOOR_Z + 1.7)
CAMERA_TARGET = (TRAY_CENTER[0], TRAY_CENTER[1], FLOOR_Z + 0.15)
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


def import_mahjong_assets() -> tuple[bpy.types.Object, bpy.types.Object]:
    """Import the full riichi mahjong table GLB and keep every part as a
    static background (tile racks, glass cover, dice shaker, table body) for
    a realistic setting. Detach the two decorative dice meshes so they can
    be driven by physics; this also removes them from their old static spot
    in the background (transform_apply re-centers them at the world origin).
    """
    if not GLB_PATH.exists():
        raise FileNotFoundError(f"Mahjong table model not found: {GLB_PATH}")

    bpy.ops.import_scene.gltf(filepath=str(GLB_PATH))

    die_parts = [bpy.data.objects.get("dice_4k_0"), bpy.data.objects.get("dice_1_4k_0")]
    die_parts = [obj for obj in die_parts if obj is not None]
    if len(die_parts) != 2:
        raise RuntimeError(f"Expected 2 dice meshes (dice_4k_0, dice_1_4k_0) in {GLB_PATH}")

    bpy.ops.object.select_all(action="DESELECT")
    for obj in die_parts:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = die_parts[0]
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")

    # Tint the two dice with distinct colours while preserving the printed
    # pips. The GLB drives Base Color from an image texture (the pip art);
    # both dice share ONE material, so we (a) give each die its own copy,
    # then (b) insert a MULTIPLY MixRGB node between the pip texture and
    # the Principled BSDF's Base Color, and set the mix colour per die.
    # Multiply-tint keeps the pip contrast intact: white body -> tinted
    # body, dark pips stay dark.
    die_tints = [
        (0.90, 0.15, 0.15, 1.0),   # red_die   (idx 0) -- warm red body
        (0.96, 0.95, 0.90, 1.0),   # white_die (idx 1) -- near-white, slight cream
    ]
    die_names = ["red_die", "white_die"]

    dice = []
    for idx, obj in enumerate(die_parts):
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        bpy.ops.object.origin_set(type="GEOMETRY_ORIGIN", center="BOUNDS")
        obj.name = die_names[idx]
        obj.location = (0.0, 0.0, 0.0)
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        obj.scale = (1.0, 1.0, 1.0)

        if obj.data.materials and obj.data.materials[0] is not None:
            base = obj.data.materials[0]
            mat = base.copy()
            mat.name = f"{die_names[idx]}_mat"
            tree = mat.node_tree
            bsdf = tree.nodes.get("Principled BSDF")
            if bsdf is not None and bsdf.inputs["Base Color"].is_linked:
                src_link = bsdf.inputs["Base Color"].links[0]
                src_socket = src_link.from_socket
                tree.links.remove(src_link)
                mix = tree.nodes.new("ShaderNodeMixRGB")
                mix.name = f"tint_{die_names[idx]}"
                mix.blend_type = "MULTIPLY"
                mix.inputs["Fac"].default_value = 1.0
                mix.inputs["Color2"].default_value = die_tints[idx]
                tree.links.new(src_socket, mix.inputs["Color1"])
                tree.links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
            obj.data.materials.clear()
            obj.data.materials.append(mat)

        dice.append(obj)

    return dice[0], dice[1]


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
                "lens_mm": 50.0,
            },
            "physics": {
                "die_edge": DIE_EDGE,
                # Per-die lists in slot order (die_1 = index 0, die_2 = index 1).
                # A PCVE edit names one die and writes at its index; the
                # scalars below stay as fallback defaults.
                "die_masses": [0.006, 0.006],
                "die_frictions": [0.5, 0.5],
                "die_restitutions": [0.72, 0.72],
                "die_active": [1, 1],
                # Per-die initial DOWNWARD speed (m/s). Pure vertical throw,
                # no lateral component and no spin: die_1 is thrown harder
                # than die_2 so the two moments of impact are distinct. This
                # is the PCVE `initial_velocity` knob per die.
                "die_initial_speeds": [1.5, 0.8],
                "die_mass": 0.006,
                "die_friction": 0.5,
                "die_restitution": 0.72,
                "floor_friction": 0.55,
                # Split out from --die-restitution: pre-PCVE the two were
                # locked together, but an edit on one die's restitution must
                # only touch that die and its pairings, not the floor.
                "floor_restitution": 0.72,
                "drop_height": 0.6,
                "floor_z": FLOOR_Z,
                "die_0_xy": list(DIE_0_XY),
                "die_1_xy": list(DIE_1_XY),
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

    script_path = Path(__file__).with_name("simulate_mahjong_dice.py")
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
            "--die-edge",
            str(float(physics["die_edge"])),
            "--die-mass",
            str(float(physics["die_mass"])),
            "--die-friction",
            str(float(physics["die_friction"])),
            "--die-restitution",
            str(float(physics["die_restitution"])),
            "--floor-friction",
            str(float(physics["floor_friction"])),
            "--drop-height",
            str(float(physics["drop_height"])),
            "--gravity-z",
            str(float(physics["gravity"][2])),
            "--floor-z",
            str(float(physics["floor_z"])),
            "--die-0-xy",
            *[str(float(v)) for v in physics["die_0_xy"]],
            "--die-1-xy",
            *[str(float(v)) for v in physics["die_1_xy"]],
            "--floor-restitution", str(float(physics.get("floor_restitution", physics["die_restitution"]))),
            "--die-masses", *[str(float(v)) for v in physics["die_masses"]],
            "--die-frictions-list", *[str(float(v)) for v in physics["die_frictions"]],
            "--die-restitutions-list", *[str(float(v)) for v in physics["die_restitutions"]],
            "--die-active", *[str(int(v)) for v in physics["die_active"]],
            "--die-initial-speeds", *[str(float(v)) for v in physics["die_initial_speeds"]],
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


def apply_physics_animation(dice: list, physics: dict) -> None:
    for obj in dice:
        if obj is None:
            continue
        obj.rotation_mode = "QUATERNION"

    for frame_record in physics["frames"]:
        frame = int(frame_record["frame_index"])
        for idx, die_obj in enumerate(dice):
            if die_obj is None:
                continue
            die_data = frame_record["dice"][idx]
            dquat = die_data["quaternion_xyzw"]
            die_obj.location = die_data["location"]
            die_obj.rotation_quaternion = (
                dquat[3],
                dquat[0],
                dquat[1],
                dquat[2],
            )
            die_obj.keyframe_insert(data_path="location", frame=frame)
            die_obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)

    set_linear_keyframes([o for o in dice if o is not None])


def export_ground_truth(
    out_dir: Path,
    dice: list[bpy.types.Object],
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
            "dice": [
                (
                    {"present": False, "object_name": None, "index": idx}
                    if die is None
                    else {"present": True, "object_name": die.name, "index": idx}
                )
                for idx, die in enumerate(dice)
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
                "dice": [
                    (
                        {"present": False}
                        if die is None
                        else {
                            "present": True,
                            "matrix_world": [[float(v) for v in row] for row in die.matrix_world],
                            "location": [float(v) for v in die.location],
                            "linear_velocity": die_data["linear_velocity"],
                            "angular_velocity": die_data["angular_velocity"],
                        }
                    )
                    for die, die_data in zip(dice, physics_frame["dice"])
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
        bg_node.inputs["Color"].default_value = (0.25, 0.22, 0.20, 1.0)
        bg_node.inputs["Strength"].default_value = 1.0
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
        bg_node.inputs["Strength"].default_value = 0.7
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

    bpy.ops.object.light_add(type="SUN", location=(1.0, -1.0, 1.5))
    sun = bpy.context.object
    sun.data.energy = 1.0
    sun.rotation_euler = (math.radians(45), math.radians(15), math.radians(30))

    add_area_light(
        "fill_light",
        location=(TRAY_CENTER[0] - 0.4, TRAY_CENTER[1] - 0.4, FLOOR_Z + 0.5),
        power=15,
        size=0.5,
        target=LIGHT_TARGET,
    )
    add_area_light(
        "rim_light",
        location=(TRAY_CENTER[0] + 0.3, TRAY_CENTER[1] + 0.3, FLOOR_Z + 0.4),
        power=10,
        size=0.4,
        target=LIGHT_TARGET,
    )

    die_0, die_1 = import_mahjong_assets()
    imported = [die_0, die_1]

    physics = scenario["physics"]
    assert isinstance(physics, dict)
    floor_z = float(physics.get("floor_z", FLOOR_Z))
    drop_height = float(physics["drop_height"])
    edge = float(physics["die_edge"])
    die_xy = [tuple(physics["die_0_xy"]), tuple(physics["die_1_xy"])]
    active = list(physics.get("die_active", [1, 1]))

    dice: list = []
    for idx, (obj, (x, y)) in enumerate(zip(imported, die_xy)):
        if not int(active[idx]):
            # Remove the imported placeholder from the scene so a deleted die
            # doesn't show up anywhere -- it's easier than trying to hide it
            # after the fact and matches the sim's frame slot being present=false.
            bpy.data.objects.remove(obj, do_unlink=True)
            dice.append(None)
            continue
        obj.location = (x, y, floor_z + edge / 2.0 + drop_height + 0.05 * idx)
        obj.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        dice.append(obj)

    frame_end = max(2, int(round(float(args.duration_sec) * int(args.fps))))
    scene.frame_start = 1
    scene.frame_end = frame_end

    return dice, camera


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

    dice, camera = build_scene(args, scenario)
    physics = run_physics_simulation(args, scenario)
    apply_physics_animation(dice, physics)
    export_ground_truth(
        out_dir,
        dice,
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
