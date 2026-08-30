from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Vector

WORKSPACE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = WORKSPACE_DIR / "assets" / "models"

ROOM_GLB = MODELS_DIR / "dining_room__kichen_baked.glb"
CAN_GLB = MODELS_DIR / "simple_cola_can.glb"
CUP_GLB = MODELS_DIR / "fast_food_soda_cup.glb"
MILK_GLB = MODELS_DIR / "milk_packaging.glb"

OUTPUT_STEM = "dining_chain"
PHYSICS_TEMP = "physics_transforms.json"
GROUND_TRUTH_NAME = "ground_truth_transforms.json"
SCENARIO_METADATA_NAME = "scenario_metadata.json"

# Uniform scale factors that bring each raw GLB to its real-world size, and the
# collision-proxy half-height used by simulate_dining_chain.py (the physics
# body centre is placed at base + this height, so the visual origin is set to
# match). Can/cup GLBs come in at ~2-5 blender-units; the milk carton is
# already near metres.
CAN_SCALE = 0.122 / 3.260      # -> 12.2 cm tall
CUP_SCALE = 0.088 / 2.396      # -> ~8.8 cm body diameter (Y, straw-free axis)
MILK_SCALE = 0.125 / 0.297     # -> 12.5 cm tall carton (~can height)

CONTAINERS = {
    # key: (glb, scale, proxy_half_height_m)
    "can": (CAN_GLB, CAN_SCALE, 0.061),
    "cup": (CUP_GLB, CUP_SCALE, 0.080),
    "milk": (MILK_GLB, MILK_SCALE, 0.0625),
}
# Physical baseline order matches simulate_dining_chain.ORDER = (can, cup, milk).
ORDER = ("can", "cup", "milk")

# Straight-on front view: camera faces the chain along -X, roughly level with
# the drinks, so the three of them stand side-by-side and slide left->right
# (along +Y) across the frame rather than at an oblique angle. The dining
# chairs on the camera (+X) side are culled (see CULL_SEATS_BEYOND_X) so they
# don't block this near-level view; the far chairs stay as background.
CAMERA_LOCATION = (2.90, -0.50, 1.02)
CAMERA_TARGET = (1.15, -0.50, 0.85)
CAMERA_LENS_MM = 35.0

CULL_SEATS_BEYOND_X = 1.5

WORLD_STRENGTH = 0.20
KEY_LIGHT = {"loc": (0.9, -0.5, 1.7), "power": 40.0, "size": 1.1, "color": (1.0, 0.95, 0.86)}
WINDOW_LIGHT = {"loc": (1.4, 2.6, 1.5), "power": 25.0, "size": 2.0, "color": (0.85, 0.9, 1.0)}
EXPOSURE = -0.8
LIGHT_TARGET = (1.05, -0.5, 0.80)


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preview", "animation"), default="animation")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resolution", nargs=2, type=int, default=(960, 540))
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-sec", type=float, default=3.0)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--preview-frame", type=int, default=30)
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--launch-speed", type=float, default=3.3)
    parser.add_argument("--table-friction", type=float, default=0.30)
    parser.add_argument(
        "--scenario-overrides-json",
        type=Path,
        default=None,
        help="JSON file merged recursively onto the scenario built from the flags "
        "above. This is how the PCVE suite applies a single edit: it writes "
        "{\"physics\": {...}} touching one parameter and leaves everything else, "
        "including the camera and the seed, identical to the source video.",
    )
    return parser.parse_args(argv)


# --- Scenario ----------------------------------------------------------------

def read_json(path: Path) -> dict:
    data = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def recursive_update(base: dict, updates: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = recursive_update(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def create_scenario(args: argparse.Namespace) -> dict:
    """Build the full scenario, then apply any overrides on top.

    The physics block below is the baseline this scene's PCVE suite edits, and
    must stay in sync with both simulate_dining_chain.py's defaults and
    edit_vocab.BASELINE_PHYSICS -- an edit's "from" value is read out of the
    vocabulary, so a drift here silently mislabels every physics diff. Every
    per-object list is in chain order (can, cup, milk).
    """
    scenario = {
        "schema_version": 2,
        "seed": int(args.seed),
        "environment": "modern_dining_room",
        "render": {
            "fps": int(args.fps),
            "duration_sec": float(args.duration_sec),
            "resolution": [int(args.resolution[0]), int(args.resolution[1])],
            "samples": int(args.samples),
            "device": str(args.device),
            "mode": str(args.mode),
        },
        "physics": {
            "can_initial_speed": float(args.launch_speed),
            "object_masses": [0.36, 0.30, 0.35],
            "object_frictions": [0.30, 0.30, 0.30],
            "object_restitutions": [0.10, 0.10, 0.10],
            "object_active": [1, 1, 1],
            "table_friction": float(args.table_friction),
            "table_restitution": 0.10,
            "gravity_z": -9.8,
        },
    }
    if args.scenario_overrides_json is not None:
        scenario = recursive_update(scenario, read_json(args.scenario_overrides_json))
        scenario["scenario_overrides_path"] = str(
            args.scenario_overrides_json.expanduser().resolve()
        )
    return scenario


# --- Physics -----------------------------------------------------------------

def run_physics(args: argparse.Namespace, scenario: dict) -> dict:
    physics_python = WORKSPACE_DIR.parent / "miniconda" / "envs" / "physics" / "bin" / "python"
    python = str(physics_python) if physics_python.exists() else (shutil.which("python3") or shutil.which("python"))
    if not python:
        raise RuntimeError("Cannot find python3/python for the PyBullet physics simulation.")
    script = Path(__file__).with_name("simulate_dining_chain.py")
    out = args.out_dir / PHYSICS_TEMP
    physics = scenario["physics"]

    def triple(key: str) -> list[str]:
        values = physics[key]
        if len(values) != 3:
            raise ValueError(f"scenario physics {key!r} must have 3 entries, got {values!r}")
        return [str(float(v)) for v in values]

    subprocess.run([
        python, str(script), "--out", str(out),
        "--fps", str(int(args.fps)), "--duration-sec", str(float(args.duration_sec)),
        "--can-initial-speed", str(float(physics["can_initial_speed"])),
        "--table-friction", str(float(physics["table_friction"])),
        "--restitution", str(float(physics["table_restitution"])),
        "--gravity-z", str(float(physics["gravity_z"])),
        "--object-masses", *triple("object_masses"),
        "--object-frictions", *triple("object_frictions"),
        "--object-restitutions", *triple("object_restitutions"),
        "--object-active", *[str(int(v)) for v in physics["object_active"]],
    ], check=True)
    data = json.loads(out.read_text(encoding="utf-8"))
    out.unlink(missing_ok=True)
    return data


# --- Blender helpers ---------------------------------------------------------

def world_bbox(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    cs = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs = [c.x for c in cs]; ys = [c.y for c in cs]; zs = [c.z for c in cs]
    return Vector((min(xs), min(ys), min(zs))), Vector((max(xs), max(ys), max(zs)))


def look_at(obj: bpy.types.Object, target) -> None:
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_area_light(name, loc, power, size, target, color=(1, 1, 1)) -> None:
    bpy.ops.object.light_add(type="AREA", location=loc)
    light = bpy.context.object
    light.name = name
    light.data.energy = float(power)
    light.data.size = float(size)
    light.data.color = color
    look_at(light, target)


def import_container(glb: Path, name: str, scale: float, proxy_half_h: float) -> bpy.types.Object:
    existing = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(glb))
    imported = [o for o in bpy.context.scene.objects if o not in existing]
    meshes = [o for o in imported if o.type == "MESH"]
    empty_names = [o.name for o in imported if o.type != "MESH"]

    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    if len(meshes) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for o in meshes:
            o.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    for nm in empty_names:
        e = bpy.data.objects.get(nm)
        if e is not None and e is not obj:
            bpy.data.objects.remove(e, do_unlink=True)

    obj.scale = (scale, scale, scale)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    mn, mx = world_bbox(obj)
    bpy.context.scene.cursor.location = ((mn.x + mx.x) / 2.0, (mn.y + mx.y) / 2.0, mn.z + proxy_half_h)
    bpy.ops.object.origin_set(type="ORIGIN_CURSOR")
    obj.name = name
    obj.rotation_mode = "QUATERNION"
    return obj


def cull_seats_beyond_x(x_threshold: float) -> None:
    for obj in list(bpy.context.scene.objects):
        if obj.type != "MESH" or "seat" not in obj.name.lower():
            continue
        mw = obj.matrix_world
        me = obj.data
        bm = bmesh.new()
        bm.from_mesh(me)
        doomed = [v for v in bm.verts if (mw @ v.co).x > x_threshold]
        if doomed:
            bmesh.ops.delete(bm, geom=doomed, context="VERTS")
        bm.to_mesh(me)
        bm.free()
        me.update()


def apply_keyframes(obj: bpy.types.Object, frames: list, key: str) -> None:
    obj.rotation_mode = "QUATERNION"
    for fr in frames:
        d = fr["objects"][key]
        obj.location = d["location"]
        q = d["quaternion_xyzw"]
        obj.rotation_quaternion = (q[3], q[0], q[1], q[2])
        obj.keyframe_insert(data_path="location", frame=int(fr["frame_index"]))
        obj.keyframe_insert(data_path="rotation_quaternion", frame=int(fr["frame_index"]))
    if obj.animation_data and obj.animation_data.action:
        for fc in obj.animation_data.action.fcurves:
            for k in fc.keyframe_points:
                k.interpolation = "LINEAR"


def build_scene(args: argparse.Namespace, physics: dict, scenario: dict):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.resolution_x = args.resolution[0]
    scene.render.resolution_y = args.resolution[1]
    scene.render.fps = args.fps
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.samples
    scene.cycles.device = "GPU" if args.device == "auto" else "CPU"
    scene.cycles.max_bounces = 12
    scene.cycles.transmission_bounces = 8
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.view_settings.exposure = EXPOSURE

    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (0.55, 0.56, 0.58, 1.0)
    bg.inputs["Strength"].default_value = WORLD_STRENGTH

    bpy.ops.object.camera_add(location=CAMERA_LOCATION)
    camera = bpy.context.object
    camera.data.lens = CAMERA_LENS_MM
    scene.camera = camera
    target = bpy.data.objects.new("camera_target", None)
    scene.collection.objects.link(target)
    target.location = CAMERA_TARGET
    con = camera.constraints.new("TRACK_TO")
    con.target = target
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"

    add_area_light("key_light", KEY_LIGHT["loc"], KEY_LIGHT["power"], KEY_LIGHT["size"], LIGHT_TARGET, KEY_LIGHT["color"])
    add_area_light("window_light", WINDOW_LIGHT["loc"], WINDOW_LIGHT["power"], WINDOW_LIGHT["size"], LIGHT_TARGET, WINDOW_LIGHT["color"])

    if not ROOM_GLB.exists():
        raise FileNotFoundError(f"Room model not found: {ROOM_GLB}")
    bpy.ops.import_scene.gltf(filepath=str(ROOM_GLB))
    cull_seats_beyond_x(CULL_SEATS_BEYOND_X)

    frames = physics["frames"]
    active_flags = scenario["physics"]["object_active"]
    containers: dict[str, bpy.types.Object | None] = {}
    for index, key in enumerate(ORDER):
        if not int(active_flags[index]):
            containers[key] = None
            continue
        glb, scale, half_h = CONTAINERS[key]
        obj = import_container(glb, f"chain_{key}", scale, half_h)
        apply_keyframes(obj, frames, key)
        containers[key] = obj

    frame_end = int(physics["frame_end"])
    scene.frame_start = 1
    scene.frame_end = frame_end
    return containers, camera


# --- Ground truth ------------------------------------------------------------

def export_ground_truth(out_dir: Path, containers, camera, physics: dict, scenario: dict) -> None:
    scene = bpy.context.scene
    fps = int(physics["fps"])
    frame_end = int(physics["frame_end"])
    records = {
        "schema_version": 1,
        "fps": fps,
        "frame_start": 1,
        "frame_end": frame_end,
        "table_top_z": physics["table_top_z"],
        "chain_x": physics["chain_x"],
        "physics": {k: v for k, v in physics.items() if k != "frames"},
        "objects": {
            key: (
                {"present": False, "object_name": None} if containers[key] is None
                else {"present": True, "object_name": containers[key].name}
            )
            for key in ORDER
        },
        "camera": {
            "object_name": camera.name,
            "lens_mm": float(camera.data.lens),
            "resolution": [int(scene.render.resolution_x), int(scene.render.resolution_y)],
        },
        "scenario": scenario,
        "frames": [],
    }
    by_frame = {int(f["frame_index"]): f for f in physics["frames"]}
    for frame in range(1, frame_end + 1):
        scene.frame_set(frame)
        source = by_frame[frame]
        entry = {"frame_index": frame, "time_sec": (frame - 1) / float(fps)}
        for key in ORDER:
            obj = containers[key]
            if obj is None:
                entry[key] = {"present": False}
                continue
            data = source["objects"][key]
            entry[key] = {
                "present": True,
                "matrix_world": [[float(v) for v in row] for row in obj.matrix_world],
                "linear_velocity": data["linear_velocity"],
                "angular_velocity": data["angular_velocity"],
            }
        entry["camera_matrix_world"] = [
            [float(v) for v in row] for row in camera.matrix_world
        ]
        records["frames"].append(entry)
    (out_dir / GROUND_TRUTH_NAME).write_text(json.dumps(records, indent=2), encoding="utf-8")


def render_preview(args: argparse.Namespace) -> None:
    scene = bpy.context.scene
    scene.frame_set(max(1, min(int(args.preview_frame), scene.frame_end)))
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(args.out_dir / "preview.png")
    bpy.ops.render.render(write_still=True)


def render_animation(args: argparse.Namespace) -> None:
    scene = bpy.context.scene
    scene.frame_set(scene.frame_start)
    scene.render.filepath = str(args.out_dir / f"{OUTPUT_STEM}.mp4")
    bpy.ops.render.render(animation=True)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    scenario = create_scenario(args)
    (args.out_dir / SCENARIO_METADATA_NAME).write_text(
        json.dumps(scenario, indent=2), encoding="utf-8"
    )

    physics = run_physics(args, scenario)
    containers, camera = build_scene(args, physics, scenario)
    export_ground_truth(args.out_dir, containers, camera, physics, scenario)

    if args.mode == "preview":
        render_preview(args)
    else:
        render_animation(args)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.out_dir / f"{OUTPUT_STEM}.blend"))
    print(f"[INFO] Render complete. Output: {args.out_dir}")


if __name__ == "__main__":
    main()
