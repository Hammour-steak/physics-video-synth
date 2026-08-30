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
TENNIS_COURT_PATH = ASSETS_DIR / "models" / "tennis_court.glb"
TENNIS_BALL_PATH = ASSETS_DIR / "models" / "tennis_ball-3.glb"

OUTPUT_STEM = "tennis_flight"
DIRECT_MP4_NAME = f"{OUTPUT_STEM}.mp4"
BLEND_NAME = f"{OUTPUT_STEM}.blend"
GROUND_TRUTH_NAME = "ground_truth_transforms.json"
PHYSICS_TEMP_NAME = "physics_transforms.json"
SCENARIO_METADATA_NAME = "scenario_metadata.json"

# Tight side view centered on the net (x=0). Close enough that the ball
# reads clearly on screen while still keeping launch (x=-4) and the
# landing (x≈+1.2) inside the frame.
CAMERA_LOCATION = (0.0, -10.0, 2.5)
CAMERA_TARGET = (0.0, 0.0, 1.0)
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
    parser.add_argument("--duration-sec", type=float, default=3.0)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--preview-frame", type=int, default=15)
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


def prepare_ball_object(obj: bpy.types.Object) -> bpy.types.Object:
    """Detach ball from its parent, bake world transform, and center origin."""
    mw = obj.matrix_world.copy()
    obj.parent = None
    obj.matrix_world = mw

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    obj.select_set(False)
    return obj


def import_tennis_court() -> None:
    """Import the tennis court GLB as static background."""
    if not TENNIS_COURT_PATH.exists():
        raise FileNotFoundError(f"Tennis court model not found: {TENNIS_COURT_PATH}")

    bpy.ops.import_scene.gltf(filepath=str(TENNIS_COURT_PATH))
    imported_count = len(bpy.context.selected_objects)
    print(f"[INFO] Imported tennis court with {imported_count} objects")


def compute_combined_dimensions(objects: list[bpy.types.Object]) -> tuple[float, float, float]:
    """Compute combined bounding box dimensions for a list of objects in world space."""
    import mathutils as mu
    bb_min = mu.Vector((float("inf"),) * 3)
    bb_max = mu.Vector((-float("inf"),) * 3)
    for obj in objects:
        if obj.type != "MESH":
            continue
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ mu.Vector(corner)
            bb_min = mu.Vector((min(a, b) for a, b in zip(bb_min, world_corner)))
            bb_max = mu.Vector((max(a, b) for a, b in zip(bb_max, world_corner)))
    return tuple(bb_max - bb_min)


def join_meshes_into_ball(mesh_objs: list[bpy.types.Object]) -> bpy.types.Object:
    """Join the given mesh objects into a single tennis_ball mesh."""
    if not mesh_objs:
        raise RuntimeError("No mesh objects to join")
    if len(mesh_objs) == 1:
        ball = mesh_objs[0]
        prepare_ball_object(ball)
        return ball

    bpy.ops.object.select_all(action="DESELECT")
    for obj in mesh_objs:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objs[0]
    bpy.ops.object.join()
    ball = bpy.context.view_layer.objects.active
    prepare_ball_object(ball)
    return ball


def import_tennis_ball() -> bpy.types.Object:
    """Import the tennis ball GLB, find and prepare the ball mesh."""
    if not TENNIS_BALL_PATH.exists():
        raise FileNotFoundError(f"Tennis ball model not found: {TENNIS_BALL_PATH}")

    existing = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(TENNIS_BALL_PATH))
    imported = [obj for obj in bpy.data.objects if obj not in existing]

    mesh_objs = [obj for obj in imported if obj.type == "MESH"]
    if not mesh_objs:
        raise RuntimeError(f"No mesh objects found in {TENNIS_BALL_PATH}")

    # Many Sketchfab-style glbs contain one root empty with a hierarchy of meshes
    # that together form the ball. Find the root empty with the most spherical
    # combined bounding box and merge its children into one mesh.
    root_empties = [obj for obj in imported if obj.type == "EMPTY" and obj.parent is None]
    if not root_empties:
        root_empties = [None]  # fallback: treat all imported meshes together

    def sphericity(dims: tuple[float, float, float]) -> float:
        nonzero = [d for d in dims if d > 1e-6]
        if not nonzero:
            return float("inf")
        return max(nonzero) / min(nonzero)

    best_root = None
    best_score = (float("inf"), float("inf"))
    for root in root_empties:
        if root is None:
            children = mesh_objs
        else:
            children = [root] + list(root.children_recursive)
            children = [c for c in children if c.type == "MESH"]
        dims = compute_combined_dimensions(children)
        volume = dims[0] * dims[1] * dims[2]
        score = (sphericity(dims), volume)
        if score < best_score:
            best_score = score
            best_root = root

    if best_root is None:
        ball_children = mesh_objs
    else:
        ball_children = [best_root] + list(best_root.children_recursive)
        ball_children = [c for c in ball_children if c.type == "MESH"]

    # Detach children from parents so they can be joined cleanly.
    for obj in ball_children:
        mw = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = mw

    # Remove all other imported objects that are not part of the chosen ball.
    keep_names = {obj.name for obj in ball_children}
    for obj in imported:
        if obj.name not in keep_names:
            bpy.data.objects.remove(obj, do_unlink=True)

    ball = join_meshes_into_ball(ball_children)
    ball.name = "tennis_ball"
    print(f"[INFO] Joined tennis ball mesh: {ball.name} "
          f"(dimensions: {ball.dimensions.x:.4f}, {ball.dimensions.y:.4f}, {ball.dimensions.z:.4f})")

    return ball


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
                "ball_radius": 0.033,
                "ball_mass": 0.057,
                "ball_friction": 0.5,
                "ball_restitution": 0.05,
                "ball_rolling_friction": 0.015,
                "floor_friction": 0.6,
                "launch_location": [-4.0, 0.0, 1.5],
                "launch_velocity": [4.5, 0.0, 4.5],
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


def run_physics_simulation(
    args: argparse.Namespace,
    scenario: dict[str, object],
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

    def vec3(name: str) -> list[float]:
        value = physics.get(name)
        if isinstance(value, (list, tuple)) and len(value) == 3:
            return [float(v) for v in value]
        return [0.0, 0.0, 0.0]

    launch_loc = vec3("launch_location")
    launch_vel = vec3("launch_velocity")
    gravity = vec3("gravity")

    script_path = Path(__file__).with_name("simulate_tennis_flight.py")
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
            "--ball-rolling-friction",
            str(float(physics.get("ball_rolling_friction", 0.015))),
            "--floor-friction",
            str(float(physics["floor_friction"])),
            "--launch-x",
            str(launch_loc[0]),
            "--launch-y",
            str(launch_loc[1]),
            "--launch-z",
            str(launch_loc[2]),
            "--launch-vx",
            str(launch_vel[0]),
            "--launch-vy",
            str(launch_vel[1]),
            "--launch-vz",
            str(launch_vel[2]),
            "--gravity-z",
            str(gravity[2]),
        ],
        check=True,
    )
    records = json.loads(physics_path.read_text(encoding="utf-8"))
    physics_path.unlink(missing_ok=True)
    return records


def setup_world_and_lights(scenario: dict[str, object]) -> bpy.types.Object:
    """Set up solid-color world, camera, and lights. Returns the camera object."""
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

    # Outdoor HDRI background
    hdri_path = POLYHAVEN_DIR / "outdoor" / "netball_court_2k.hdr"
    if hdri_path.exists():
        env_tex = world_nodes.new(type="ShaderNodeTexEnvironment")
        env_tex.location = (-300, 0)
        hdri_img = load_image(hdri_path, "Non-Color")
        hdri_img.colorspace_settings.name = "Linear"
        env_tex.image = hdri_img

        mapping_node = world_nodes.new(type="ShaderNodeMapping")
        mapping_node.location = (-550, 0)
        mapping_node.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(90))
        tex_coord_node = world_nodes.new(type="ShaderNodeTexCoord")
        tex_coord_node.location = (-750, 0)
        world_links.new(tex_coord_node.outputs["Generated"], mapping_node.inputs["Vector"])
        world_links.new(mapping_node.outputs["Vector"], env_tex.inputs["Vector"])

        bg_node = world_nodes.new(type="ShaderNodeBackground")
        bg_node.location = (0, 0)
        bg_node.inputs["Strength"].default_value = 0.5
        world_links.new(env_tex.outputs["Color"], bg_node.inputs["Color"])

        output_node = world_nodes.new(type="ShaderNodeOutputWorld")
        output_node.location = (200, 0)
        world_links.new(bg_node.outputs["Background"], output_node.inputs["Surface"])
        print(f"[INFO] Using HDRI: {hdri_path.name}")
    else:
        print(f"[WARN] HDRI not found: {hdri_path}, using solid fallback")
        bg_node = world_nodes.new(type="ShaderNodeBackground")
        bg_node.location = (0, 0)
        bg_node.inputs["Color"].default_value = (0.40, 0.55, 0.75, 1.0)
        bg_node.inputs["Strength"].default_value = 1.0
        output_node = world_nodes.new(type="ShaderNodeOutputWorld")
        output_node.location = (200, 0)
        world_links.new(bg_node.outputs["Background"], output_node.inputs["Surface"])

    # Camera
    camera_cfg = scenario["camera"]
    assert isinstance(camera_cfg, dict)
    camera_location = tuple(camera_cfg.get("location", CAMERA_LOCATION))
    camera_target = tuple(camera_cfg.get("target", CAMERA_TARGET))

    bpy.ops.object.camera_add(location=camera_location)
    camera = bpy.context.object
    camera.name = "render_camera"
    camera.data.lens = float(camera_cfg.get("lens_mm", CAMERA_LENS_MM))
    scene.camera = camera

    target = bpy.data.objects.new("camera_target", None)
    scene.collection.objects.link(target)
    target.location = camera_target
    constraint = camera.constraints.new("TRACK_TO")
    constraint.target = target
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"

    # Sun light (outdoor feel)
    bpy.ops.object.light_add(type="SUN", location=(5.0, -5.0, 8.0))
    sun = bpy.context.object
    sun.data.energy = 0.6
    sun.rotation_euler = (math.radians(50), math.radians(10), math.radians(35))
    sun.data.angle = math.radians(2)

    # Fill light
    bpy.ops.object.light_add(type="AREA", location=(-3.0, 3.0, 4.0))
    fill_light = bpy.context.object
    fill_light.data.energy = 40
    fill_light.data.size = 4.0

    return camera


def build_scene(
    args: argparse.Namespace,
    scenario: dict[str, object],
) -> tuple[bpy.types.Object, bpy.types.Object]:
    """Build the tennis scene. Returns (tennis_ball, camera)."""
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

    # Import models
    import_tennis_court()
    tennis_ball = import_tennis_ball()

    # The GLB may be in arbitrary units; enforce real-world tennis ball size.
    TARGET_BALL_RADIUS = 0.080  # enlarged tennis ball (12 cm diameter) for better visibility
    current_radius = max(tennis_ball.dimensions) / 2.0
    scale_factor = TARGET_BALL_RADIUS / current_radius if current_radius > 0 else 1.0
    tennis_ball.scale = (scale_factor,) * 3
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    print(f"[INFO] Scaled tennis ball from radius {current_radius:.4f} to {TARGET_BALL_RADIUS:.4f}")

    # Update physics scenario with the target ball radius
    physics = scenario["physics"]
    assert isinstance(physics, dict)
    physics["ball_radius"] = TARGET_BALL_RADIUS

    # Setup world, camera, lights
    camera = setup_world_and_lights(scenario)

    # Position the ball at its launch location
    launch_loc = physics.get("launch_location", [-8.0, 0.0, 5.0])
    tennis_ball.location = (
        float(launch_loc[0]),
        float(launch_loc[1]),
        float(launch_loc[2]),
    )
    tennis_ball.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)

    frame_end = max(2, int(round(float(args.duration_sec) * int(args.fps))))
    scene.frame_start = 1
    scene.frame_end = frame_end

    return tennis_ball, camera


def apply_physics_animation(
    tennis_ball: bpy.types.Object,
    physics: dict,
) -> None:
    tennis_ball.rotation_mode = "QUATERNION"

    for frame_record in physics["frames"]:
        frame = int(frame_record["frame_index"])

        ball_quat = frame_record["ball_quaternion_xyzw"]
        tennis_ball.location = frame_record["ball_location"]
        tennis_ball.rotation_quaternion = (
            ball_quat[3],
            ball_quat[0],
            ball_quat[1],
            ball_quat[2],
        )
        tennis_ball.keyframe_insert(data_path="location", frame=frame)
        tennis_ball.keyframe_insert(data_path="rotation_quaternion", frame=frame)

    set_linear_keyframes([tennis_ball])


def export_ground_truth(
    out_dir: Path,
    tennis_ball: bpy.types.Object,
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
            "tennis_ball": {
                "object_name": tennis_ball.name,
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
                "tennis_ball_matrix_world": [[float(v) for v in row] for row in tennis_ball.matrix_world],
                "tennis_ball_location": [float(v) for v in tennis_ball.location],
                "tennis_ball_linear_velocity": physics_frame["ball_linear_velocity"],
                "tennis_ball_angular_velocity": physics_frame["ball_angular_velocity"],
                "tennis_ball_floor_gap": physics_frame["ball_floor_gap"],
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
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_scenario_metadata(out_dir, scenario)

    tennis_ball, camera = build_scene(args, scenario)
    physics = run_physics_simulation(args, scenario)
    apply_physics_animation(tennis_ball, physics)
    export_ground_truth(
        out_dir,
        tennis_ball,
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