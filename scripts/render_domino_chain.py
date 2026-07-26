from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import bpy


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import render_ball_block_impact as scene_base  # noqa: E402


OUTPUT_STEM = "domino_chain"
DIRECT_MP4_NAME = f"{OUTPUT_STEM}_blender_direct.mp4"
FINAL_MP4_NAME = f"{OUTPUT_STEM}.mp4"
TEMP_MP4_NAME = f"{OUTPUT_STEM}_tmp.mp4"
BLEND_NAME = f"{OUTPUT_STEM}.blend"
GROUND_TRUTH_NAME = "ground_truth_transforms.json"
PHYSICS_TEMP_NAME = "domino_physics_transforms.json"
SCENARIO_METADATA_NAME = "scenario_metadata.json"
DOMINO_DIMENSIONS = (0.18, 0.58, 1.18)
DOMINO_NAMES = ("domino_1", "domino_2", "domino_3")


def _configure_output_names() -> None:
    scene_base.OUTPUT_STEM = OUTPUT_STEM
    scene_base.DIRECT_MP4_NAME = DIRECT_MP4_NAME
    scene_base.FINAL_MP4_NAME = FINAL_MP4_NAME
    scene_base.TEMP_MP4_NAME = TEMP_MP4_NAME
    scene_base.BLEND_NAME = BLEND_NAME
    scene_base.GROUND_TRUTH_NAME = GROUND_TRUTH_NAME
    scene_base.PHYSICS_TEMP_NAME = PHYSICS_TEMP_NAME
    scene_base.SCENARIO_METADATA_NAME = SCENARIO_METADATA_NAME


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preview", "animation"), default="animation")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resolution", nargs=2, type=int, default=(1280, 720))
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-sec", type=float, default=8.0)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--preview-frame", type=int, default=40)
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=3501)
    parser.add_argument("--spacing", type=float, default=0.68)
    parser.add_argument("--first-tilt-deg", type=float, default=12.0)
    parser.add_argument("--block-texture-asset", choices=tuple(scene_base.BLOCK_TEXTURES), default="wood_table")
    parser.add_argument("--surface-marks", choices=("none", "subtle", "full"), default="none")
    parser.add_argument("--camera-jitter", type=float, default=0.0)
    parser.add_argument("--video-postprocess", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def create_scenario(args: argparse.Namespace) -> dict[str, object]:
    template_args = copy.copy(args)
    template_args.motion = "side_impact"
    template_args.physics_jitter = 0.0
    template_args.drop_x_velocity = None
    template_args.drop_y_velocity = None
    template_args.scenario_json = None
    template_args.scenario_overrides_json = None
    scenario = scene_base.create_scenario(template_args)
    scenario["motion"] = "domino_chain"
    scenario["active_objects"] = list(DOMINO_NAMES)
    scenario["physics"] = {
        "motion": "domino_chain",
        "gravity": [0.0, 0.0, -9.81],
        "domino_count": 3,
        "domino_dimensions": list(DOMINO_DIMENSIONS),
        "domino_spacing": float(args.spacing),
        "first_domino_tilt_deg": float(args.first_tilt_deg),
        "initial_linear_velocity": [0.0, 0.0, 0.0],
        "initial_angular_velocity": [0.0, 0.0, 0.0],
        "external_impulse_applied": False,
        "domino_mass": 0.36,
        "floor_friction": 0.90,
        "domino_friction": 0.90,
        "domino_restitution": 0.04,
        "ramp_enabled": False,
        "wall_enabled": False,
    }
    camera_location = (3.45, -7.20, 2.62)
    camera_target = (0.18, 0.0, 0.55)
    focus_distance = math.dist(camera_location, camera_target)
    scenario["camera"] = {
        "base_location": list(camera_location),
        "target": list(camera_target),
        "lens_mm": 52.0,
        "sensor_width_mm": 32.0,
        "focus_distance": float(focus_distance),
        "aperture_fstop": 8.0,
    }
    render = scenario["render"]
    assert isinstance(render, dict)
    render["motion_blur_shutter"] = 0.24
    return scenario


def _pip_positions(count: int) -> tuple[tuple[float, float], ...]:
    offset_x = 0.048
    offset_z = 0.24
    if count == 1:
        return ((0.0, 0.0),)
    if count == 2:
        return ((-offset_x, offset_z), (offset_x, -offset_z))
    return ((-offset_x, offset_z), (0.0, 0.0), (offset_x, -offset_z))


def _add_pips(domino: bpy.types.Object, count: int) -> None:
    pip_material = scene_base.create_principled_material(
        f"domino {count} inset dark pips",
        (0.025, 0.022, 0.019, 1.0),
        roughness=0.72,
    )
    half_width = 0.5 * float(DOMINO_DIMENSIONS[1])
    for pip_index, (local_x, local_z) in enumerate(_pip_positions(count), start=1):
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=48,
            radius=0.048,
            depth=0.010,
            location=(0.0, 0.0, 0.0),
            rotation=(0.5 * math.pi, 0.0, 0.0),
        )
        pip = bpy.context.object
        pip.name = f"{domino.name}_pip_{pip_index}"
        pip.data.materials.append(pip_material)
        pip.parent = domino
        pip.matrix_parent_inverse.identity()
        pip.location = (float(local_x), -half_width - 0.006, float(local_z))
        pip.rotation_euler = (0.5 * math.pi, 0.0, 0.0)


def add_dominoes(
    scenario: dict[str, object],
    physics: dict[str, Any],
) -> list[bpy.types.Object]:
    first_frame = physics["frames"][0]
    dominoes: list[bpy.types.Object] = []
    for index, name in enumerate(DOMINO_NAMES, start=1):
        material = scene_base.create_wood_material(scenario)
        location = tuple(float(value) for value in first_frame[f"{name}_location"])
        quaternion_xyzw = first_frame[f"{name}_quaternion_xyzw"]
        domino = scene_base.add_box(
            name,
            location,
            DOMINO_DIMENSIONS,
            material,
            bevel_width=0.012,
        )
        scene_base.cube_project_uvs(domino, cube_size=0.72)
        domino.rotation_mode = "QUATERNION"
        domino.rotation_quaternion = (
            float(quaternion_xyzw[3]),
            float(quaternion_xyzw[0]),
            float(quaternion_xyzw[1]),
            float(quaternion_xyzw[2]),
        )
        domino.pass_index = int(index)
        _add_pips(domino, index)
        dominoes.append(domino)
    return dominoes


def run_physics_simulation(
    args: argparse.Namespace,
    scenario: dict[str, object],
) -> dict[str, Any]:
    python = shutil.which("python3") or shutil.which("python")
    if python is None:
        raise RuntimeError("Cannot find python3/python for the PyBullet simulation.")
    physics_path = args.out_dir / PHYSICS_TEMP_NAME
    subprocess.run(
        [
            python,
            str(SCRIPT_DIR / "simulate_domino_chain.py"),
            "--out",
            str(physics_path),
            "--fps",
            str(int(args.fps)),
            "--duration-sec",
            str(float(args.duration_sec)),
            "--spacing",
            str(float(args.spacing)),
            "--first-tilt-deg",
            str(float(args.first_tilt_deg)),
        ],
        check=True,
    )
    physics = json.loads(physics_path.read_text(encoding="utf-8"))
    physics_path.unlink(missing_ok=True)
    if not bool(physics.get("summary", {}).get("ordered_chain_complete")):
        raise RuntimeError("Domino simulation did not produce the ordered two-contact chain.")
    scenario["contact_events"] = copy.deepcopy(physics["contact_events"])
    return physics


def apply_physics_animation(
    dominoes: list[bpy.types.Object],
    physics: dict[str, Any],
) -> None:
    for domino in dominoes:
        domino.rotation_mode = "QUATERNION"
    for frame_record in physics["frames"]:
        frame = int(frame_record["frame_index"])
        for domino, name in zip(dominoes, DOMINO_NAMES, strict=True):
            quaternion_xyzw = frame_record[f"{name}_quaternion_xyzw"]
            domino.location = frame_record[f"{name}_location"]
            domino.rotation_quaternion = (
                float(quaternion_xyzw[3]),
                float(quaternion_xyzw[0]),
                float(quaternion_xyzw[1]),
                float(quaternion_xyzw[2]),
            )
            domino.keyframe_insert(data_path="location", frame=frame)
            domino.keyframe_insert(data_path="rotation_quaternion", frame=frame)
    scene_base.set_linear_keyframes(dominoes)


def export_ground_truth(
    out_dir: Path,
    dominoes: list[bpy.types.Object],
    camera: bpy.types.Object,
    physics: dict[str, Any],
    scenario: dict[str, object],
    *,
    fps: int,
) -> None:
    scene = bpy.context.scene
    records: dict[str, Any] = {
        "schema_version": 1,
        "fps": int(fps),
        "frame_start": int(scene.frame_start),
        "frame_end": int(scene.frame_end),
        "scenario_metadata_path": str((out_dir / SCENARIO_METADATA_NAME).resolve()),
        "physics": {
            key: value
            for key, value in physics.items()
            if key != "frames"
        },
        "objects": {
            name: {
                "object_name": name,
                "label": f"wooden domino {index}",
                "dimensions_scene_units": list(DOMINO_DIMENSIONS),
                "instance_id": int(index),
            }
            for index, name in enumerate(DOMINO_NAMES, start=1)
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
            "realism_profile": scenario["realism_profile"],
            "motion": "domino_chain",
        },
        "static_scene_surfaces": [scene_base.floor_surface_metadata()],
        "contact_events": copy.deepcopy(physics["contact_events"]),
        "frames": [],
    }
    physics_by_frame = {
        int(frame_record["frame_index"]): frame_record
        for frame_record in physics["frames"]
    }
    for frame in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(frame)
        physics_frame = physics_by_frame[int(frame)]
        object_states: dict[str, Any] = {}
        frame_record: dict[str, Any] = {
            "frame_index": int(frame),
            "time_sec": float(frame - scene.frame_start) / float(fps),
            "camera_matrix_world": [
                [float(value) for value in row]
                for row in camera.matrix_world
            ],
            "camera_world_to_camera_matrix": [
                [float(value) for value in row]
                for row in camera.matrix_world.inverted()
            ],
            "object_states": object_states,
            "domino_1_domino_2_contact": bool(
                physics_frame["domino_1_domino_2_contact"]
            ),
            "domino_2_domino_3_contact": bool(
                physics_frame["domino_2_domino_3_contact"]
            ),
        }
        for domino, name in zip(dominoes, DOMINO_NAMES, strict=True):
            state = {
                "matrix_world": [
                    [float(value) for value in row]
                    for row in domino.matrix_world
                ],
                "location": [float(value) for value in domino.location],
                "quaternion_xyzw": physics_frame[f"{name}_quaternion_xyzw"],
                "linear_velocity": physics_frame[f"{name}_linear_velocity"],
                "angular_velocity": physics_frame[f"{name}_angular_velocity"],
                "floor_gap": physics_frame[f"{name}_floor_gap"],
                "tilt_deg": physics_frame[f"{name}_tilt_deg"],
            }
            object_states[name] = state
            frame_record[f"{name}_matrix_world"] = state["matrix_world"]
            frame_record[f"{name}_location"] = state["location"]
            frame_record[f"{name}_linear_velocity"] = state["linear_velocity"]
            frame_record[f"{name}_angular_velocity"] = state["angular_velocity"]
        records["frames"].append(frame_record)
    (out_dir / GROUND_TRUTH_NAME).write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )


def build_scene(
    args: argparse.Namespace,
    scenario: dict[str, object],
) -> None:
    _configure_output_names()
    scene_base.clear_scene()
    scene_base.setup_render(args, scenario)
    physics = run_physics_simulation(args, scenario)
    scene_base.write_scenario_metadata(args.out_dir, scenario)
    scene_base.add_environment(scenario)
    camera = scene_base.add_camera(scenario)
    dominoes = add_dominoes(scenario, physics)
    apply_physics_animation(dominoes, physics)
    export_ground_truth(
        args.out_dir,
        dominoes,
        camera,
        physics,
        scenario,
        fps=int(args.fps),
    )
    bpy.ops.wm.save_as_mainfile(filepath=str((args.out_dir / BLEND_NAME).resolve()))


def render_preview(args: argparse.Namespace) -> None:
    scene = bpy.context.scene
    frame = max(scene.frame_start, min(int(args.preview_frame), scene.frame_end))
    scene.frame_set(frame)
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str((args.out_dir / f"preview_frame_{frame:05d}.png").resolve())
    bpy.ops.render.render(write_still=True)


def render_animation(args: argparse.Namespace, scenario: dict[str, object]) -> None:
    scene = bpy.context.scene
    scene.frame_set(scene.frame_start)
    scene_base.configure_ffmpeg(scene, args.out_dir)
    bpy.ops.render.render(animation=True)
    scene_base.write_compatible_mp4(args.out_dir, int(args.fps), scenario)
    shutil.copy2(args.out_dir / FINAL_MP4_NAME, args.out_dir / "video.mp4")


def main() -> None:
    args = parse_args()
    scenario = create_scenario(args)
    build_scene(args, scenario)
    if args.mode == "preview":
        render_preview(args)
        return
    render_animation(args, scenario)


if __name__ == "__main__":
    main()
