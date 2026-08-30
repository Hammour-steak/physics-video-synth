from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path

import bpy
from mathutils import Vector

WORKSPACE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = WORKSPACE_DIR / "assets" / "models"
POLYHAVEN_DIR = WORKSPACE_DIR / "assets" / "polyhaven"
AMBIENTCG_DIR = WORKSPACE_DIR / "assets" / "ambientcg"

CAR_GLB = MODELS_DIR / "nissan_gtr-35_lbworks.glb"

OUTPUT_STEM = "car_gap_jump"
PHYSICS_TEMP = "physics_transforms.json"
GROUND_TRUTH_NAME = "ground_truth_transforms.json"
SCENARIO_METADATA_NAME = "scenario_metadata.json"

# --- Geometry (must match simulate_car_gap_jump.py) --------------------------
CAR_HALF_LENGTH = 2.35   # along travel (+X)
CAR_HALF_WIDTH = 0.95
CAR_HALF_HEIGHT = 0.67
CAR_TARGET_LENGTH = 2.0 * CAR_HALF_LENGTH  # 4.70 m real GT-R length

DECK_TOP_Z = 0.0
DECK_THICKNESS = 0.6
DECK_WIDTH = 5.0
RAMP_ANGLE_DEG = 12.0
RAMP_LENGTH = 13.0
APPROACH_RUN = 12.0
FAR_LEN = 45.0
CHASM_DEPTH = 8.0

CAMERA_LENS_MM = 40.0


def ramp_geometry() -> dict:
    angle = math.radians(RAMP_ANGLE_DEG)
    run = RAMP_LENGTH * math.cos(angle)
    rise = RAMP_LENGTH * math.sin(angle)
    mid_top = (-run / 2.0, 0.0, rise / 2.0)
    normal = (-math.sin(angle), 0.0, math.cos(angle))
    center = tuple(mid_top[i] - (DECK_THICKNESS / 2.0) * normal[i] for i in range(3))
    return {"angle": angle, "run": run, "rise": rise, "center": center}


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preview", "animation"), default="animation")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resolution", nargs=2, type=int, default=(1280, 720))
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-sec", type=float, default=4.5)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--preview-frame", type=int, default=40)
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--launch-speed", type=float, default=12.0)
    parser.add_argument("--gap-width", type=float, default=8.0)
    return parser.parse_args(argv)


# --- Materials ---------------------------------------------------------------

def set_input_default(node, name, value) -> None:
    socket = node.inputs.get(name)
    if socket is not None:
        socket.default_value = value


def require_path(base_dir: Path, filename: str) -> Path:
    file_path = base_dir / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Texture missing: {file_path}")
    return file_path


def create_asphalt_material() -> bpy.types.Material:
    mat = bpy.data.materials.new("road_asphalt")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    set_input_default(bsdf, "Roughness", 0.9)
    set_input_default(bsdf, "Metallic", 0.0)

    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    mapping = nodes.new(type="ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (0.4, 0.4, 0.4)
    links.new(tex_coord.outputs["Object"], mapping.inputs["Vector"])

    asset_dir = AMBIENTCG_DIR / "Asphalt031"
    try:
        def img(fname, cs):
            image = bpy.data.images.load(str(require_path(asset_dir, fname)), check_existing=True)
            image.colorspace_settings.name = cs
            tex = nodes.new(type="ShaderNodeTexImage")
            tex.image = image
            tex.projection = "BOX"
            tex.projection_blend = 0.3
            links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
            return tex

        color = img("Asphalt031_4K-JPG_Color.jpg", "sRGB")
        # The bare scan is a pale concrete-grey; darken and add contrast so the
        # road reads as dark asphalt against the bright sky rather than washing
        # out to near-white.
        contrast = nodes.new(type="ShaderNodeBrightContrast")
        contrast.inputs["Bright"].default_value = -0.12
        contrast.inputs["Contrast"].default_value = 0.2
        links.new(color.outputs["Color"], contrast.inputs["Color"])
        darken = nodes.new(type="ShaderNodeMixRGB")
        darken.blend_type = "MULTIPLY"
        darken.inputs["Fac"].default_value = 1.0
        darken.inputs["Color2"].default_value = (0.55, 0.55, 0.58, 1.0)
        links.new(contrast.outputs["Color"], darken.inputs["Color1"])
        links.new(darken.outputs["Color"], bsdf.inputs["Base Color"])
        rough = img("Asphalt031_4K-JPG_Roughness.jpg", "Non-Color")
        links.new(rough.outputs["Color"], bsdf.inputs["Roughness"])
        nor = img("Asphalt031_4K-JPG_NormalGL.jpg", "Non-Color")
        nmap = nodes.new(type="ShaderNodeNormalMap")
        nmap.inputs["Strength"].default_value = 1.0
        links.new(nor.outputs["Color"], nmap.inputs["Color"])
        links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
    except FileNotFoundError:
        set_input_default(bsdf, "Base Color", (0.05, 0.05, 0.055, 1.0))
    return mat


def create_rock_material() -> bpy.types.Material:
    # Procedural weathered rock/dirt for the chasm walls and floor: a
    # noise-driven brown-grey with a coarse bump so the canyon under the
    # broken bridge doesn't read as a flat grey box.
    mat = bpy.data.materials.new("chasm_rock")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    set_input_default(bsdf, "Roughness", 1.0)
    set_input_default(bsdf, "Metallic", 0.0)

    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    mapping = nodes.new(type="ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (0.15, 0.15, 0.15)
    links.new(tex_coord.outputs["Object"], mapping.inputs["Vector"])

    noise = nodes.new(type="ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 4.0
    noise.inputs["Detail"].default_value = 8.0
    noise.inputs["Roughness"].default_value = 0.7
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])

    ramp = nodes.new(type="ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.06, 0.05, 0.04, 1.0)
    ramp.color_ramp.elements[1].color = (0.20, 0.16, 0.12, 1.0)
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    bump = nodes.new(type="ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.5
    bump.inputs["Distance"].default_value = 0.1
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


# --- Mesh builders -----------------------------------------------------------

def add_box(name, center, dims, material, rotation_euler=(0.0, 0.0, 0.0)) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=center, rotation=rotation_euler)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dims
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if material is not None:
        obj.data.materials.append(material)
    return obj


# --- Car import --------------------------------------------------------------

def world_bbox(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    return Vector((min(xs), min(ys), min(zs))), Vector((max(xs), max(ys), max(zs)))


def import_car() -> bpy.types.Object:
    if not CAR_GLB.exists():
        raise FileNotFoundError(f"Car model not found: {CAR_GLB}")
    existing = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(CAR_GLB))
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

    # The raw model's length runs along its local Y. Yaw -90 deg about Z so the
    # length (nose->tail) aligns with world X, the travel direction.
    obj.rotation_mode = "XYZ"
    obj.rotation_euler = (0.0, 0.0, math.radians(-90.0))
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    # Uniformly rescale so the length (now along X) matches the real GT-R.
    mn, mx = world_bbox(obj)
    raw_length = mx.x - mn.x
    scale = CAR_TARGET_LENGTH / raw_length
    obj.scale = (scale, scale, scale)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    # Origin at the geometric center so the per-frame physics quaternion (which
    # is about the box center) rotates the mesh about its center.
    mn, mx = world_bbox(obj)
    bpy.context.scene.cursor.location = ((mn.x + mx.x) / 2.0, (mn.y + mx.y) / 2.0, (mn.z + mx.z) / 2.0)
    bpy.ops.object.origin_set(type="ORIGIN_CURSOR")
    obj.location = (0.0, 0.0, 0.0)
    obj.name = "car"
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
    return obj


# --- Physics ------------------------------------------------------------------

def run_physics(args: argparse.Namespace) -> dict:
    physics_python = WORKSPACE_DIR.parent / "miniconda" / "envs" / "physics" / "bin" / "python"
    python = str(physics_python) if physics_python.exists() else (shutil.which("python3") or shutil.which("python"))
    if not python:
        raise RuntimeError("Cannot find python3/python for the PyBullet physics simulation.")
    script = Path(__file__).with_name("simulate_car_gap_jump.py")
    out = args.out_dir / PHYSICS_TEMP
    subprocess.run(
        [
            python, str(script),
            "--out", str(out),
            "--fps", str(int(args.fps)),
            "--duration-sec", str(float(args.duration_sec)),
            "--launch-speed", str(float(args.launch_speed)),
            "--gap-width", str(float(args.gap_width)),
        ],
        check=True,
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    out.unlink(missing_ok=True)
    return data


def apply_keyframes(obj: bpy.types.Object, physics: dict) -> None:
    obj.rotation_mode = "QUATERNION"
    for fr in physics["frames"]:
        d = fr["car"]
        obj.location = d["location"]
        q = d["quaternion_xyzw"]
        obj.rotation_quaternion = (q[3], q[0], q[1], q[2])
        obj.keyframe_insert(data_path="location", frame=int(fr["frame_index"]))
        obj.keyframe_insert(data_path="rotation_quaternion", frame=int(fr["frame_index"]))
    if obj.animation_data and obj.animation_data.action:
        for fc in obj.animation_data.action.fcurves:
            for k in fc.keyframe_points:
                k.interpolation = "LINEAR"


# --- Scene -------------------------------------------------------------------

def look_at(obj, target) -> None:
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def build_scene(args: argparse.Namespace, physics: dict) -> tuple[bpy.types.Object, bpy.types.Object]:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.resolution_x = args.resolution[0]
    scene.render.resolution_y = args.resolution[1]
    scene.render.fps = args.fps
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.samples
    scene.cycles.device = "GPU" if args.device == "auto" else "CPU"
    scene.cycles.max_bounces = 8
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.view_settings.exposure = -0.4

    gap_width = float(args.gap_width)

    # World: clear-sky HDRI.
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    wn = world.node_tree.nodes
    wl = world.node_tree.links
    for n in list(wn):
        wn.remove(n)
    tc = wn.new(type="ShaderNodeTexCoord")
    mp = wn.new(type="ShaderNodeMapping")
    mp.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(120.0))
    wl.new(tc.outputs["Generated"], mp.inputs["Vector"])
    env = wn.new(type="ShaderNodeTexEnvironment")
    bg = wn.new(type="ShaderNodeBackground")
    out = wn.new(type="ShaderNodeOutputWorld")
    hdri = POLYHAVEN_DIR / "outdoor" / "syferfontein_0d_clear_puresky_2k.hdr"
    if hdri.exists():
        env.image = bpy.data.images.load(str(hdri), check_existing=True)
        wl.new(mp.outputs["Vector"], env.inputs["Vector"])
        wl.new(env.outputs["Color"], bg.inputs["Color"])
    else:
        bg.inputs["Color"].default_value = (0.45, 0.62, 0.85, 1.0)
    bg.inputs["Strength"].default_value = 1.1
    wl.new(bg.outputs["Background"], out.inputs["Surface"])

    # Sun.
    bpy.ops.object.light_add(type="SUN", location=(5.0, -8.0, 20.0))
    sun = bpy.context.object
    sun.data.energy = 4.5
    sun.data.angle = math.radians(2.0)
    sun.data.color = (1.0, 0.97, 0.9)
    sun.rotation_euler = (math.radians(50), math.radians(8), math.radians(-40))

    # Materials.
    asphalt = create_asphalt_material()
    rock = create_rock_material()

    ramp = ramp_geometry()
    run = ramp["run"]

    # Approach deck.
    approach_center_x = -(run + APPROACH_RUN / 2.0)
    add_box("approach_deck", (approach_center_x, 0.0, -DECK_THICKNESS / 2.0),
            (APPROACH_RUN, DECK_WIDTH, DECK_THICKNESS), asphalt)
    # Take-off ramp.
    add_box("takeoff_ramp", ramp["center"], (RAMP_LENGTH, DECK_WIDTH, DECK_THICKNESS),
            asphalt, rotation_euler=(0.0, -ramp["angle"], 0.0))
    # Far landing deck.
    far_center_x = gap_width + FAR_LEN / 2.0
    add_box("far_deck", (far_center_x, 0.0, -DECK_THICKNESS / 2.0),
            (FAR_LEN, DECK_WIDTH, DECK_THICKNESS), asphalt)

    # Chasm floor (rock) far below the decks, so a short jump visibly plummets
    # into a canyon. A wide floor slab reads as the canyon bottom under the
    # gap; the sky fills the rest of the background (no back wall).
    total_span_x = (run + APPROACH_RUN) + gap_width + FAR_LEN
    span_center_x = -(run + APPROACH_RUN) + total_span_x / 2.0
    add_box("chasm_floor", (span_center_x, 0.0, -CHASM_DEPTH - 1.0),
            (total_span_x + 60.0, 60.0, 2.0), rock)

    # Camera: low side view at roughly deck height, looking along +Y nearly
    # horizontally. The road decks read as a horizontal band low in frame, the
    # gap as a break showing distant sky/canyon through it, and the car arcs
    # above the band against the sky -- the iconic gap-jump silhouette.
    action_center_x = gap_width * 0.5 + 6.0
    cam_loc = (action_center_x - 6.0, -33.0, 2.8)
    cam_tgt = (action_center_x + 2.0, 0.0, 2.6)
    bpy.ops.object.camera_add(location=cam_loc)
    camera = bpy.context.object
    camera.data.lens = CAMERA_LENS_MM
    camera.data.sensor_width = 36.0
    scene.camera = camera
    tgt = bpy.data.objects.new("camera_target", None)
    scene.collection.objects.link(tgt)
    tgt.location = cam_tgt
    con = camera.constraints.new("TRACK_TO")
    con.target = tgt
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"

    # Car with physics animation.
    car = import_car()
    apply_keyframes(car, physics)

    frame_end = int(physics["frame_end"])
    scene.frame_start = 1
    scene.frame_end = frame_end
    scene.frame_set(1)
    return car, camera


def export_ground_truth(out_dir: Path, car, camera, physics: dict, scenario: dict) -> None:
    scene = bpy.context.scene
    fps = int(physics["fps"])
    frame_end = int(physics["frame_end"])
    records = {
        "schema_version": 1,
        "fps": fps,
        "frame_start": 1,
        "frame_end": frame_end,
        "physics": {k: v for k, v in physics.items() if k != "frames"},
        "objects": {"car": {"object_name": car.name}},
        "camera": {
            "object_name": camera.name,
            "lens_mm": float(camera.data.lens),
            "sensor_width_mm": float(camera.data.sensor_width),
            "resolution": [int(scene.render.resolution_x), int(scene.render.resolution_y)],
        },
        "scenario": scenario,
        "frames": [],
    }
    by_frame = {int(fr["frame_index"]): fr for fr in physics["frames"]}
    for frame in range(1, frame_end + 1):
        scene.frame_set(frame)
        pf = by_frame[frame]
        records["frames"].append({
            "frame_index": frame,
            "time_sec": (frame - 1) / float(fps),
            "car": {
                "matrix_world": [[float(v) for v in row] for row in car.matrix_world],
                "linear_velocity": pf["car"]["linear_velocity"],
                "angular_velocity": pf["car"]["angular_velocity"],
            },
            "camera_matrix_world": [[float(v) for v in row] for row in camera.matrix_world],
        })
    (out_dir / GROUND_TRUTH_NAME).write_text(json.dumps(records, indent=2), encoding="utf-8")


def render_preview(args: argparse.Namespace) -> None:
    scene = bpy.context.scene
    scene.frame_set(max(scene.frame_start, min(int(args.preview_frame), scene.frame_end)))
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

    scenario = {
        "schema_version": 1,
        "seed": int(args.seed),
        "render": {
            "fps": int(args.fps),
            "duration_sec": float(args.duration_sec),
            "resolution": [int(args.resolution[0]), int(args.resolution[1])],
            "samples": int(args.samples),
            "device": str(args.device),
            "mode": str(args.mode),
        },
        "physics_params": {
            "launch_speed": float(args.launch_speed),
            "gap_width": float(args.gap_width),
        },
    }
    (args.out_dir / SCENARIO_METADATA_NAME).write_text(json.dumps(scenario, indent=2), encoding="utf-8")

    physics = run_physics(args)
    car, camera = build_scene(args, physics)
    export_ground_truth(args.out_dir, car, camera, physics, scenario)

    if args.mode == "preview":
        render_preview(args)
    else:
        render_animation(args)

    bpy.ops.wm.save_as_mainfile(filepath=str(args.out_dir / f"{OUTPUT_STEM}.blend"))
    print(f"[INFO] Render complete. Output: {args.out_dir}")


if __name__ == "__main__":
    main()
