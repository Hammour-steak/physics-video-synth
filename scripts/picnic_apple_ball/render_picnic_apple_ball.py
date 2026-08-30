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

import bpy
from mathutils import Vector, noise

WORKSPACE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = WORKSPACE_DIR / "assets" / "models"
POLYHAVEN_DIR = WORKSPACE_DIR / "assets" / "polyhaven"

SOCCER_BALL_GLB = MODELS_DIR / "soccer_ball.glb"
APPLE_GLB = MODELS_DIR / "apple.glb"
PICNIC_GLB = MODELS_DIR / "french_picnic.glb"

OUTPUT_STEM = "picnic_apple_ball"
PHYSICS_TEMP = "physics_transforms.json"
GROUND_TRUTH_NAME = "ground_truth_transforms.json"
SCENARIO_METADATA_NAME = "scenario_metadata.json"

# Real-world target sizes. Matches simulate_picnic_apple_ball.py's collision
# proxy radii -- the visual mesh is rescaled to line up with the physics.
BALL_DIAMETER = 0.22   # FIFA size-5 ball
APPLE_DIAMETER = 0.075  # medium apple
PICNIC_BLANKET_WIDTH = 2.1  # a large family-size blanket spread

# The picnic blanket sits centered (x~0, matching the ball/apple action's own
# x~0) but well back in +Y, in the background of a camera that looks back
# along +Y toward it from behind the ball/apple action, which plays out in
# the foreground close to the camera.
PICNIC_LOCATION = (0.05, 1.55, 0.0)
PICNIC_YAW_DEG = -15.0

# A two-tier ground (dense blade patch + plain far extension) left a visible
# seam where the two met -- the boundary's straight polygon edge read as
# obviously artificial no matter how closely the far material's color was
# tuned to match. Using one single blade-covered patch, sized to reach past
# the edges of the camera's frame, avoids that seam entirely at the cost of
# more particles.
HERO_GROUND_SIZE = 22.0
GROUND_SUBDIV = 220

CAMERA_LOCATION = (0.0, -2.0, 1.5)
CAMERA_TARGET = (0.05, 0.75, -0.25)
CAMERA_LENS_MM = 28.0

GRASS_BASE_COLOR = (0.045, 0.075, 0.02, 1.0)
GRASS_BLADE_BASE_COLOR = (0.05, 0.16, 0.035, 1.0)
GRASS_BLADE_TIP_COLOR = (0.28, 0.42, 0.10, 1.0)


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
    parser.add_argument("--preview-frame", type=int, default=18)
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--grass-density", type=int, default=260000)
    parser.add_argument("--drop-height", type=float, default=1.3)
    parser.add_argument("--apple-offset-x", type=float, default=0.105)
    parser.add_argument(
        "--grass-friction",
        type=float,
        default=None,
        help="Ground lateral friction forwarded to the physics sim; controls how "
        "far the struck ball rolls before stopping. None keeps the sim default.",
    )
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

    The physics block below is the baseline this scene's PCVE suite edits
    and must stay in sync with both simulate_picnic_apple_ball.py's defaults
    and edit_vocab.BASELINE_PHYSICS.
    """
    scenario = {
        "schema_version": 2,
        "seed": int(args.seed),
        "environment": "outdoor_picnic",
        "render": {
            "fps": int(args.fps),
            "duration_sec": float(args.duration_sec),
            "resolution": [int(args.resolution[0]), int(args.resolution[1])],
            "samples": int(args.samples),
            "device": str(args.device),
            "mode": str(args.mode),
        },
        "physics": {
            "ball_mass": 0.43,
            "ball_friction": 0.25,
            "ball_rolling_friction": 0.015,
            "ball_spinning_friction": 0.01,
            "ball_restitution": 0.35,
            "apple_mass": 0.15,
            "apple_friction": 0.5,
            "apple_restitution": 0.25,
            # Two-slot presence list, in fixed order (apple, ball). A PCVE
            # DELETE edit writes 0 at the object's slot.
            "active": [1, 1],
            "grass_friction": (0.25 if args.grass_friction is None else float(args.grass_friction)),
            "drop_height": float(args.drop_height),
            "apple_offset_x": float(args.apple_offset_x),
            "apple_offset_y": 0.0,
            "gravity_z": -9.8,
        },
    }
    if args.scenario_overrides_json is not None:
        scenario = recursive_update(scenario, read_json(args.scenario_overrides_json))
        scenario["scenario_overrides_path"] = str(
            args.scenario_overrides_json.expanduser().resolve()
        )
    return scenario


def world_bbox(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    return Vector((min(xs), min(ys), min(zs))), Vector((max(xs), max(ys), max(zs)))


def look_at(obj: bpy.types.Object, target) -> None:
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_area_light(name, loc, power, size, target, color=(1.0, 1.0, 1.0)) -> bpy.types.Object:
    bpy.ops.object.light_add(type="AREA", location=loc)
    light = bpy.context.object
    light.name = name
    light.data.energy = float(power)
    light.data.size = float(size)
    light.data.color = color
    light.visible_camera = False
    look_at(light, target)
    return light


# --- GLB import helpers ------------------------------------------------------

def import_glb_group(
    glb: Path, name: str, target_diameter: float,
    ref_axes: tuple[str, ...] = ("x", "y", "z"), origin: str = "bottom_center",
) -> bpy.types.Object:
    """Import a (possibly multi-mesh) GLB, join it into one object, and
    uniformly rescale it so the average of `ref_axes` bbox spans matches
    `target_diameter`.

    `origin` controls where the object's own origin (and thus its rotation
    pivot) ends up: "bottom_center" for static set-dressing placed by its
    ground-contact point (e.g. the picnic blanket), or "center" for a
    physics-driven rigid body -- simulate_picnic_apple_ball.py's PyBullet
    frames give sphere *center* positions plus a per-frame rotation
    quaternion, and a sphere rotated about anything other than its true
    center visibly orbits/wobbles as it spins (this is what caused the ball
    to dip below the grass -- it was rotating around a pivot at its bottom
    instead of its center).
    """
    if not glb.exists():
        raise FileNotFoundError(f"Model not found: {glb}")
    existing = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(glb))
    imported = [o for o in bpy.context.scene.objects if o not in existing]
    meshes = [o for o in imported if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh found in {glb}")
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

    mn, mx = world_bbox(obj)
    dims = {"x": mx.x - mn.x, "y": mx.y - mn.y, "z": mx.z - mn.z}
    raw_ref = sum(dims[a] for a in ref_axes) / len(ref_axes)
    scale = target_diameter / raw_ref
    obj.scale = (scale, scale, scale)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    mn, mx = world_bbox(obj)
    origin_z = mn.z if origin == "bottom_center" else (mn.z + mx.z) / 2.0
    bpy.context.scene.cursor.location = ((mn.x + mx.x) / 2.0, (mn.y + mx.y) / 2.0, origin_z)
    bpy.ops.object.origin_set(type="ORIGIN_CURSOR")
    obj.location = (0.0, 0.0, 0.0)
    obj.name = name
    obj.rotation_mode = "QUATERNION"
    return obj


def apply_keyframes(obj: bpy.types.Object, frames: list, key: str) -> None:
    obj.rotation_mode = "QUATERNION"
    for fr in frames:
        d = fr[key]
        obj.location = d["location"]
        q = d["quaternion_xyzw"]
        obj.rotation_quaternion = (q[3], q[0], q[1], q[2])
        obj.keyframe_insert(data_path="location", frame=int(fr["frame_index"]))
        obj.keyframe_insert(data_path="rotation_quaternion", frame=int(fr["frame_index"]))
    if obj.animation_data and obj.animation_data.action:
        for fc in obj.animation_data.action.fcurves:
            for k in fc.keyframe_points:
                k.interpolation = "LINEAR"


# --- Grass -------------------------------------------------------------------

def create_grass_blade_mesh() -> bpy.types.Object:
    # A slightly bent, tapered blade: 4 segments tall so it can curve, built
    # directly from vertex/face lists rather than a primitive -- a flat single
    # quad reads as a paper cutout, the taper + bend is what sells it as a
    # blade instead of a green card.
    height = 0.055
    base_width = 0.0035
    bend = 0.012
    segments = 4
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int, int]] = []
    for i in range(segments + 1):
        t = i / segments
        z = height * t
        x_bend = bend * (t ** 2)
        width = base_width * (1.0 - 0.88 * t)
        verts.append((-width / 2.0 + x_bend, 0.0, z))
        verts.append((width / 2.0 + x_bend, 0.0, z))
    for i in range(segments):
        a, b, c, d = 2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2
        faces.append((a, b, c, d))

    mesh = bpy.data.meshes.new("grass_blade_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("grass_blade", mesh)
    bpy.context.scene.collection.objects.link(obj)

    mat = bpy.data.materials.new("grass_blade_material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = 0.55
    if "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = 0.2

    geom = nodes.new(type="ShaderNodeNewGeometry")
    ramp = nodes.new(type="ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = GRASS_BLADE_BASE_COLOR
    ramp.color_ramp.elements[1].color = GRASS_BLADE_TIP_COLOR
    separate = nodes.new(type="ShaderNodeSeparateXYZ")
    links.new(geom.outputs["Position"], separate.inputs["Vector"])
    map_range = nodes.new(type="ShaderNodeMapRange")
    map_range.inputs["From Min"].default_value = 0.0
    map_range.inputs["From Max"].default_value = height
    links.new(separate.outputs["Z"], map_range.inputs["Value"])
    links.new(map_range.outputs["Result"], ramp.inputs["Fac"])

    part_info = nodes.new(type="ShaderNodeParticleInfo")
    hue_sat = nodes.new(type="ShaderNodeHueSaturation")
    map_range_hue = nodes.new(type="ShaderNodeMapRange")
    map_range_hue.inputs["From Min"].default_value = 0.0
    map_range_hue.inputs["From Max"].default_value = 1.0
    map_range_hue.inputs["To Min"].default_value = 0.42
    map_range_hue.inputs["To Max"].default_value = 0.58
    links.new(part_info.outputs["Random"], map_range_hue.inputs["Value"])
    links.new(map_range_hue.outputs["Result"], hue_sat.inputs["Hue"])
    map_range_val = nodes.new(type="ShaderNodeMapRange")
    map_range_val.inputs["From Min"].default_value = 0.0
    map_range_val.inputs["From Max"].default_value = 1.0
    map_range_val.inputs["To Min"].default_value = 0.75
    map_range_val.inputs["To Max"].default_value = 1.15
    links.new(part_info.outputs["Random"], map_range_val.inputs["Value"])
    links.new(map_range_val.outputs["Result"], hue_sat.inputs["Value"])
    links.new(ramp.outputs["Color"], hue_sat.inputs["Color"])

    # A uniform lawn of identical green reads as artificial turf -- real grass
    # always has a scatter of drier, yellow-brown blades mixed in. Reusing
    # the same per-instance random value (thresholded) picks a fixed ~12% of
    # blades to shift toward straw color instead of leaving every blade the
    # same live green.
    dry_threshold = nodes.new(type="ShaderNodeMath")
    dry_threshold.operation = "GREATER_THAN"
    dry_threshold.inputs[1].default_value = 0.88
    links.new(part_info.outputs["Random"], dry_threshold.inputs[0])
    dry_mix = nodes.new(type="ShaderNodeMixRGB")
    dry_mix.inputs["Color2"].default_value = (0.42, 0.34, 0.10, 1.0)
    links.new(dry_threshold.outputs["Value"], dry_mix.inputs["Fac"])
    links.new(hue_sat.outputs["Color"], dry_mix.inputs["Color1"])
    links.new(dry_mix.outputs["Color"], bsdf.inputs["Base Color"])

    # Slight translucency so backlit blades glow like real grass.
    if "Transmission" in bsdf.inputs:
        bsdf.inputs["Transmission"].default_value = 0.15
    mesh.materials.append(mat)
    return obj


def create_ground_material() -> bpy.types.Material:
    mat = bpy.data.materials.new("grass_ground_base")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = 0.95
    bsdf.inputs["Base Color"].default_value = GRASS_BASE_COLOR

    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    mapping = nodes.new(type="ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (25.0, 25.0, 25.0)
    links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])

    noise = nodes.new(type="ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 30.0
    noise.inputs["Detail"].default_value = 6.0
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])

    ramp = nodes.new(type="ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.03, 0.05, 0.015, 1.0)
    ramp.color_ramp.elements[1].color = (0.07, 0.10, 0.03, 1.0)
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    bump = nodes.new(type="ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.25
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def displace_ground_vertices(ground: bpy.types.Object) -> None:
    # Real turf is never a perfectly flat plane, and it matters here beyond
    # looks: hair particles align to the *surface normal* (rotation_mode
    # "NOR"), so a perfectly flat emitter makes every blade point exactly
    # straight up with only yaw varying -- that shared verticality is what
    # was reading as a "combed"/lattice-like repeat at this steep camera
    # angle even though blade positions are genuinely randomized (checked
    # directly by plotting particle locations). Gently bumping the actual
    # mesh (not a Displace modifier -- particle emission uses the base mesh)
    # varies local normals enough that blade tilt stops looking uniform.
    # The physics sim assumes a perfectly flat z=0 ground, so the ball/apple
    # path (roughly x in [-0.8, 0.3], y in [-0.3, 0.3]) is faded back to flat
    # to avoid the rigid bodies visibly floating over or sinking into a
    # bumped-up patch right where they're resting.
    mesh = ground.data
    for v in mesh.vertices:
        dx = max(0.0, -0.8 - v.co.x, v.co.x - 0.3)
        dy = max(0.0, abs(v.co.y) - 0.3)
        clear_dist = math.hypot(dx, dy)
        fade = min(1.0, clear_dist / 0.6)
        n1 = noise.noise((v.co.x * 1.6, v.co.y * 1.6, 0.0))
        n2 = noise.noise((v.co.x * 7.0, v.co.y * 7.0, 5.0))
        v.co.z += (n1 * 0.02 + n2 * 0.005) * fade
    mesh.update()
    for poly in mesh.polygons:
        poly.use_smooth = True


def create_grass_density_vertex_group(ground: bpy.types.Object) -> str:
    # Blades were being emitted under the picnic blanket too and poking up
    # through the thin cloth mesh (visible as faint green streaks across the
    # checkered pattern). Zero out particle density in a padded footprint
    # under the blanket -- matching PICNIC_LOCATION/PICNIC_YAW_DEG/
    # PICNIC_BLANKET_WIDTH -- so nothing grows there for the cloth to clip
    # through, with a short linear falloff at the boundary instead of a hard
    # cutoff.
    # The masked (zero-density) region stops just *inside* the blanket edge
    # rather than padding outside it: an outside pad left a visible bare
    # halo of blade-free ground ringing the blanket. Keeping the whole mask
    # (and its falloff) tucked under the cloth hides the transition, while
    # the blades growing right at the edge naturally lean over it.
    vg = ground.vertex_groups.new(name="grass_density")
    yaw = math.radians(PICNIC_YAW_DEG)
    cos_a, sin_a = math.cos(-yaw), math.sin(-yaw)
    half = (PICNIC_BLANKET_WIDTH / 2.0) * 0.97
    margin = 0.10
    cx, cy = PICNIC_LOCATION[0], PICNIC_LOCATION[1]
    for v in ground.data.vertices:
        dx = v.co.x - cx
        dy = v.co.y - cy
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        dist_outside = max(abs(lx) - half, abs(ly) - half)
        if dist_outside <= 0.0:
            weight = 0.0
        elif dist_outside >= margin:
            weight = 1.0
        else:
            weight = dist_outside / margin
        vg.add([v.index], weight, "REPLACE")
    return vg.name


def create_grass_ground(size: float, subdiv: int, density: int) -> bpy.types.Object:
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=subdiv, y_subdivisions=subdiv, size=size, location=(0.0, 0.0, 0.0),
    )
    ground = bpy.context.object
    ground.name = "grass_ground"
    ground.data.materials.append(create_ground_material())
    density_vg_name = create_grass_density_vertex_group(ground)

    blade = create_grass_blade_mesh()
    blade.hide_render = True
    blade.hide_viewport = True

    bpy.context.view_layer.objects.active = ground
    bpy.ops.object.particle_system_add()
    psys = ground.particle_systems[-1]
    psys.name = "grass"
    psys.vertex_group_density = density_vg_name
    settings = psys.settings
    settings.type = "HAIR"
    settings.use_advanced_hair = True
    settings.count = int(density)
    settings.hair_length = 1.0
    settings.render_type = "OBJECT"
    settings.instance_object = blade
    settings.particle_size = 1.0
    settings.size_random = 0.35
    settings.rotation_mode = "NOR"
    settings.rotation_factor_random = 0.6
    settings.phase_factor_random = 2.0
    settings.child_type = "INTERPOLATED"
    settings.rendered_child_count = 3
    settings.child_radius = 0.012
    settings.child_roundness = 0.6
    settings.use_modifier_stack = False

    return ground




# --- Physics ------------------------------------------------------------------

def run_physics(args: argparse.Namespace, scenario: dict) -> dict:
    physics_python = WORKSPACE_DIR.parent / "miniconda" / "envs" / "physics" / "bin" / "python"
    python = str(physics_python) if physics_python.exists() else (shutil.which("python3") or shutil.which("python"))
    if not python:
        raise RuntimeError("Cannot find python3/python for the PyBullet physics simulation.")
    script = Path(__file__).with_name("simulate_picnic_apple_ball.py")
    out = args.out_dir / PHYSICS_TEMP
    physics = scenario["physics"]
    command = [
        python, str(script),
        "--out", str(out),
        "--fps", str(int(args.fps)),
        "--duration-sec", str(float(args.duration_sec)),
        "--drop-height", str(float(physics["drop_height"])),
        "--apple-offset-x", str(float(physics["apple_offset_x"])),
        "--apple-offset-y", str(float(physics["apple_offset_y"])),
        "--grass-friction", str(float(physics["grass_friction"])),
        "--ball-mass", str(float(physics["ball_mass"])),
        "--ball-friction", str(float(physics["ball_friction"])),
        "--ball-rolling-friction", str(float(physics["ball_rolling_friction"])),
        "--ball-spinning-friction", str(float(physics["ball_spinning_friction"])),
        "--ball-restitution", str(float(physics["ball_restitution"])),
        "--apple-mass", str(float(physics["apple_mass"])),
        "--apple-friction", str(float(physics["apple_friction"])),
        "--apple-restitution", str(float(physics["apple_restitution"])),
        "--gravity-z", str(float(physics["gravity_z"])),
        "--apple-active", str(int(physics["active"][0])),
        "--ball-active",  str(int(physics["active"][1])),
    ]
    subprocess.run(command, check=True)
    data = json.loads(out.read_text(encoding="utf-8"))
    out.unlink(missing_ok=True)
    return data


def normalize_frames(physics: dict) -> list:
    frames = []
    for fr in physics["frames"]:
        frames.append({
            "frame_index": fr["frame_index"],
            "soccer_ball": {
                "location": fr["soccer_ball"]["location"],
                "quaternion_xyzw": fr["soccer_ball"]["quaternion_xyzw"],
            },
            "apple": {
                "location": fr["apple"]["location"],
                "quaternion_xyzw": fr["apple"]["quaternion_xyzw"],
            },
        })
    return frames


# --- Scene ---------------------------------------------------------------

def build_scene(args: argparse.Namespace, physics: dict, scenario: dict) -> tuple:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.resolution_x = args.resolution[0]
    scene.render.resolution_y = args.resolution[1]
    scene.render.fps = args.fps
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.samples
    scene.cycles.device = "GPU" if args.device == "auto" else "CPU"
    scene.cycles.max_bounces = 8
    scene.cycles.transmission_bounces = 4
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.view_settings.exposure = -0.2

    # World: sunny outdoor sky HDRI, matching the picnic setting.
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    for node in list(nodes):
        nodes.remove(node)
    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    mapping = nodes.new(type="ShaderNodeMapping")
    mapping.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(200.0))
    links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
    env_tex = nodes.new(type="ShaderNodeTexEnvironment")
    hdri_path = POLYHAVEN_DIR / "outdoor" / "syferfontein_0d_clear_puresky_2k.hdr"
    bg = nodes.new(type="ShaderNodeBackground")
    output = nodes.new(type="ShaderNodeOutputWorld")
    if hdri_path.exists():
        env_tex.image = bpy.data.images.load(str(hdri_path), check_existing=True)
        links.new(mapping.outputs["Vector"], env_tex.inputs["Vector"])
        links.new(env_tex.outputs["Color"], bg.inputs["Color"])
    else:
        bg.inputs["Color"].default_value = (0.45, 0.62, 0.85, 1.0)
    bg.inputs["Strength"].default_value = 1.15
    links.new(bg.outputs["Background"], output.inputs["Surface"])

    # Camera
    bpy.ops.object.camera_add(location=CAMERA_LOCATION)
    camera = bpy.context.object
    camera.data.lens = CAMERA_LENS_MM
    camera.data.sensor_width = 32.0
    scene.camera = camera
    target = bpy.data.objects.new("camera_target", None)
    scene.collection.objects.link(target)
    target.location = CAMERA_TARGET
    con = camera.constraints.new("TRACK_TO")
    con.target = target
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"

    # Sun matching the HDRI's implied light direction, plus a soft fill to
    # keep the shadow side of the ball/apple from going fully black.
    bpy.ops.object.light_add(type="SUN", location=(2.0, -2.0, 4.0))
    sun = bpy.context.object
    sun.data.energy = 5.5
    sun.data.angle = math.radians(2.0)
    sun.data.color = (1.0, 0.97, 0.9)
    sun.rotation_euler = (math.radians(52), math.radians(6), math.radians(-35))

    add_area_light(
        "fill_light", (-0.6, -1.3, 1.1), power=35.0, size=1.2,
        target=(0.0, 0.2, 0.15), color=(0.8, 0.87, 1.0),
    )

    create_grass_ground(HERO_GROUND_SIZE, GROUND_SUBDIV, args.grass_density)

    # Picnic spread (static set dressing, no physics)
    picnic = import_glb_group(PICNIC_GLB, "picnic_spread", PICNIC_BLANKET_WIDTH, ref_axes=("x", "y"))
    picnic.location = PICNIC_LOCATION
    picnic.rotation_quaternion = (
        math.cos(math.radians(PICNIC_YAW_DEG) / 2.0), 0.0, 0.0, math.sin(math.radians(PICNIC_YAW_DEG) / 2.0),
    )

    # Physics-driven props. A DELETE edit sets active=0 on one side and we
    # simply skip importing that GLB; the other slot still gets animated
    # from the sim frames.
    frames = normalize_frames(physics)
    active = scenario["physics"]["active"]
    apple_active = bool(int(active[0]))
    ball_active = bool(int(active[1]))

    soccer_ball = None
    if ball_active:
        soccer_ball = import_glb_group(SOCCER_BALL_GLB, "soccer_ball", BALL_DIAMETER, origin="center")
        apply_keyframes(soccer_ball, frames, "soccer_ball")

    apple = None
    if apple_active:
        apple = import_glb_group(APPLE_GLB, "apple", APPLE_DIAMETER, ref_axes=("x", "z"), origin="center")
        apply_keyframes(apple, frames, "apple")

    frame_end = int(physics["frame_end"])
    scene.frame_start = 1
    scene.frame_end = frame_end
    scene.frame_set(1)

    return soccer_ball, apple, camera


def export_ground_truth(out_dir: Path, soccer_ball, apple, camera, physics: dict, scenario: dict) -> None:
    scene = bpy.context.scene
    fps = int(physics["fps"])
    frame_end = int(physics["frame_end"])

    def obj_entry(obj):
        if obj is None:
            return {"present": False, "object_name": None}
        return {"present": True, "object_name": obj.name}

    records = {
        "schema_version": 1,
        "fps": fps,
        "frame_start": 1,
        "frame_end": frame_end,
        "physics": {k: v for k, v in physics.items() if k != "frames"},
        "objects": {
            "soccer_ball": obj_entry(soccer_ball),
            "apple": obj_entry(apple),
        },
        "camera": {
            "object_name": camera.name,
            "lens_mm": float(camera.data.lens),
            "sensor_width_mm": float(camera.data.sensor_width),
            "resolution": [int(scene.render.resolution_x), int(scene.render.resolution_y)],
        },
        "scenario": scenario,
        "frames": [],
    }
    physics_by_frame = {int(fr["frame_index"]): fr for fr in physics["frames"]}

    def frame_entry(obj, pf_entry):
        if obj is None:
            return {"present": False}
        return {
            "present": True,
            "matrix_world": [[float(v) for v in row] for row in obj.matrix_world],
            "linear_velocity": pf_entry["linear_velocity"],
            "angular_velocity": pf_entry["angular_velocity"],
        }

    for frame in range(1, frame_end + 1):
        scene.frame_set(frame)
        pf = physics_by_frame[frame]
        records["frames"].append({
            "frame_index": frame,
            "time_sec": (frame - 1) / float(fps),
            "soccer_ball": frame_entry(soccer_ball, pf["soccer_ball"]),
            "apple": frame_entry(apple, pf["apple"]),
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

    scenario = create_scenario(args)
    (args.out_dir / SCENARIO_METADATA_NAME).write_text(json.dumps(scenario, indent=2), encoding="utf-8")

    physics = run_physics(args, scenario)
    soccer_ball, apple, camera = build_scene(args, physics, scenario)
    export_ground_truth(args.out_dir, soccer_ball, apple, camera, physics, scenario)

    if args.mode == "preview":
        render_preview(args)
    else:
        render_animation(args)

    bpy.ops.wm.save_as_mainfile(filepath=str(args.out_dir / f"{OUTPUT_STEM}.blend"))
    print(f"[INFO] Render complete. Output: {args.out_dir}")


if __name__ == "__main__":
    main()
