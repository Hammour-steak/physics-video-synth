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
from mathutils import Matrix, Vector, noise


WORKSPACE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = WORKSPACE_DIR / "assets" / "models"
POLYHAVEN_DIR = WORKSPACE_DIR / "assets" / "polyhaven"

ROOM_GLB = MODELS_DIR / "living_room_interior_free.glb"
BALL_GLB = MODELS_DIR / "volleyball.glb"

OUTPUT_STEM = "ball_carpet_climb"
BLEND_NAME = f"{OUTPUT_STEM}.blend"
PHYSICS_TEMP_NAME = "physics_transforms.json"
GROUND_TRUTH_NAME = "ground_truth_transforms.json"
SCENARIO_METADATA_NAME = "scenario_metadata.json"

# --- Room model calibration --------------------------------------------------
# assets/models/living_room_interior_free.glb is authored in an arbitrary unit,
# not metres: its floor-to-ceiling span is 20.131 units. Taking that as a
# standard 2.75 m ceiling gives ROOM_SCALE, which independently checks out
# against the model's other furniture (the sectional sofa comes to 2.20 m wide
# and 0.97 m deep, the rug to 1.53 m square) -- so the whole room is uniformly
# scaled by it and everything downstream is in real metres.
#
# The room is then shifted so the physics frame is the natural one for this
# scene: world z = 0 is the floor the ball rolls on, and world (0, 0) is the
# centre of the rug it climbs onto. simulate_ball_carpet_climb.py works in
# exactly these coordinates, so its output needs no transform at all.
ROOM_SCALE = 2.75 / 20.131
ROOM_FLOOR_Z = 0.827          # floor height in raw model units
ROOM_RUG_CENTER = (0.6823, 5.3358)   # rug centre in raw model units

# The model's own rug is a flat plate carrying a flat grey material. It is
# hidden and replaced by a rug built here (see build_carpet) on the exact same
# footprint, so the scene keeps the room's rug where the room put it while
# giving it a real woven surface.
ROOM_RUG_OBJECT = "Cube.001_Material.006_0"
CARPET_HALF = 0.7657          # 1.5314 m square, measured off the model's rug
CARPET_SINK = 0.005           # how far the slab's underside goes below the floor

# The rug is flush with the floor in physics (carpet_thickness = 0), so its
# rendered top face would land exactly on the room's floor and z-fight with it
# across the whole 1.53 m square. It is lifted by this much to separate them.
# 1.2 mm is under 1.2% of the ball's radius -- far below the contact tolerance,
# so the ball still reads as sitting on the rug rather than hovering, and if
# anything the tiny overlap looks like a soft rug compressing under it.
CARPET_RENDER_LIFT = 0.0012

BALL_DIAMETER = 0.21          # FIVB volleyball

# Framing: a wide establishing shot from high on the room's south side, chosen
# from a survey of eight positions (kept in renders/bcc_camera_survey). It puts
# the whole room in frame -- TV console, sofa, window wall, coffee table, rug --
# and holds the ball's entire path, which the tighter alternatives could not:
# every low south-east angle ends with the coffee table's near leg beside the
# stopped ball, and every west-side angle loses the run-up across the bare
# floor off the bottom of frame. The camera sits close to the south wall
# (y = -2.45) and looks down about 30 deg, which is what lets it see the near
# floor the ball starts on; a lower version of the same shot could not.
CAMERA_LOCATION = (0.10, -2.35, 1.25)
CAMERA_TARGET = (-0.38, -0.30, 0.06)
CAMERA_LENS_MM = 24.0


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preview", "animation", "frames"), default="animation")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resolution", nargs=2, type=int, default=(1280, 720))
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-sec", type=float, default=2.0)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--preview-frame", type=int, default=22)
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--scenario-json", type=Path, default=None,
        help="Optional complete scenario_metadata.json to render instead of building one.",
    )
    parser.add_argument(
        "--scenario-overrides-json", type=Path, default=None,
        help="Optional JSON object recursively merged onto the built scenario.",
    )
    return parser.parse_args(argv)


# --- Scenario -----------------------------------------------------------------

def deep_merge(base: dict, overrides: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def build_scenario(args: argparse.Namespace) -> dict[str, Any]:
    if args.scenario_json is not None:
        return json.loads(args.scenario_json.read_text(encoding="utf-8"))

    scenario: dict[str, Any] = {
        "schema_version": 1,
        "seed": int(args.seed),
        "carpet_style": "wool_greige",
        "render": {
            "mode": str(args.mode),
            "fps": int(args.fps),
            "duration_sec": float(args.duration_sec),
            "resolution": [int(args.resolution[0]), int(args.resolution[1])],
            "samples": int(args.samples),
            "device": str(args.device),
        },
        "camera": {
            "location": list(CAMERA_LOCATION),
            "target": list(CAMERA_TARGET),
            "lens_mm": CAMERA_LENS_MM,
        },
        # Tuned in simulate_ball_carpet_climb.py. The rug is flush with the
        # floor, so the ball crosses ~0.5 m of bare floor barely slowing, meets
        # the rug still doing 2.0 m/s with no step to check it, and is brought
        # down purely by the rug's rolling resistance over a further ~0.49 m,
        # stopping beside the coffee table about 0.8 s in.
        #
        # carpet_rolling_friction is 0.060 rather than the 0.028 this scene used
        # when the rug was a raised slab: back then the 18 mm edge itself ate
        # 0.75 m/s before the pile got the ball at all. Take the step away and
        # the ball arrives with all of its speed intact, so the rug has to do
        # roughly twice the work to stop it in the same distance.
        #
        # launch_y is set by the framing rather than by the physics: the floor
        # is slick enough that the run-up's length barely changes the speed
        # reaching the rug, so it is set to the longest approach that still has
        # the ball fully inside frame 1 of the wide camera -- at -1.55 the ball
        # is half cut off by the left edge.
        "physics": {
            "launch_x": -0.57,
            "launch_y": -1.35,
            "launch_speed": 2.05,
            "carpet_thickness": 0.0,
            "carpet_rolling_friction": 0.060,
            "carpet_friction": 0.9,
            "carpet_restitution": 0.1,
            "floor_rolling_friction": 0.0015,
            "floor_friction": 0.45,
            "floor_restitution": 0.7,
            "ball_mass": 0.27,
            "ball_friction": 1.0,
            "ball_restitution": 0.45,
            "gravity_z": -9.8,
        },
    }
    if args.scenario_overrides_json is not None:
        scenario = deep_merge(
            scenario, json.loads(args.scenario_overrides_json.read_text(encoding="utf-8")),
        )
    return scenario


# --- Blender helpers ----------------------------------------------------------

def world_bbox(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    return (Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners))),
            Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners))))


def look_at(obj: bpy.types.Object, target) -> None:
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_area_light(name, location, power, size, target, color=(1.0, 1.0, 1.0)) -> bpy.types.Object:
    bpy.ops.object.light_add(type="AREA", location=location)
    light = bpy.context.object
    light.name = name
    light.data.energy = float(power)
    light.data.size = float(size)
    light.data.color = color
    light.visible_camera = False
    look_at(light, target)
    return light


def set_input(node, name, value) -> None:
    socket = node.inputs.get(name)
    if socket is not None:
        socket.default_value = value


def enable_gpu() -> None:
    prefs = bpy.context.preferences.addons.get("cycles")
    if prefs is None:
        return
    cprefs = prefs.preferences
    for backend in ("OPTIX", "CUDA"):
        try:
            cprefs.compute_device_type = backend
        except TypeError:
            continue
        cprefs.get_devices()
        devices = [d for d in cprefs.devices if d.type == backend]
        if devices:
            for device in cprefs.devices:
                device.use = device.type == backend
            print(f"[INFO] Cycles device backend: {backend} ({len(devices)} device(s))")
            return
    print("[WARN] No OPTIX/CUDA device found; Cycles will fall back to CPU.")


# --- Room ---------------------------------------------------------------------

def tame_floor_gloss(imported: list[bpy.types.Object]) -> None:
    """Make the room's floor a floor instead of a mirror.

    The GLB ships its floor material at Metallic 0.98 / Roughness 0.09, i.e. as
    polished chrome. Under the viewer-less EEVEE preview the model was authored
    against that mostly passes, but in Cycles the floor turns into a perfect
    mirror and the room renders as though it were flooded -- the ball's
    reflection is sharper than the ball. Since the ball rolls across this
    surface for the first two thirds of the shot, that is not a background
    detail.

    The floor is found by geometry rather than by name (the model's floor
    object has a mojibake name that is not safe to match on): the largest
    perfectly flat horizontal mesh sitting at the model's floor height.
    """
    floor_obj = None
    best_area = 0.0
    for obj in imported:
        if obj.type != "MESH":
            continue
        mn, mx = world_bbox(obj)
        if mx.z - mn.z > 1e-4 or abs(mn.z - ROOM_FLOOR_Z) > 0.05:
            continue
        area = (mx.x - mn.x) * (mx.y - mn.y)
        if area > best_area:
            best_area, floor_obj = area, obj
    if floor_obj is None:
        print("[WARN] Could not identify the room's floor; leaving its material as authored.")
        return

    for slot in floor_obj.material_slots:
        mat = slot.material
        if mat is None or not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.bl_idname != "ShaderNodeBsdfPrincipled":
                continue
            set_input(node, "Metallic", 0.0)
            roughness = node.inputs.get("Roughness")
            if roughness is not None and not roughness.is_linked:
                roughness.default_value = max(0.22, float(roughness.default_value))
            set_input(node, "Specular", 0.4)
    print(f"[INFO] De-mirrored the room floor ({floor_obj.name!r}).")


def import_room() -> bpy.types.Object:
    """Import the living room and drop it into the physics world frame.

    Rather than baking a transform into every one of the model's meshes, all of
    its root objects are parented to a single empty carrying the unit
    conversion and the shift onto the rug centre. Parenting with an identity
    parent-inverse means each mesh's own transform composes with the empty's,
    so the model stays internally untouched and the mapping stays inspectable
    in the saved .blend.
    """
    if not ROOM_GLB.exists():
        raise FileNotFoundError(f"Room model not found: {ROOM_GLB}")
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(ROOM_GLB))
    imported = [o for o in bpy.context.scene.objects if o not in before]

    room_root = bpy.data.objects.new("room_root", None)
    bpy.context.scene.collection.objects.link(room_root)
    room_root.location = (
        -ROOM_SCALE * ROOM_RUG_CENTER[0],
        -ROOM_SCALE * ROOM_RUG_CENTER[1],
        -ROOM_SCALE * ROOM_FLOOR_Z,
    )
    room_root.scale = (ROOM_SCALE, ROOM_SCALE, ROOM_SCALE)

    for obj in imported:
        if obj.parent is None:
            obj.parent = room_root
            obj.matrix_parent_inverse = Matrix.Identity(4)

    tame_floor_gloss(imported)

    flat_rug = bpy.data.objects.get(ROOM_RUG_OBJECT)
    if flat_rug is None:
        raise RuntimeError(
            f"Expected the room's rug plate {ROOM_RUG_OBJECT!r}; the model may have changed."
        )
    flat_rug.hide_render = True
    flat_rug.hide_viewport = True

    bpy.context.view_layer.update()
    return room_root


# --- Carpet -------------------------------------------------------------------

def create_carpet_material() -> bpy.types.Material:
    """A procedural cut-pile wool rug.

    No woven-fabric photo scan ships with this repo and ambientCG is not
    reachable from the render host, so the rug is built out of shader nodes the
    same way car_ramp_climb builds its grip-tape and turf surfaces.

    What actually makes it read as carpet rather than as felt or poured
    concrete is the *tuft* layer: a Voronoi cell pattern at roughly 9 mm, bump-
    mapped into rounded domes and given per-cell colour jitter. A first pass
    used three plain noise octaves instead and rendered as a smooth grey slab --
    noise alone has no cell structure, so nothing in it reads as individual
    tufts of yarn. The Voronoi cells carry the silhouette; a very fine noise on
    top of them carries the fibres inside each tuft; a coarse noise underneath
    carries the dye blotching and the slight unevenness of a laid rug.

    A darker bound border rings the rug. With the rug flush to the floor there
    is no edge geometry left to catch the light, so this band is the only thing
    marking where the high-friction surface starts -- and the moment the ball
    crosses it is the whole point of the scene. It is 55 mm wide: an earlier
    26 mm band read as barely a line at the hero camera's standoff.
    """
    mat = bpy.data.materials.new("carpet_wool")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")

    set_input(bsdf, "Roughness", 0.96)
    set_input(bsdf, "Metallic", 0.0)
    set_input(bsdf, "Specular", 0.10)
    set_input(bsdf, "Sheen", 0.55)          # wool's soft grazing-angle lift
    set_input(bsdf, "Sheen Tint", 0.5)

    tex_coord = nodes.new("ShaderNodeTexCoord")

    def noise_node(scale: float, detail: float, roughness: float):
        n = nodes.new("ShaderNodeTexNoise")
        n.inputs["Scale"].default_value = scale
        n.inputs["Detail"].default_value = detail
        n.inputs["Roughness"].default_value = roughness
        links.new(tex_coord.outputs["Object"], n.inputs["Vector"])
        return n

    dye = noise_node(4.0, 6.0, 0.6)
    lint = noise_node(38.0, 4.0, 0.6)
    fibre = noise_node(900.0, 3.0, 0.85)

    tuft = nodes.new("ShaderNodeTexVoronoi")
    tuft.feature = "F1"
    tuft.inputs["Scale"].default_value = 165.0     # ~9 mm cells on a 1.53 m rug
    tuft.inputs["Randomness"].default_value = 1.0
    links.new(tex_coord.outputs["Object"], tuft.inputs["Vector"])

    # Voronoi F1 Distance is 0 at each cell's seed and rises toward the cell
    # walls, so it has to be inverted before it can drive a bump: uninverted it
    # would pit every tuft instead of doming it.
    tuft_dome = nodes.new("ShaderNodeMapRange")
    tuft_dome.inputs["From Min"].default_value = 0.0
    tuft_dome.inputs["From Max"].default_value = 0.70
    tuft_dome.inputs["To Min"].default_value = 1.0
    tuft_dome.inputs["To Max"].default_value = 0.0
    tuft_dome.clamp = True
    links.new(tuft.outputs["Distance"], tuft_dome.inputs["Value"])

    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.36
    ramp.color_ramp.elements[0].color = (0.068, 0.058, 0.049, 1.0)
    ramp.color_ramp.elements[1].position = 0.66
    ramp.color_ramp.elements[1].color = (0.165, 0.146, 0.126, 1.0)
    links.new(dye.outputs["Fac"], ramp.inputs["Fac"])

    # Per-tuft colour jitter. Voronoi's Color output is constant within a cell
    # and random between cells, which is exactly how a cut pile looks: each
    # tuft catches the light slightly differently from its neighbours.
    tuft_bw = nodes.new("ShaderNodeRGBToBW")
    links.new(tuft.outputs["Color"], tuft_bw.inputs["Color"])
    tuft_gain = nodes.new("ShaderNodeMapRange")
    tuft_gain.inputs["From Min"].default_value = 0.0
    tuft_gain.inputs["From Max"].default_value = 1.0
    tuft_gain.inputs["To Min"].default_value = 0.82
    tuft_gain.inputs["To Max"].default_value = 1.20
    links.new(tuft_bw.outputs["Val"], tuft_gain.inputs["Value"])
    tuft_tint = nodes.new("ShaderNodeHueSaturation")
    links.new(tuft_gain.outputs["Result"], tuft_tint.inputs["Value"])
    links.new(ramp.outputs["Color"], tuft_tint.inputs["Color"])

    lint_mix = nodes.new("ShaderNodeMixRGB")
    lint_mix.blend_type = "MULTIPLY"
    lint_mix.inputs["Fac"].default_value = 0.22
    links.new(tuft_tint.outputs["Color"], lint_mix.inputs["Color1"])
    links.new(lint.outputs["Color"], lint_mix.inputs["Color2"])

    # Bound border. `cheb` is the Chebyshev distance from the rug's centre in
    # the slab's own object space, so the mask is a square ring that follows
    # the edge exactly regardless of where the object sits in the room.
    sep = nodes.new("ShaderNodeSeparateXYZ")
    links.new(tex_coord.outputs["Object"], sep.inputs["Vector"])
    abs_x = nodes.new("ShaderNodeMath")
    abs_x.operation = "ABSOLUTE"
    links.new(sep.outputs["X"], abs_x.inputs[0])
    abs_y = nodes.new("ShaderNodeMath")
    abs_y.operation = "ABSOLUTE"
    links.new(sep.outputs["Y"], abs_y.inputs[0])
    cheb = nodes.new("ShaderNodeMath")
    cheb.operation = "MAXIMUM"
    links.new(abs_x.outputs["Value"], cheb.inputs[0])
    links.new(abs_y.outputs["Value"], cheb.inputs[1])
    border = nodes.new("ShaderNodeMapRange")
    border.inputs["From Min"].default_value = CARPET_HALF - 0.055
    border.inputs["From Max"].default_value = CARPET_HALF - 0.044
    border.clamp = True
    links.new(cheb.outputs["Value"], border.inputs["Value"])

    border_mix = nodes.new("ShaderNodeMixRGB")
    border_mix.blend_type = "MIX"
    border_mix.inputs["Color2"].default_value = (0.031, 0.026, 0.022, 1.0)
    links.new(border.outputs["Result"], border_mix.inputs["Fac"])
    links.new(lint_mix.outputs["Color"], border_mix.inputs["Color1"])
    links.new(border_mix.outputs["Color"], bsdf.inputs["Base Color"])

    # Three-stage bump, coarsest first: the laid-rug unevenness carries the
    # tufts, and the tufts carry the fibres.
    lint_bump = nodes.new("ShaderNodeBump")
    lint_bump.inputs["Strength"].default_value = 0.35
    lint_bump.inputs["Distance"].default_value = 0.0035
    links.new(lint.outputs["Fac"], lint_bump.inputs["Height"])
    tuft_bump = nodes.new("ShaderNodeBump")
    tuft_bump.inputs["Strength"].default_value = 1.0
    tuft_bump.inputs["Distance"].default_value = 0.0065
    links.new(tuft_dome.outputs["Result"], tuft_bump.inputs["Height"])
    links.new(lint_bump.outputs["Normal"], tuft_bump.inputs["Normal"])
    fibre_bump = nodes.new("ShaderNodeBump")
    fibre_bump.inputs["Strength"].default_value = 0.6
    fibre_bump.inputs["Distance"].default_value = 0.0006
    links.new(fibre.outputs["Fac"], fibre_bump.inputs["Height"])
    links.new(tuft_bump.outputs["Normal"], fibre_bump.inputs["Normal"])
    links.new(fibre_bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def build_carpet(thickness: float, seed: int) -> bpy.types.Object:
    """The rug: a slab whose top face matches the physics surface.

    The collision shape in simulate_ball_carpet_climb.py has its top face at
    z = thickness, which at the default thickness of 0 means the rug is exactly
    flush with the bare floor and the ball rolls over one continuous plane.

    Rendering that flush is the one place the rug is allowed to disagree with
    the physics: a mesh sitting exactly on the room's floor z-fights with it
    over the whole 1.53 m square, so the whole rug is lifted by
    CARPET_RENDER_LIFT. The surface noise on top of that is deliberately
    one-sided (mapped to [0, 1] rather than [-1, 1]) so that no vertex can dip
    back down through the floor and reopen the z-fighting the lift exists to
    prevent -- a symmetric lift leaves roughly half the rug's vertices below the
    floor and renders as a shimmering chequerboard. The noise is capped at ~1 mm
    and faded out over the outer 0.18 m, which also keeps the underside flat
    where it meets the floor so no daylight shows under the rim.
    """
    rng = random.Random(seed)
    offset = Vector((rng.uniform(-40.0, 40.0), rng.uniform(-40.0, 40.0), rng.uniform(-40.0, 40.0)))

    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=96, y_subdivisions=96, size=CARPET_HALF * 2.0,
        location=(0.0, 0.0, thickness + CARPET_RENDER_LIFT),
    )
    carpet = bpy.context.object
    carpet.name = "carpet"

    lift_amplitude = 0.0011
    fade_band = 0.18
    for v in carpet.data.vertices:
        edge_dist = CARPET_HALF - max(abs(v.co.x), abs(v.co.y))
        fade = min(1.0, max(0.0, edge_dist / fade_band))
        n = noise.noise(Vector((v.co.x * 2.6, v.co.y * 2.6, 0.0)) + offset)
        v.co.z += (n * 0.5 + 0.5) * lift_amplitude * fade
    carpet.data.update()

    solidify = carpet.modifiers.new("thickness", "SOLIDIFY")
    solidify.thickness = thickness + CARPET_RENDER_LIFT + CARPET_SINK
    solidify.offset = -1.0          # grow downward from the top face
    solidify.use_even_offset = True

    # Soften only the rim. An angle limit keeps the bevel off the grid's own
    # near-flat interior edges, which would otherwise eat the whole top face.
    bevel = carpet.modifiers.new("edge_bind", "BEVEL")
    bevel.width = 0.0035
    bevel.segments = 3
    bevel.limit_method = "ANGLE"
    bevel.angle_limit = math.radians(40.0)

    carpet.data.use_auto_smooth = True
    carpet.data.auto_smooth_angle = math.radians(35.0)
    for poly in carpet.data.polygons:
        poly.use_smooth = True

    carpet.data.materials.append(create_carpet_material())
    return carpet


# --- Ball ---------------------------------------------------------------------

def import_ball(diameter: float) -> bpy.types.Object:
    if not BALL_GLB.exists():
        raise FileNotFoundError(f"Ball model not found: {BALL_GLB}")
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(BALL_GLB))
    imported = [o for o in bpy.context.scene.objects if o not in before]
    meshes = [o for o in imported if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh found in {BALL_GLB}")

    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    if len(meshes) > 1:
        bpy.ops.object.join()
    ball = bpy.context.view_layer.objects.active
    for o in imported:
        if o is not ball and o.name in bpy.data.objects:
            bpy.data.objects.remove(o, do_unlink=True)

    mn, mx = world_bbox(ball)
    raw = ((mx.x - mn.x) + (mx.y - mn.y) + (mx.z - mn.z)) / 3.0
    ball.scale = (diameter / raw,) * 3
    bpy.ops.object.select_all(action="DESELECT")
    ball.select_set(True)
    bpy.context.view_layer.objects.active = ball
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    # Origin at the true centre, not the bottom: the physics frames give sphere
    # centre positions plus a spin quaternion, and a sphere pivoting about
    # anything other than its centre visibly wobbles as it rolls.
    mn, mx = world_bbox(ball)
    bpy.context.scene.cursor.location = ((mn.x + mx.x) / 2.0, (mn.y + mx.y) / 2.0, (mn.z + mx.z) / 2.0)
    bpy.ops.object.origin_set(type="ORIGIN_CURSOR")
    ball.location = (0.0, 0.0, 0.0)
    ball.name = "toy_ball"
    ball.rotation_mode = "QUATERNION"
    for poly in ball.data.polygons:
        poly.use_smooth = True
    return ball


def apply_keyframes(obj: bpy.types.Object, frames: list) -> None:
    obj.rotation_mode = "QUATERNION"
    for fr in frames:
        d = fr["ball"]
        obj.location = d["location"]
        q = d["quaternion_xyzw"]
        obj.rotation_quaternion = (q[3], q[0], q[1], q[2])
        obj.keyframe_insert(data_path="location", frame=int(fr["frame_index"]))
        obj.keyframe_insert(data_path="rotation_quaternion", frame=int(fr["frame_index"]))
    if obj.animation_data and obj.animation_data.action:
        for fc in obj.animation_data.action.fcurves:
            for k in fc.keyframe_points:
                k.interpolation = "LINEAR"


# --- Physics ------------------------------------------------------------------

def run_physics(args: argparse.Namespace, scenario: dict) -> dict:
    physics_python = WORKSPACE_DIR.parent / "miniconda" / "envs" / "physics" / "bin" / "python"
    python = str(physics_python) if physics_python.exists() else (
        shutil.which("python3") or shutil.which("python")
    )
    if not python:
        raise RuntimeError("Cannot find python3/python for the PyBullet physics simulation.")

    p = scenario["physics"]
    out = args.out_dir / PHYSICS_TEMP_NAME
    command = [
        python, str(Path(__file__).with_name("simulate_ball_carpet_climb.py")),
        "--out", str(out),
        "--fps", str(int(scenario["render"]["fps"])),
        "--duration-sec", str(float(scenario["render"]["duration_sec"])),
        "--launch-x", str(float(p["launch_x"])),
        "--launch-y", str(float(p["launch_y"])),
        "--launch-speed", str(float(p["launch_speed"])),
        "--carpet-thickness", str(float(p["carpet_thickness"])),
        "--carpet-rolling-friction", str(float(p["carpet_rolling_friction"])),
        "--carpet-friction", str(float(p["carpet_friction"])),
        "--carpet-restitution", str(float(p["carpet_restitution"])),
        "--floor-rolling-friction", str(float(p["floor_rolling_friction"])),
        "--floor-friction", str(float(p["floor_friction"])),
        "--floor-restitution", str(float(p["floor_restitution"])),
        "--ball-mass", str(float(p["ball_mass"])),
        "--ball-friction", str(float(p["ball_friction"])),
        "--ball-restitution", str(float(p["ball_restitution"])),
        "--gravity-z", str(float(p["gravity_z"])),
    ]
    subprocess.run(command, check=True)
    data = json.loads(out.read_text(encoding="utf-8"))
    out.unlink(missing_ok=True)

    q = data["quality"]
    if not q["reached_carpet"]:
        print("[WARN] The ball never reached the rug in this scenario.")
    if q["hit_table_leg"]:
        print(f"[WARN] The ball reached a coffee-table leg at frame {q['table_leg_contact_frame']}.")
    if q["settled_frame"] is None:
        print("[WARN] The ball was still moving at the last frame; it never settles on camera.")
    return data


# --- Scene --------------------------------------------------------------------

def build_scene(args: argparse.Namespace, scenario: dict, physics: dict):
    scene = bpy.context.scene
    render = scenario["render"]
    scene.render.resolution_x = int(render["resolution"][0])
    scene.render.resolution_y = int(render["resolution"][1])
    scene.render.fps = int(render["fps"])
    scene.render.engine = "CYCLES"
    scene.cycles.samples = int(render["samples"])
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 10
    scene.cycles.transmission_bounces = 6
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.view_settings.look = "Medium Contrast"

    if str(render["device"]) == "auto":
        enable_gpu()
        scene.cycles.device = "GPU"
    else:
        scene.cycles.device = "CPU"

    # World: an interior HDRI, chosen over a plain grey background because the
    # room's whole far wall is glazing -- whatever the world is, the camera sees
    # it straight through the window behind the sofa.
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    for node in list(nodes):
        nodes.remove(node)
    output = nodes.new("ShaderNodeOutputWorld")
    bg = nodes.new("ShaderNodeBackground")
    hdri = POLYHAVEN_DIR / "lythwood_room" / "lythwood_room_2k.hdr"
    if hdri.exists():
        tex_coord = nodes.new("ShaderNodeTexCoord")
        mapping = nodes.new("ShaderNodeMapping")
        mapping.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(115.0))
        env = nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(str(hdri), check_existing=True)
        links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], env.inputs["Vector"])
        links.new(env.outputs["Color"], bg.inputs["Color"])
        bg.inputs["Strength"].default_value = 1.6
    else:
        print(f"[WARN] HDRI not found at {hdri}; falling back to a flat sky colour.")
        bg.inputs["Color"].default_value = (0.62, 0.68, 0.78, 1.0)
        bg.inputs["Strength"].default_value = 2.2
    links.new(bg.outputs["Background"], output.inputs["Surface"])

    import_room()
    carpet = build_carpet(float(scenario["physics"]["carpet_thickness"]), int(scenario["seed"]))

    ball = import_ball(BALL_DIAMETER)
    apply_keyframes(ball, physics["frames"])

    cam_cfg = scenario["camera"]
    bpy.ops.object.camera_add(location=tuple(cam_cfg["location"]))
    camera = bpy.context.object
    camera.data.lens = float(cam_cfg["lens_mm"])
    camera.data.sensor_width = 36.0
    camera.data.clip_start = 0.02
    scene.camera = camera
    target = bpy.data.objects.new("camera_target", None)
    scene.collection.objects.link(target)
    target.location = tuple(cam_cfg["target"])
    con = camera.constraints.new("TRACK_TO")
    con.target = target
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"

    # Daylight through the window wall behind the sofa (+Y), plus a soft bounce
    # from the camera side so the ball's shadow-facing half does not go black.
    bpy.ops.object.light_add(type="SUN", location=(-1.5, 3.5, 2.6))
    sun = bpy.context.object
    sun.data.energy = 3.2
    sun.data.angle = math.radians(3.0)
    sun.data.color = (1.0, 0.96, 0.90)
    sun.rotation_euler = (math.radians(56.0), math.radians(4.0), math.radians(196.0))

    add_area_light("window_light", (-1.2, 2.4, 1.5), power=180.0, size=2.4,
                   target=(-0.3, -0.3, 0.2), color=(0.95, 0.97, 1.0))
    add_area_light("camera_fill", (1.3, -2.0, 1.2), power=45.0, size=1.6,
                   target=(-0.45, -0.5, 0.10), color=(1.0, 0.97, 0.94))

    scene.frame_start = 1
    scene.frame_end = int(physics["frame_end"])
    scene.frame_set(1)
    return ball, carpet, camera


# --- Outputs ------------------------------------------------------------------

def export_ground_truth(out_dir: Path, ball, carpet, camera, physics: dict, scenario: dict) -> None:
    scene = bpy.context.scene
    fps = int(physics["fps"])
    frame_end = int(physics["frame_end"])
    thickness = float(scenario["physics"]["carpet_thickness"])
    records = {
        "schema_version": 1,
        "fps": fps,
        "frame_start": 1,
        "frame_end": frame_end,
        "physics": {k: v for k, v in physics.items() if k != "frames"},
        "objects": {
            "ball": {"object_name": ball.name, "radius": physics["objects"]["ball"]["radius"]},
            "carpet": {
                "object_name": carpet.name,
                "half_extent": CARPET_HALF,
                "thickness": thickness,
                "top_z": thickness,
            },
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
    by_frame = {int(fr["frame_index"]): fr for fr in physics["frames"]}
    for frame in range(1, frame_end + 1):
        scene.frame_set(frame)
        pf = by_frame[frame]
        records["frames"].append({
            "frame_index": frame,
            "time_sec": (frame - 1) / float(fps),
            "ball": {
                "matrix_world": [[float(v) for v in row] for row in ball.matrix_world],
                "linear_velocity": pf["ball"]["linear_velocity"],
                "angular_velocity": pf["ball"]["angular_velocity"],
                "speed": pf["ball"]["speed"],
                "on_carpet": pf["ball"]["on_carpet"],
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


def render_frames(args: argparse.Namespace) -> None:
    scene = bpy.context.scene
    scene.render.image_settings.file_format = "PNG"
    scene.frame_set(scene.frame_start)
    scene.render.filepath = str(args.out_dir / "frame_")
    bpy.ops.render.render(animation=True)


def render_animation(args: argparse.Namespace) -> None:
    scene = bpy.context.scene
    scene.frame_set(scene.frame_start)
    scene.render.filepath = str(args.out_dir / f"{OUTPUT_STEM}.mp4")
    bpy.ops.render.render(animation=True)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    scenario = build_scenario(args)
    (args.out_dir / SCENARIO_METADATA_NAME).write_text(
        json.dumps(scenario, indent=2), encoding="utf-8",
    )

    physics = run_physics(args, scenario)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    ball, carpet, camera = build_scene(args, scenario, physics)
    export_ground_truth(args.out_dir, ball, carpet, camera, physics, scenario)

    if args.mode == "preview":
        render_preview(args)
    elif args.mode == "frames":
        render_frames(args)
    else:
        render_animation(args)

    bpy.ops.wm.save_as_mainfile(filepath=str(args.out_dir / BLEND_NAME))
    print(f"[INFO] Render complete. Output: {args.out_dir}")


if __name__ == "__main__":
    main()
