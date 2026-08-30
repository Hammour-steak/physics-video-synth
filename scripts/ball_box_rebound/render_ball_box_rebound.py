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
from mathutils import Matrix, Vector


WORKSPACE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = WORKSPACE_DIR / "assets" / "models"
POLYHAVEN_DIR = WORKSPACE_DIR / "assets" / "polyhaven"
COLLISION_DIR = WORKSPACE_DIR / "assets" / "collision"

ROOM_GLB = MODELS_DIR / "toy_box.glb"
CHEST_COLLIDER_OBJ = COLLISION_DIR / "toy_box_chest.obj"
PROPS_COLLIDER_OBJ = COLLISION_DIR / "toy_box_props.obj"

OUTPUT_STEM = "ball_box_rebound"
BLEND_NAME = f"{OUTPUT_STEM}.blend"
PHYSICS_TEMP_NAME = "physics_transforms.json"
GROUND_TRUTH_NAME = "ground_truth_transforms.json"
SCENARIO_METADATA_NAME = "scenario_metadata.json"

# --- Room model calibration ---------------------------------------------------
# assets/models/toy_box.glb is authored in an arbitrary unit, not metres: its
# floor-to-wall-top span is 181.571 units. Taking that as a 2.60 m room gives
# ROOM_SCALE, which checks out independently against the model's own window --
# the sill lands at 1.02 m and the head at 2.15 m, which is where a real window
# sits. It also puts the toy chest at 1.32 x 0.80 x 0.67 m and the printed play
# mat at 2.47 x 1.87 m, both plausible for the oversized-toy look of the set.
ROOM_WALL_UNITS = 181.571
ROOM_HEIGHT_M = 2.60
ROOM_SCALE = ROOM_HEIGHT_M / ROOM_WALL_UNITS

# The world frame is chosen to be the natural one for this scene, so that
# simulate_ball_box_rebound.py's output needs no transform at all:
#   z = 0   is the floor the balls roll on,
#   (0, 0)  is on the toy chest's north face at the mid-point of its width.
#
# ROOM_GROUND_Z is model z = 0.272, the top of the room's printed play mat --
# not of its own floorboards, which are 0.92 units (13 mm) below that. The mat
# is taken out of this scene (see ROOM_HIDDEN_OBJECTS), but it is still the
# plane the set was built on: the model has the chest, the toy basket and the
# teddy bear all standing on the mat rather than on the boards. Ground level is
# therefore the mat's surface, and it is the laid-in hardwood floor that is
# brought up to meet it, not the props that are dropped 13 mm to meet the
# model's boards.
ROOM_GROUND_Z = 0.272
ROOM_ORIGIN_XY = (55.096, 10.923)

# The chest's north face, in world metres and measured off the model by ray-
# casting at the height a rolling ball actually touches it. The chest is yawed
# 3.74 deg relative to the room's axes, so its face is the line
# y = CHEST_FACE_SLOPE * x rather than y = 0, and the rebound is *not* a mirror
# about the world axes. The simulation never assumes it is -- it reflects off
# the contact normal Bullet reports off the real mesh -- but the aiming
# constants below were derived from this line.
CHEST_FACE_SLOPE = 0.0653
CHEST_FACE_X = (-0.617, 0.571)

# The model ships two fake lighting cards: a window-shaped quad and a huge
# god-ray wedge spanning the whole room. Both carry an ordinary diffuse
# material meant to be composited additively by a game engine. Cycles has no
# such blend mode, so it renders them as what they geometrically are: two large
# grey-yellow slabs hanging in the room, one of which passes straight through
# the play area. They are hidden and replaced with real lights.
# Hidden from the render, on top of the two fake light cards above.
#
# The printed road-map play mat goes because the balls would otherwise cross a
# surface boundary in the middle of the shot, and because its printed roads and
# houses are busy enough at this scale to compete with the two things the shot
# is about. Without it the ground is one plane with one set of properties, which
# is both easier to read and easier to simulate honestly.
#
# The model's own floorboards go with it, and this is not optional. They are a
# single quad covering 3.11 x 2.60 m -- the set is a corner, with a west wall, a
# south wall and two open sides -- and it is mapped into a shared texture atlas,
# so it can be neither tiled nor enlarged. The ball's run-up alone is 1.15 m and
# the camera has to stand behind it, which puts every usable camera position
# past the edge of that quad and its far side in frame as a hard line with
# nothing under it. They are replaced by a real tiling hardwood floor (see
# build_hardwood_floor), which is also a considerably better floor: the model's
# is a low-resolution baked bitmap, and this shot is largely a picture of it.
ROOM_HIDDEN_OBJECTS = (
    "Lightbeam2_Mat_Light_0",    # fake god-ray card
    "WindowLight_Mat_Light_0",   # fake window-glow card
    "Rug_Mat_Rug_0",             # printed play mat
    "Floor_Mat_Room_0",          # baked floorboard quad
)

# The laid-in floor. Poly Haven's worn pine boards, already in the repo for
# ball_block_impact. FLOOR_TEXTURE_SPAN is how much floor one tile of the 4K
# texture covers; the boards read at roughly the right width for a room at 2.2 m.
FLOOR_SIZE = 9.0
FLOOR_TEXTURE_SPAN = 2.2
FLOOR_ASSET = "wood_floor_worn"

# The static geometry the balls can reach, exported to metres as triangle
# meshes for PyBullet. It goes out as two files, not one, and the split is not
# cosmetic: with everything in a single body "the ball bounced off the chest"
# and "the ball fetched up against the toy basket" are the same contact event
# as far as the metrics can tell, and the simulation happily reported the
# second as the scene's rebound. The chest is the scene; the lid, the basket
# and the teddy bear are only in the simulation so that a ball pushed harder
# than the hero case stops against them the way it visibly would instead of
# rolling through the rendered furniture.
CHEST_COLLIDER_OBJECTS = ("Box_Base_Mat_Box_0",)
PROP_COLLIDER_OBJECTS = (
    "Box_Lid_Mat_Box_0",
    "Basket1_Mat_Box_0",
    "Basket_Handle_Mat_Box_0",
    "Bear_Mat_Toys_0",
)

# The two balls are the room's own toys, lifted out of the set and driven by
# the physics. Ball1 is the pink ball with a star printed on it -- that star is
# why it is the one that rolls, because it is the only thing in the shot that
# makes the ball's spin readable. Ball2 is the little football lying inside the
# chest; it comes out onto the mat as the target.
#
# Both are re-sized. As authored they are a 0.32 m and a 0.24 m ball, and at
# that size two of them plus a bounce do not fit in the 1.6 x 0.72 m of clear
# mat north of the chest. At 0.16 m and 0.13 m they are still perfectly
# ordinary toys and the whole chain fits with room to spare.
BALL_A_OBJECT = "Ball1_Mat_Toys_0"
BALL_B_OBJECT = "Ball2_Mat_Toys_0"
BALL_A_DIAMETER = 0.160
BALL_B_DIAMETER = 0.130

# Framing: a low three-quarter angle from the room's north-east, chosen from a
# survey of ten positions (kept in renders/bbr_camera_survey/, with cam_survey.py
# to regenerate it). The camera sits 0.36 m off the floor, which is barely more
# than two ball diameters, so the balls read as objects rolling across a floor
# rather than as markers on a plan. The three-quarter angle on the chest's front
# panel is what makes the bounce legible: square-on to the panel the approach
# and the rebound project onto nearly the same line and the ball just appears to
# stop and start again.
#
# The wider north-east positions hold more of the room but shrink the balls; the
# north-west ones give the approach more depth at the cost of a foreground ball
# nearly filling frame 1; the tighter north position loses the launch off the
# edge entirely.
CAMERA_LOCATION = (1.20, 1.40, 0.36)
CAMERA_TARGET = (-0.12, 0.48, 0.07)
CAMERA_LENS_MM = 35.0


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("preview", "animation", "frames", "colliders"),
        default="animation",
        help="'colliders' only rebuilds the assets/collision/*.obj collision "
        "meshes and exits, which is all simulate_ball_box_rebound.py needs to "
        "run standalone.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resolution", nargs=2, type=int, default=(1280, 720))
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-sec", type=float, default=2.6)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--preview-frame", type=int, default=23)
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--refresh-colliders", action="store_true",
        help="Rebuild the room collision meshes even if they already exist.",
    )
    parser.add_argument("--scenario-json", type=Path, default=None)
    parser.add_argument("--scenario-overrides-json", type=Path, default=None)
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
        # Tuned in simulate_ball_box_rebound.py.
        #
        # The ball is rolled 1.15 m across the open boards north of the chest
        # and meets its front panel 30 deg off the normal. Squarer than that is
        # not available: the toy basket blocks everything west of x = -0.73 up
        # to y = 0.61, so the run-up has to come in over the basket's shoulder,
        # and a genuinely head-on bounce would send the ball straight back up
        # its own approach line and there would be nowhere to stand the target
        # ball that it did not already run through on the way in. At 30 deg the
        # ball still visibly *bounces off* rather than glances, and the rebound
        # crosses the room diagonally instead of retracing the approach.
        #
        # ball_b sits on that rebound line 0.40 m past the chest, offset 40 mm
        # off centre. Two things come out of those numbers. The gap is what
        # makes the shot read as three events rather than one scramble: the
        # bounce lands on frame 13 and the ball-ball hit on frame 24, nearly
        # half a second apart. The offset is what keeps both balls alive
        # afterwards -- dead centre, ball A stops where it stands (0.17 m/s) and
        # only ball B moves, which is a tidier demonstration of momentum
        # transfer but a much less informative one. Offset, A comes away at
        # 0.27 m/s on one heading and B at 0.92 m/s on another, and the split is
        # visible.
        "physics": {
            "launch_x": -0.744,
            "launch_y": 1.030,
            "launch_speed": 2.60,
            "launch_heading_deg": 303.74,
            "ball_b_x": 0.358,
            "ball_b_y": 0.660,
            "chest_restitution": 0.80,
            "chest_friction": 0.55,
            "floor_rolling_friction": 0.0060,
            "floor_spinning_friction": 0.008,
            "floor_friction": 0.60,
            "floor_restitution": 0.45,
            "ball_a_mass": 0.120,
            "ball_a_restitution": 0.88,
            "ball_a_friction": 1.0,
            "ball_b_mass": 0.090,
            "ball_b_restitution": 0.80,
            "ball_b_friction": 1.0,
            "gravity_z": -9.8,
            # Whether the target football is in the scene at all. A one-element
            # list rather than a bare int so the PCVE edit DSL can write it by
            # index, matching the other scenes' active flags. Set to [0] and
            # the ball is left out of both the simulation and the render, and
            # the rebound simply crosses empty floor.
            "ball_b_active": [1],
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

def import_room() -> tuple[bpy.types.Object, list[bpy.types.Object]]:
    """Import the toy room and drop it into the physics world frame.

    Rather than baking a transform into every one of the model's meshes, all of
    its root objects are parented to a single empty carrying the unit
    conversion and the shift onto the chest's north face. Parenting with an
    identity parent-inverse means each mesh's own transform composes with the
    empty's, so the model stays internally untouched and the mapping stays
    inspectable in the saved .blend -- and, more to the point, the collider
    export and the render read the exact same matrices.
    """
    if not ROOM_GLB.exists():
        raise FileNotFoundError(f"Room model not found: {ROOM_GLB}")
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(ROOM_GLB))
    imported = [o for o in bpy.context.scene.objects if o not in before]

    room_root = bpy.data.objects.new("room_root", None)
    bpy.context.scene.collection.objects.link(room_root)
    room_root.location = (
        -ROOM_SCALE * ROOM_ORIGIN_XY[0],
        -ROOM_SCALE * ROOM_ORIGIN_XY[1],
        -ROOM_SCALE * ROOM_GROUND_Z,
    )
    room_root.scale = (ROOM_SCALE, ROOM_SCALE, ROOM_SCALE)

    for obj in imported:
        if obj.parent is None:
            obj.parent = room_root
            obj.matrix_parent_inverse = Matrix.Identity(4)

    for name in ROOM_HIDDEN_OBJECTS:
        hidden = bpy.data.objects.get(name)
        if hidden is None:
            print(f"[WARN] {name!r} is not in the model; it may have changed.")
            continue
        hidden.hide_render = True
        hidden.hide_viewport = True

    bpy.context.view_layer.update()
    return room_root, imported


# --- Floor --------------------------------------------------------------------

def _image(asset: str, filename: str, colorspace: str):
    path = POLYHAVEN_DIR / asset / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Poly Haven texture not found: {path}\n"
            "Fetch it with: python3 scripts/download_render_assets.py"
        )
    image = bpy.data.images.load(str(path), check_existing=True)
    image.colorspace_settings.name = colorspace
    return image


def build_hardwood_floor() -> bpy.types.Object:
    """The floor the whole shot happens on, laid in place of the model's own.

    A 9 m plane at z = 0 carrying Poly Haven's 4K worn-pine boards. It runs far
    enough past the set in every direction that no camera the survey considered
    can see an edge, which the model's 3.11 x 2.60 m baked quad could not manage
    once the ball's run-up pushed the camera outside it.
    """
    bpy.ops.mesh.primitive_plane_add(size=FLOOR_SIZE, location=(0.0, 0.0, 0.0))
    floor = bpy.context.object
    floor.name = "hardwood_floor"

    mat = bpy.data.materials.new("hardwood_floor")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    set_input(bsdf, "Specular", 0.35)

    tex_coord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    repeats = FLOOR_SIZE / FLOOR_TEXTURE_SPAN
    mapping.inputs["Scale"].default_value = (repeats, repeats, 1.0)
    links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])

    def mapped(filename: str, colorspace: str):
        node = nodes.new("ShaderNodeTexImage")
        node.image = _image(FLOOR_ASSET, filename, colorspace)
        links.new(mapping.outputs["Vector"], node.inputs["Vector"])
        return node

    diffuse = mapped(f"{FLOOR_ASSET}_diff_4k.jpg", "sRGB")
    ao = mapped(f"{FLOOR_ASSET}_ao_4k.jpg", "Non-Color")
    ao_mix = nodes.new("ShaderNodeMixRGB")
    ao_mix.blend_type = "MULTIPLY"
    ao_mix.inputs["Fac"].default_value = 0.20
    links.new(diffuse.outputs["Color"], ao_mix.inputs["Color1"])
    links.new(ao.outputs["Color"], ao_mix.inputs["Color2"])
    links.new(ao_mix.outputs["Color"], bsdf.inputs["Base Color"])

    rough = mapped(f"{FLOOR_ASSET}_rough_4k.jpg", "Non-Color")
    links.new(rough.outputs["Color"], bsdf.inputs["Roughness"])

    normal = mapped(f"{FLOOR_ASSET}_nor_gl_4k.jpg", "Non-Color")
    normal_map = nodes.new("ShaderNodeNormalMap")
    normal_map.inputs["Strength"].default_value = 0.36
    links.new(normal.outputs["Color"], normal_map.inputs["Color"])

    # The board joints have to be a bump rather than real displacement: the
    # simulation rolls the balls on a perfectly flat z = 0 plane, so any actual
    # relief here would be relief the physics does not know about.
    height = mapped(f"{FLOOR_ASSET}_disp_4k.png", "Non-Color")
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.05
    bump.inputs["Distance"].default_value = 0.006
    links.new(height.outputs["Color"], bump.inputs["Height"])
    links.new(normal_map.outputs["Normal"], bump.inputs["Normal"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    floor.data.materials.append(mat)
    return floor


def export_collider(path: Path, object_names) -> Path:
    """Write the room's static collision geometry as a metre-scale OBJ.

    PyBullet and Blender have to agree about where the chest's north panel is to
    within a couple of millimetres, because a ball bouncing off that panel is
    the entire scene. Rather than hand-fitting a box -- which would have to
    guess whether a ball of a given radius meets the frame or the recessed panel
    behind it, and would silently be wrong when the radius changes -- the
    chest's own triangles are handed to Bullet through the same transform the
    render uses.

    Axis conversion is switched off (forward +Y, up +Z). Blender's OBJ exporter
    defaults to the Y-up convention, and PyBullet reads OBJ coordinates
    literally into its own Z-up world, so the default would lay the whole room
    on its side.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    missing = []
    for name in object_names:
        obj = bpy.data.objects.get(name)
        if obj is None:
            missing.append(name)
            continue
        obj.select_set(True)
    if missing:
        raise RuntimeError(
            f"Room model is missing collider objects {missing}; it may have changed."
        )
    bpy.ops.wm.obj_export(
        filepath=str(path),
        export_selected_objects=True,
        export_triangulated_mesh=True,
        export_materials=False,
        export_normals=False,
        export_uv=False,
        export_colors=False,
        apply_modifiers=True,
        forward_axis="Y",
        up_axis="Z",
    )
    bpy.ops.object.select_all(action="DESELECT")
    print(f"[INFO] Wrote room collision mesh: {path}")
    return path


def extract_ball(object_name: str, diameter: float, new_name: str) -> bpy.types.Object:
    """Lift one of the room's own balls out of the set and make it a free prop.

    The mesh keeps its material and its printing; only its place in the
    hierarchy, its size and its origin change. The origin has to move to the
    sphere's true centre: the physics frames give a centre position plus a spin
    quaternion, and a ball pivoting about anything else visibly wobbles as it
    rolls.
    """
    ball = bpy.data.objects.get(object_name)
    if ball is None:
        raise RuntimeError(f"Room model is missing {object_name!r}; it may have changed.")

    bpy.ops.object.select_all(action="DESELECT")
    ball.select_set(True)
    bpy.context.view_layer.objects.active = ball
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    mn, mx = world_bbox(ball)
    raw = ((mx.x - mn.x) + (mx.y - mn.y) + (mx.z - mn.z)) / 3.0
    ball.scale = (diameter / raw,) * 3
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    mn, mx = world_bbox(ball)
    bpy.context.scene.cursor.location = (
        (mn.x + mx.x) / 2.0, (mn.y + mx.y) / 2.0, (mn.z + mx.z) / 2.0,
    )
    bpy.ops.object.origin_set(type="ORIGIN_CURSOR")
    ball.location = (0.0, 0.0, 0.0)
    ball.name = new_name
    ball.rotation_mode = "QUATERNION"
    for poly in ball.data.polygons:
        poly.use_smooth = True
    bpy.ops.object.select_all(action="DESELECT")
    return ball


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


# --- Physics ------------------------------------------------------------------

def ball_b_active(physics: dict) -> bool:
    """Is the target football part of this scenario?

    Stored as a one-element list so the PCVE edit DSL can write it by index,
    but a hand-written scenario JSON may well say ``"ball_b_active": 0``.
    """
    value = physics.get("ball_b_active", [1])
    if isinstance(value, (list, tuple)):
        return bool(int(value[0])) if value else True
    return bool(int(value))


def run_physics(args: argparse.Namespace, scenario: dict) -> dict:
    physics_python = WORKSPACE_DIR.parent / "miniconda" / "envs" / "physics" / "bin" / "python"
    python = str(physics_python) if physics_python.exists() else (
        shutil.which("python3") or shutil.which("python")
    )
    if not python:
        raise RuntimeError("Cannot find python3/python for the PyBullet physics simulation.")

    ph = scenario["physics"]
    out = args.out_dir / PHYSICS_TEMP_NAME
    command = [
        python, str(Path(__file__).with_name("simulate_ball_box_rebound.py")),
        "--out", str(out),
        "--chest-collider", str(CHEST_COLLIDER_OBJ),
        "--props-collider", str(PROPS_COLLIDER_OBJ),
        "--fps", str(int(scenario["render"]["fps"])),
        "--duration-sec", str(float(scenario["render"]["duration_sec"])),
        "--ball-a-radius", str(BALL_A_DIAMETER / 2.0),
        "--ball-b-radius", str(BALL_B_DIAMETER / 2.0),
    ]
    for flag, key in (
        ("--launch-x", "launch_x"),
        ("--launch-y", "launch_y"),
        ("--launch-speed", "launch_speed"),
        ("--launch-heading-deg", "launch_heading_deg"),
        ("--ball-b-x", "ball_b_x"),
        ("--ball-b-y", "ball_b_y"),
        ("--chest-restitution", "chest_restitution"),
        ("--chest-friction", "chest_friction"),
        ("--floor-rolling-friction", "floor_rolling_friction"),
        ("--floor-spinning-friction", "floor_spinning_friction"),
        ("--floor-friction", "floor_friction"),
        ("--floor-restitution", "floor_restitution"),
        ("--ball-a-mass", "ball_a_mass"),
        ("--ball-a-restitution", "ball_a_restitution"),
        ("--ball-a-friction", "ball_a_friction"),
        ("--ball-b-mass", "ball_b_mass"),
        ("--ball-b-restitution", "ball_b_restitution"),
        ("--ball-b-friction", "ball_b_friction"),
        ("--gravity-z", "gravity_z"),
    ):
        command += [flag, str(float(ph[key]))]

    ball_b_present = ball_b_active(ph)
    if not ball_b_present:
        command += ["--disable-ball-b"]

    subprocess.run(command, check=True)
    data = json.loads(out.read_text(encoding="utf-8"))
    out.unlink(missing_ok=True)

    q = data["quality"]
    if not q["hit_chest"]:
        print("[WARN] The ball never reached the toy chest in this scenario.")
    # Only a miss is worth warning about. With the target ball deliberately
    # removed there is nothing to hit, and the warning would fire on every
    # frame of a case that is behaving exactly as intended.
    if ball_b_present and not q["hit_ball_b"]:
        print("[WARN] The rebounding ball missed the target ball.")
    if q["spinning_on_the_spot"]:
        print(f"[WARN] A ball is still turning where it stopped: "
              f"{q['residual_spin']} rad/s. Check floor_spinning_friction.")
    if q["hit_room_props"]:
        print(f"[WARN] A ball reached the lid/basket/bear at frame "
              f"{q['prop_contact_frame']}; only the chest is meant to be touched.")
    if q["left_floor"] is not None:
        # Not a missing floor -- the laid-in hardwood plane is 9 m across. What
        # the ball has left is the dressed room, and at this camera that means
        # it is on its way out of frame. Expected for cases where nothing stops
        # the ball, e.g. with the target football deleted.
        print(f"[NOTE] {q['left_floor']['ball']} left the dressed room area at "
              f"frame {q['left_floor']['frame']}; it is still on rendered floor "
              f"but is heading out of shot.")
    unsettled = q["ball_a_settled_frame"] is None or (
        ball_b_present and q["ball_b_settled_frame"] is None
    )
    if unsettled:
        print("[WARN] A ball was still moving at the last frame; it never settles on camera.")
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

    # World: an interior HDRI. The set is a corner -- it has a west wall and a
    # south wall and nothing else -- so whatever the world is, the camera sees
    # it past the open sides and through the window. A room HDRI reads as the
    # rest of the playroom; a flat colour reads as a void.
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
        mapping.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(200.0))
        env = nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(str(hdri), check_existing=True)
        links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], env.inputs["Vector"])
        links.new(env.outputs["Color"], bg.inputs["Color"])
        bg.inputs["Strength"].default_value = 1.5
    else:
        print(f"[WARN] HDRI not found at {hdri}; falling back to a flat sky colour.")
        bg.inputs["Color"].default_value = (0.66, 0.70, 0.76, 1.0)
        bg.inputs["Strength"].default_value = 2.0
    links.new(bg.outputs["Background"], output.inputs["Surface"])

    import_room()
    build_hardwood_floor()
    ball_a = extract_ball(BALL_A_OBJECT, BALL_A_DIAMETER, "ball_a_star")
    # A DELETE edit leaves the football out of the render entirely rather than
    # hiding it, so the .blend matches the video.
    ball_b = (
        extract_ball(BALL_B_OBJECT, BALL_B_DIAMETER, "ball_b_football")
        if ball_b_active(scenario["physics"])
        else None
    )
    apply_keyframes(ball_a, physics["frames"], "ball_a")
    if ball_b is not None:
        apply_keyframes(ball_b, physics["frames"], "ball_b")

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

    # Daylight through the window in the west wall, which is the only opening
    # the set actually has, plus a soft fill from the open north side so the
    # balls' shadow-facing halves do not go black against the dark chest.
    bpy.ops.object.light_add(type="SUN", location=(-2.6, -0.2, 2.2))
    sun = bpy.context.object
    sun.data.energy = 3.0
    sun.data.angle = math.radians(3.5)
    sun.data.color = (1.0, 0.96, 0.89)
    sun.rotation_euler = (math.radians(62.0), math.radians(0.0), math.radians(-72.0))

    add_area_light("window_light", (-1.75, 0.15, 1.15), power=150.0, size=1.5,
                   target=(0.0, 0.15, 0.10), color=(0.95, 0.97, 1.0))
    add_area_light("north_fill", (0.65, 1.55, 1.05), power=60.0, size=1.8,
                   target=(-0.05, 0.20, 0.08), color=(1.0, 0.98, 0.95))

    scene.frame_start = 1
    scene.frame_end = int(physics["frame_end"])
    scene.frame_set(1)
    return ball_a, ball_b, camera


# --- Outputs ------------------------------------------------------------------

def export_ground_truth(out_dir: Path, ball_a, ball_b, camera, physics: dict,
                        scenario: dict) -> None:
    scene = bpy.context.scene
    fps = int(physics["fps"])
    frame_end = int(physics["frame_end"])
    records = {
        "schema_version": 1,
        "fps": fps,
        "frame_start": 1,
        "frame_end": frame_end,
        "physics": {k: v for k, v in physics.items() if k != "frames"},
        "objects": {
            "ball_a": {"object_name": ball_a.name,
                       "radius": physics["objects"]["ball_a"]["radius"]},
            # A deleted football keeps its slot with present=false rather than
            # vanishing, so a consumer diffing an edit against its source sees
            # the same object keys on both sides.
            "ball_b": (
                {"present": False, "object_name": None}
                if ball_b is None
                else {"present": True, "object_name": ball_b.name,
                      "radius": physics["objects"]["ball_b"]["radius"]}
            ),
            "chest_face": {"slope": CHEST_FACE_SLOPE, "x_range": list(CHEST_FACE_X)},
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
        entry = {"frame_index": frame, "time_sec": (frame - 1) / float(fps),
                 "camera_matrix_world": [[float(v) for v in row] for row in camera.matrix_world]}
        present = [("ball_a", ball_a)] + ([("ball_b", ball_b)] if ball_b is not None else [])
        for key, obj in present:
            entry[key] = {
                "matrix_world": [[float(v) for v in row] for row in obj.matrix_world],
                "linear_velocity": pf[key]["linear_velocity"],
                "angular_velocity": pf[key]["angular_velocity"],
                "speed": pf[key]["speed"],
            }
        entry["ball_a"]["phase"] = pf["ball_a"]["phase"]
        if ball_b is None:
            entry["ball_b"] = {"present": False}
        else:
            entry["ball_b"]["present"] = True
            entry["ball_b"]["moving"] = pf["ball_b"]["moving"]
        records["frames"].append(entry)
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

    stale = not (CHEST_COLLIDER_OBJ.exists() and PROPS_COLLIDER_OBJ.exists())
    if args.mode == "colliders" or args.refresh_colliders or stale:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        import_room()
        export_collider(CHEST_COLLIDER_OBJ, CHEST_COLLIDER_OBJECTS)
        export_collider(PROPS_COLLIDER_OBJ, PROP_COLLIDER_OBJECTS)
        if args.mode == "colliders":
            return

    scenario = build_scenario(args)
    (args.out_dir / SCENARIO_METADATA_NAME).write_text(
        json.dumps(scenario, indent=2), encoding="utf-8",
    )

    physics = run_physics(args, scenario)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    ball_a, ball_b, camera = build_scene(args, scenario, physics)
    export_ground_truth(args.out_dir, ball_a, ball_b, camera, physics, scenario)

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
