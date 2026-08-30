"""Blender/Cycles renderer for the big/small marble collision on a bar table.

Imports assets/models/modern_living_room.glb, calibrates it into the physics
world frame, puts two glass marbles on the kitchen's bar table, runs
simulate_table_marble_collision.py and replays the result as keyframes.

Run through Blender, not python:

    blender -b --python render_table_marble_collision.py -- --out-dir ...
"""

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

ROOM_GLB = MODELS_DIR / "modern_living_room.glb"
MARBLE_GLB = MODELS_DIR / "marble_yellow_ball.glb"
PROPS_COLLIDER_OBJ = COLLISION_DIR / "modern_living_room_table_dressing.obj"

OUTPUT_STEM = "table_marble_collision"
BLEND_NAME = f"{OUTPUT_STEM}.blend"
PHYSICS_TEMP_NAME = "physics_transforms.json"
GROUND_TRUTH_NAME = "ground_truth_transforms.json"
SCENARIO_METADATA_NAME = "scenario_metadata.json"


# --- Room model calibration ---------------------------------------------------
#
# assets/models/modern_living_room.glb is a SketchUp-style export: it is not in
# metres, and every one of its 124 meshes is named "Material2.0NN", so nothing
# in it can be found by name without measuring it first. Everything below was
# measured by raycasting the imported model, and the constants are checked
# against each other by VERIFY_OBJECTS at import time.
#
# The unit is fixed by the one dimension in a kitchen that is standardised: the
# worktop. It sits at model z = 37.219, and the bar table's top is the *same*
# surface -- the two are flush, and a raycast walks from one to the other with no
# step at all -- so taking the worktop as 0.90 m fixes the scale.
#
# Four independent checks agree, which is why this is the number and not a
# guess:
#   - the interior ceiling lands at 2.571 m, and the wall cabinets run exactly
#     up to it (they are modelled to the ceiling, as fitted cabinets are);
#   - the window's sill lands at 1.058 m and its head at 2.010 m;
#   - the coffee table's top lands at 0.404 m and the sofa's seat at 0.421 m;
#   - the bar stools' seats land at 0.718 m, which is a bar stool for a 0.90 m
#     counter.
ROOM_TABLE_UNITS = 37.219
TABLE_HEIGHT_M = 0.900
ROOM_SCALE = TABLE_HEIGHT_M / ROOM_TABLE_UNITS

# The world frame is the natural one for this scene, so that the simulation's
# output needs no transform at all before being keyframed:
#   z = 0    is the bar table's top surface,
#   (0, 0)   is the centre of that surface,
#   +x       runs east along the table's 1.83 m length,
#   +y       runs north, out of the wall the table stands against and toward the
#            bar stools.
ROOM_ORIGIN_XY = (419.249, 410.898)

# The table top, in world metres: 1.830 x 0.579 m, and dead flat -- 1701 probe
# points over the clear area came back at the same z to four decimals. Both
# figures are re-derived from the model at import time and warned about if the
# model has changed under us.
TABLE_X = (-0.9152, 0.9152)
TABLE_Y = (-0.2897, 0.2897)

# Object names this scene depends on, with the world-metre bounding box each
# one is expected to occupy. The names are meaningless, so the box is the real
# identity check: if the model is ever replaced or re-exported these will not
# line up and the render says so instead of quietly putting marbles in mid-air.
VERIFY_OBJECTS = {
    "Material3.012": ("bar table top (wood)", (-0.915, 0.915), (-0.290, 0.290), (-0.900, 0.000)),
    "Material2.083": ("bar table frame (metal)", (-0.915, 0.915), (-0.290, 0.290), (-0.900, -0.051)),
    "Material2.022": ("kitchen worktop", (-3.993, -0.915), (-0.299, 1.918), (-0.900, 0.000)),
    "Material2.002": ("ceramic floor", (-4.721, 4.327), (-2.425, 6.924), (-0.900, -0.900)),
    "Material2.077": ("wooden tray", (-0.103, 0.163), (-0.244, -0.147), (0.000, 0.015)),
    "Material2.078": ("candlestick stems", (-0.068, 0.126), (-0.220, -0.171), (0.015, 0.198)),
    "Material2.079": ("candles", (-0.053, 0.112), (-0.205, -0.187), (0.198, 0.256)),
    "Material2.080": ("bar stools", (-0.780, 0.929), (0.206, 0.823), (-0.900, -0.106)),
    "Material2.043": ("luminous ceiling panel", (-3.255, 2.663), (-0.350, 2.473), (1.326, 1.726)),
}
VERIFY_TOLERANCE = 0.010

# The only static geometry a marble on the table can reach: a shallow wooden
# tray with two brass candlesticks standing on it, a third of the way along the
# table and hugging the wall side. It is exported as its own PyBullet body, not
# merged into the table, so that touching it is reported as its own event -- the
# hero take passes 0.117 m clear, and a ball that reaches it has gone somewhere
# the shot does not intend.
#
# The dressing is *kept*. It was tempting to hide it and have the whole table to
# aim across, but it is the only thing in the shot that gives the table's surface
# a scale and a foreground, and moving the action into the table's east half
# costs nothing: there is still 1.0 m of clear run east of the candlesticks,
# which is more than the shot needs.
PROP_COLLIDER_OBJECTS = ("Material2.077", "Material3.008", "Material2.078", "Material2.079")

# The room's ceiling carries a large recessed luminous panel: a housing, a warm
# yellow face and a grey translucent diffuser, together spanning nearly the whole
# room directly above the table. As authored it is an ordinary diffuse material
# with no emission, which is this model's version of the fake-light-card problem
# -- Cycles renders what it geometrically is, a dull mustard slab across the
# ceiling, and the room has no light in it at all.
#
# The face is given a real emission shader (on its own copy of the material, so
# nothing else that shares it is affected) and the diffuser is hidden: at alpha
# 0.5 it halves the panel's output and greys everything under it, including the
# table top, for no gain the camera can see.
CEILING_PANEL_FACE = "Material2.043"
CEILING_PANEL_DIFFUSER = "Material2.095"
CEILING_PANEL_COLOR = (1.0, 0.955, 0.88)

# The two marbles, both cut from assets/models/marble_yellow_ball.glb. Same
# glass, one twice the diameter of the other -- which is the whole scene: it
# makes the big one eight times the mass, and eight to one is what sends the
# small one away faster than the big one arrived.
#
# 100 mm is a large marble and 50 mm an ordinary one; on a 0.579 m deep table
# the big one covers a sixth of the depth, which is big enough to read as an
# object rolling on a surface rather than a dot on a plan.
BALL_A_DIAMETER = 0.100
BALL_B_DIAMETER = 0.050

# Framing: a low three-quarter from the room's north-east, 1.17 m above the
# floor -- 0.27 m above the table top -- chosen from a survey of twelve positions
# (kept in renders/tmc_camera_survey/, cam_survey.py regenerates it).
#
# Every candidate stands north of the table, because that is the only side there
# is: the south edge is against a wall, the east end is 67 mm from a wardrobe,
# and the west end runs into the kitchen worktop.
#
# The angle was picked for one property the prettier candidates do not have: the
# whole shot stays in frame from the launch to both marbles at rest. The view
# down the table's length from the east end had the best depth of the twelve --
# the top receding toward the kitchen with a daylit window behind it -- but the
# struck marble rolls *toward* that camera and leaves frame before it stops, so
# the last 15 frames have only one ball in them. For a reconstruction clip the
# two balls' relative position is the content, and it has to be legible in the
# last frame as much as the first.
#
# The distance was then set by a second round that walked the camera in along
# this sight line. At 35 per cent closer the marbles read about twice the size
# and the frame stops being mostly bare wall, but the launch clipped the right
# edge. **That was fixed by shortening the run-up rather than by backing the
# camera off** -- `launch_x` moved from -0.780 to -0.620 with the push dropped
# from 1.42 to 1.31 m/s to compensate, which puts the marble in frame at the
# launch and reproduces the old arrival speed to 2 per cent and both resting
# positions to 2 cm. Moving the subject is nearly always the cheaper fix; the
# camera position is load-bearing and the run-up was not.
#
# Half the standoff was also rendered and is the best-looking of the lot -- the
# impact fills the frame and the top's reflection comes alive -- but no amount of
# shortening the run-up puts the launch back in it, and a collision scene that
# does not show the ball being sent has given away its premise.
#
# It is also low on purpose. The subject is two balls 100 and 50 mm across on a
# horizontal surface; from standing height they flatten into dots on a plan, and
# what makes them read as objects rolling on a table is being near the height of
# the table itself. The bar stools help here rather than getting in the way --
# their backs stop 0.106 m *below* the table top, so they read as a foreground
# frieze without ever crossing the line the marbles roll along.
CAMERA_LOCATION = (0.58, 1.40, 0.24)
CAMERA_TARGET = (0.08, 0.00, 0.03)
CAMERA_LENS_MM = 35.0


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("preview", "animation", "frames", "colliders"),
        default="animation",
        help="'colliders' only rebuilds assets/collision/*.obj and exits, which "
        "is all simulate_table_marble_collision.py needs to run standalone.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resolution", nargs=2, type=int, default=(1280, 720))
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-sec", type=float, default=3.0)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--preview-frame", type=int, default=26)
    parser.add_argument(
        "--preview-frames", nargs="+", type=int, default=None,
        help="Render several stills instead of one, as preview_f<N>.png. "
        "Preview mode only; this is what cam_survey.py drives.",
    )
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--refresh-colliders", action="store_true")
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
        scenario = json.loads(args.scenario_json.read_text(encoding="utf-8"))
    else:
        scenario = {
            "schema_version": 1,
            "seed": int(args.seed),
            "scene": OUTPUT_STEM,
            "render": {
                "mode": str(args.mode),
                "fps": int(args.fps),
                "duration_sec": float(args.duration_sec),
                "resolution": [int(args.resolution[0]), int(args.resolution[1])],
                "samples": int(args.samples),
                "device": str(args.device),
                # A real 24 fps camera has its shutter open for about half a
                # frame, and this scene needs that: the struck marble leaves at
                # 1.4 m/s, which is 58 mm per frame against its own 50 mm
                # diameter. Without blur it reads as a ball teleporting.
                "motion_blur": True,
                "motion_blur_shutter": 0.5,
            },
            # Hue is a rotation applied to the marble model's own swirl texture
            # rather than a flat colour: 0.5 leaves it as authored and the offset
            # from 0.5 is the rotation. The swirl is doing real work -- a plain
            # sphere is rotationally invariant on screen, and the swirl is the
            # only thing that shows these marbles arrive *rolling*.
            #
            # Deliberately no diameter here. The size is a physics parameter --
            # it is what sets the mass, which is the entire scene -- so it lives
            # in the physics block only and the rendered marbles are sized from
            # what the simulation reports back. Carrying it in two places is how
            # a suite case ends up simulating a 70 mm ball and rendering a 50 mm
            # one, which no amount of looking at the video would reveal.
            "balls": {
                "a": {"hue": 0.50, "saturation": 1.05},
                "b": {"hue": 0.95, "saturation": 1.25},
            },
            "camera": {
                "location": list(CAMERA_LOCATION),
                "target": list(CAMERA_TARGET),
                "lens_mm": CAMERA_LENS_MM,
            },
            # Kept in the scenario rather than hard-coded because the flat's own
            # lighting had to be built from nothing -- the model ships no lights
            # at all -- and the balance between the ceiling panel and the window
            # was found by rendering it, not by reasoning about it.
            "lighting": {
                "ceiling_panel_strength": 9.0,
                "world_strength": 0.9,
                "sun_energy": 2.6,
                "window_bounce_power": 90.0,
                "bar_fill_power": 26.0,
            },
            # Tuned in simulate_table_marble_collision.py; see its module
            # comment for what each one is for. Masses are left out on purpose:
            # they are derived from the radii at glass density, because the point
            # of the scene is that the mass ratio follows from the size ratio.
            "physics": {
                "launch_x": -0.620,
                "launch_y": 0.020,
                "launch_speed": 1.31,
                "launch_heading_deg": 0.0,
                "ball_a_radius": BALL_A_DIAMETER / 2.0,
                "ball_b_radius": BALL_B_DIAMETER / 2.0,
                # Baseline glass-density masses (solid sphere, 2500 kg/m^3
                # -- same as the sim's default when --ball-*-mass is not
                # given). Made explicit here so a PCVE `mass` edit has a
                # baseline to diff against and so the scenario snapshot
                # matches the value the sim actually uses.
                "ball_a_mass": (4.0 / 3.0) * math.pi * (BALL_A_DIAMETER / 2.0) ** 3 * 2500.0,
                "ball_b_mass": (4.0 / 3.0) * math.pi * (BALL_B_DIAMETER / 2.0) ** 3 * 2500.0,
                # Two-slot presence list (big_marble, small_marble). A PCVE
                # DELETE edit writes 0 at the marble's slot; only slot 1
                # (small) is actually removable -- the big marble drives
                # the sim so it is always kept.
                "active": [1, 1],
                "ball_b_x": 0.060,
                "ball_b_y": 0.008,
                "ball_a_restitution": 0.87,
                "ball_b_restitution": 0.87,
                "ball_a_friction": 0.30,
                "ball_b_friction": 0.30,
                "table_friction": 0.42,
                "table_restitution": 0.30,
                "table_rolling_friction": 0.0200,
                "table_spinning_friction": 0.006,
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
    return (Vector((min(c.x for c in corners), min(c.y for c in corners),
                    min(c.z for c in corners))),
            Vector((max(c.x for c in corners), max(c.y for c in corners),
                    max(c.z for c in corners))))


def look_at(obj: bpy.types.Object, target) -> None:
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_area_light(name, location, power, size, target, color=(1.0, 1.0, 1.0)):
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

def import_room() -> bpy.types.Object:
    """Import the flat and drop it into the physics world frame.

    Rather than baking a transform into every one of the model's 124 meshes, all
    of its root objects are parented to a single empty carrying the unit
    conversion and the shift onto the table top. Parenting with an identity
    parent-inverse means each mesh's own transform composes with the empty's, so
    the model stays internally untouched, the mapping stays inspectable in the
    saved .blend, and -- the point -- the collider export and the render read the
    exact same matrices.
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
        -ROOM_SCALE * ROOM_TABLE_UNITS,
    )
    room_root.scale = (ROOM_SCALE, ROOM_SCALE, ROOM_SCALE)

    for obj in imported:
        if obj.parent is None:
            obj.parent = room_root
            obj.matrix_parent_inverse = Matrix.Identity(4)

    bpy.context.view_layer.update()
    verify_room()
    return room_root


def verify_room() -> None:
    """Check the calibration against the model instead of trusting it.

    Every object this scene reaches for is named "Material2.0NN", so a name is
    no evidence at all that the right mesh has been found. Each one's world
    bounding box is compared against the box it was measured in; a mismatch
    means the model has been replaced or re-exported and every constant in this
    file needs re-deriving.
    """
    for name, (label, xs, ys, zs) in VERIFY_OBJECTS.items():
        obj = bpy.data.objects.get(name)
        if obj is None:
            print(f"[WARN] {name!r} ({label}) is not in the model; it may have changed.")
            continue
        mn, mx = world_bbox(obj)
        got = ((mn.x, mx.x), (mn.y, mx.y), (mn.z, mx.z))
        drift = max(abs(g - e) for pair, exp in zip(got, (xs, ys, zs))
                    for g, e in zip(pair, exp))
        if drift > VERIFY_TOLERANCE:
            print(f"[WARN] {name!r} ({label}) is {drift * 1000:.0f} mm from where "
                  f"it was measured: x {got[0][0]:+.3f}..{got[0][1]:+.3f} "
                  f"y {got[1][0]:+.3f}..{got[1][1]:+.3f} "
                  f"z {got[2][0]:+.3f}..{got[2][1]:+.3f}, expected "
                  f"x {xs[0]:+.3f}..{xs[1]:+.3f} y {ys[0]:+.3f}..{ys[1]:+.3f} "
                  f"z {zs[0]:+.3f}..{zs[1]:+.3f}. The calibration in this file "
                  "no longer describes the model.")

    table = bpy.data.objects.get("Material3.012")
    if table is not None:
        mn, mx = world_bbox(table)
        if abs(mx.z) > 0.002:
            print(f"[WARN] The bar table's top is at z = {mx.z:+.4f}, not 0. "
                  f"ROOM_SCALE or ROOM_TABLE_UNITS is wrong.")
        print(f"[INFO] Bar table top: {mx.x - mn.x:.3f} x {mx.y - mn.y:.3f} m at "
              f"z = {mx.z:+.4f}; scale {ROOM_SCALE:.7f} "
              f"({TABLE_HEIGHT_M:.3f} m / {ROOM_TABLE_UNITS} units).")


def light_the_ceiling_panel(strength: float) -> None:
    """Turn the ceiling's luminous panel into an actual light.

    See CEILING_PANEL_FACE. The material is copied first: material_46 is shared,
    and rewriting the shared datablock would set fire to anything else using it.
    """
    face = bpy.data.objects.get(CEILING_PANEL_FACE)
    if face is None or not face.data.materials:
        print(f"[WARN] {CEILING_PANEL_FACE!r} (luminous ceiling panel) is not in "
              "the model or carries no material; the room will be lit only by "
              "the window.")
        return
    material = face.data.materials[0].copy()
    material.name = "ceiling_panel_emission"
    face.data.materials[0] = material
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    for node in list(nodes):
        nodes.remove(node)
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*CEILING_PANEL_COLOR, 1.0)
    emission.inputs["Strength"].default_value = float(strength)
    links.new(emission.outputs["Emission"], output.inputs["Surface"])

    diffuser = bpy.data.objects.get(CEILING_PANEL_DIFFUSER)
    if diffuser is not None:
        diffuser.hide_render = True
        diffuser.hide_viewport = True
    mn, mx = world_bbox(face)
    print(f"[INFO] Luminous ceiling panel lit at strength {strength}: "
          f"{mx.x - mn.x:.2f} x {mx.y - mn.y:.2f} m at z {mn.z:+.3f}..{mx.z:+.3f}; "
          f"diffuser {'hidden' if diffuser is not None else 'not found'}.")


def polish_table_top() -> None:
    """Give the table top the sheen a lacquered wooden top has.

    The export flattens every material in the flat to metallic 0, roughness 0.6,
    which on the one surface the camera is looking straight down at reads as
    unfinished chipboard. Only the specular and roughness are touched -- the
    model's own wood texture is what should be seen.
    """
    top = bpy.data.objects.get("Material3.012")
    if top is None or not top.data.materials:
        print("[WARN] The bar table's top carries no material to polish.")
        return
    material = top.data.materials[0].copy()
    material.name = "bar_table_top"
    top.data.materials[0] = material
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is None:
        print("[WARN] The bar table's top has no Principled BSDF to polish.")
        return
    set_input(bsdf, "Roughness", 0.28)
    set_input(bsdf, "Specular", 0.55)


# --- Marbles ------------------------------------------------------------------

def import_marble(name: str, diameter: float, hue: float, saturation: float):
    """Import the marble model, size it, centre it, and recolour it.

    Lifted from twin_ramp_collision, which established the pattern: the model
    ships as one sphere carrying a photographed swirl on an image texture, and
    the colour is changed by rotating the hue of that texture rather than by
    overwriting the base colour, which would throw the swirl away along with the
    yellow. Each marble gets its own copy of the material, otherwise both would
    share one datablock and recolouring the second would recolour the first.
    """
    if not MARBLE_GLB.exists():
        raise FileNotFoundError(f"Marble model not found: {MARBLE_GLB}")

    scene = bpy.context.scene
    before = set(scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(MARBLE_GLB))
    imported = [o for o in scene.objects if o not in before]
    meshes = [o for o in imported if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"{MARBLE_GLB.name} imported no mesh.")

    ball = meshes[0]
    ball.name = name
    matrix = ball.matrix_world.copy()
    ball.parent = None
    ball.matrix_world = matrix
    for obj in imported:
        if obj is not ball:
            bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = ball
    ball.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # Scale to the physical diameter, then put the origin at the sphere's
    # centre. The simulation hands over centre positions and spin quaternions,
    # so the object's origin has to *be* the centre or the marble visibly orbits
    # its own offset as it rolls.
    mn, mx = world_bbox(ball)
    span = max(mx.z - mn.z, 1e-9)
    ball.scale = (float(diameter) / span,) * 3
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    ball.location = (0.0, 0.0, 0.0)
    ball.select_set(False)

    material = ball.data.materials[0].copy() if ball.data.materials else None
    if material is None:
        raise RuntimeError(f"{MARBLE_GLB.name} carries no material to recolour.")
    material.name = f"{name}_glass"
    ball.data.materials[0] = material
    nodes, links = material.node_tree.nodes, material.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    base = bsdf.inputs.get("Base Color") if bsdf is not None else None
    if base is not None and base.is_linked:
        source = base.links[0].from_socket
        hue_sat = nodes.new("ShaderNodeHueSaturation")
        hue_sat.inputs["Hue"].default_value = float(hue)
        hue_sat.inputs["Saturation"].default_value = float(saturation)
        links.new(source, hue_sat.inputs["Color"])
        links.new(hue_sat.outputs["Color"], base)
    else:
        print(f"[WARN] {name}: base colour is not textured; hue shift skipped.")

    for poly in ball.data.polygons:
        poly.use_smooth = True
    ball.rotation_mode = "QUATERNION"
    print(f"[INFO] Marble {name}: {span:.4f} model units -> {diameter:.3f} m, hue {hue}")
    return ball


# --- Colliders ----------------------------------------------------------------

def export_collider(path: Path, object_names) -> Path:
    """Write the table's dressing as a metre-scale OBJ for PyBullet.

    Handing over the props' own triangles through the same transform the render
    uses is the only way the two can be guaranteed to agree about where a
    candlestick is, and the candlesticks are turned, not boxy.

    Axis conversion is switched off (forward +Y, up +Z). Blender's OBJ exporter
    defaults to the Y-up convention and PyBullet reads OBJ coordinates literally
    into its own Z-up world, so the default would lay the whole table on its
    side.
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
    print(f"[INFO] Wrote table-dressing collision mesh: {path}")
    return path


# --- Physics ------------------------------------------------------------------

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


def run_physics(args: argparse.Namespace, scenario: dict) -> dict:
    physics_python = WORKSPACE_DIR.parent / "miniconda" / "envs" / "physics" / "bin" / "python"
    python = str(physics_python) if physics_python.exists() else (
        shutil.which("python3") or shutil.which("python")
    )
    if not python:
        raise RuntimeError("Cannot find python3/python for the PyBullet simulation.")

    ph = scenario["physics"]
    out = args.out_dir / PHYSICS_TEMP_NAME
    command = [
        python, str(Path(__file__).with_name("simulate_table_marble_collision.py")),
        "--out", str(out),
        "--props-collider", str(PROPS_COLLIDER_OBJ),
        "--fps", str(int(scenario["render"]["fps"])),
        "--duration-sec", str(float(scenario["render"]["duration_sec"])),
    ]
    for flag, key in (
        ("--ball-a-radius", "ball_a_radius"),
        ("--ball-b-radius", "ball_b_radius"),
        ("--launch-x", "launch_x"),
        ("--launch-y", "launch_y"),
        ("--launch-speed", "launch_speed"),
        ("--launch-heading-deg", "launch_heading_deg"),
        ("--ball-b-x", "ball_b_x"),
        ("--ball-b-y", "ball_b_y"),
        ("--ball-a-restitution", "ball_a_restitution"),
        ("--ball-b-restitution", "ball_b_restitution"),
        ("--ball-a-friction", "ball_a_friction"),
        ("--ball-b-friction", "ball_b_friction"),
        ("--table-friction", "table_friction"),
        ("--table-restitution", "table_restitution"),
        ("--table-rolling-friction", "table_rolling_friction"),
        ("--table-spinning-friction", "table_spinning_friction"),
        ("--gravity-z", "gravity_z"),
    ):
        command += [flag, str(float(ph[key]))]
    # Masses are always passed explicitly now (the scenario baseline holds
    # the glass-density defaults), so a PCVE `mass` edit has a well-defined
    # from-value and there is no ambiguity between "derived" and "overridden".
    for flag, key in (("--ball-a-mass", "ball_a_mass"), ("--ball-b-mass", "ball_b_mass")):
        if ph.get(key) is not None:
            command += [flag, str(float(ph[key]))]

    # Ball A is always present; only ball B is optional (the "target" that
    # a DELETE edit removes).
    active = ph.get("active", [1, 1])
    command += ["--ball-b-active", str(int(active[1]))]

    subprocess.run(command, check=True)
    data = json.loads(out.read_text(encoding="utf-8"))
    out.unlink(missing_ok=True)

    q, c = data["quality"], data["collision"]
    if not q["hit_ball_b"]:
        print("[WARN] The big marble never reached the small one in this scenario.")
    if q["spinning_on_the_spot"]:
        print(f"[WARN] A marble is still turning where it stopped: "
              f"{q['residual_spin']} rad/s. Check table_spinning_friction.")
    if q["hit_table_dressing"]:
        print(f"[WARN] A marble reached the tray or the candlesticks at frame "
              f"{q['prop_contact_frame']}; the shot is meant to pass clear of them.")
    if q["left_table"] is not None:
        print(f"[WARN] {q['left_table']['ball']} left the table at frame "
              f"{q['left_table']['frame']} over the {q['left_table']['edge']} edge.")
    if q["ball_a_settled_frame"] is None or q["ball_b_settled_frame"] is None:
        print("[WARN] A marble was still moving on the last frame; it never "
              "settles on camera.")
    # The scene's standing check: the solver against the closed form. A quiet
    # disagreement here is the single most likely way for this scene to be
    # wrong, so it is compared on every render rather than once during tuning.
    for label, measured, predicted in (
        ("struck marble", c["measured_b_speed_ratio"], c["predicted"]["b_speed_ratio"]),
        ("big marble", c["measured_a_speed_ratio"], c["predicted"]["a_speed_ratio"]),
    ):
        if measured is None:
            continue
        # The absolute floor is not slack: when the closed form predicts the big
        # marble nearly stopping, what it has left is mostly what friction and
        # spin did to it during the contact, which no rigid-impact formula
        # models. A 0.02 disagreement there is expected; a 15 per cent one on a
        # ball that is still moving is not.
        if abs(measured - predicted) > 0.15 * predicted + 0.02:
            print(f"[WARN] The {label} left at {measured:.3f} x the approach "
                  f"speed, but the closed form for a {c['mass_ratio']:.2f}:1 "
                  f"impact at e = {c['effective_restitution']:.2f} and "
                  f"{c['obliquity_deg']:.1f} deg predicts {predicted:.3f}. "
                  "One of them is wrong.")
    # The measured ratios only exist if the marbles actually met. A scenario
    # where they do not is a legitimate one to render -- the soft-push edit is
    # exactly that, and the warning above has already said so -- so report the
    # geometry and leave the closed form uncompared rather than formatting None.
    if c["measured_b_speed_ratio"] is None or c["measured_a_speed_ratio"] is None:
        print(f"[INFO] {c['size_ratio']:.2f}:1 in diameter is "
              f"{c['mass_ratio']:.2f}:1 in mass; no contact this run, so there "
              f"is no measured speed ratio to check the closed form against.")
    else:
        print(f"[INFO] {c['size_ratio']:.2f}:1 in diameter is {c['mass_ratio']:.2f}:1 in "
              f"mass; the struck marble left at {c['measured_b_speed_ratio']:.3f} x the "
              f"approach speed (closed form {c['predicted']['b_speed_ratio']:.3f}) and "
              f"the big one kept {c['measured_a_speed_ratio']:.3f} "
              f"({c['predicted']['a_speed_ratio']:.3f}).")
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
    scene.cycles.max_bounces = 12
    scene.cycles.transmission_bounces = 12
    scene.render.use_motion_blur = bool(render.get("motion_blur", True))
    scene.render.motion_blur_shutter = float(render.get("motion_blur_shutter", 0.5))
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

    # World: a sky, not a room. The flat has one large glazed wall 4.2 m west of
    # the table, and what that window sees is outdoors -- an interior HDRI behind
    # it reads as a second room hanging in the air. Kept dim, because the
    # daylight itself comes from the sun below and the room's own light from the
    # ceiling panel; this is here to fill the window and give the marbles
    # something to reflect.
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    for node in list(nodes):
        nodes.remove(node)
    lighting = scenario.get("lighting", {})
    output = nodes.new("ShaderNodeOutputWorld")
    bg = nodes.new("ShaderNodeBackground")
    hdri = POLYHAVEN_DIR / "outdoor" / "kloofendal_48d_partly_cloudy_puresky_2k.hdr"
    if hdri.exists():
        tex_coord = nodes.new("ShaderNodeTexCoord")
        mapping = nodes.new("ShaderNodeMapping")
        mapping.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(95.0))
        env = nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(str(hdri), check_existing=True)
        links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], env.inputs["Vector"])
        links.new(env.outputs["Color"], bg.inputs["Color"])
        bg.inputs["Strength"].default_value = float(lighting.get("world_strength", 0.9))
    else:
        print(f"[WARN] HDRI not found at {hdri}; falling back to a flat sky colour.")
        bg.inputs["Color"].default_value = (0.62, 0.70, 0.82, 1.0)
        bg.inputs["Strength"].default_value = float(lighting.get("world_strength", 0.9)) * 1.8
    links.new(bg.outputs["Background"], output.inputs["Surface"])

    import_room()
    light_the_ceiling_panel(float(lighting.get("ceiling_panel_strength", 9.0)))
    polish_table_top()

    # Sized from what the simulation actually used, not from the scenario, so
    # the two cannot disagree.
    cfg = scenario["balls"]
    ball_a = import_marble("marble_big",
                           2.0 * float(physics["objects"]["ball_a"]["radius"]),
                           float(cfg["a"]["hue"]), float(cfg["a"]["saturation"]))
    ball_b = import_marble("marble_small",
                           2.0 * float(physics["objects"]["ball_b"]["radius"]),
                           float(cfg["b"]["hue"]), float(cfg["b"]["saturation"]))
    apply_keyframes(ball_a, physics["frames"], "ball_a")
    # DELETE edit: sim skipped creating ball_b (present=false, no per-frame
    # data). Hide the mesh from the render instead of trying to animate it.
    if physics["objects"]["ball_b"].get("present", True):
        apply_keyframes(ball_b, physics["frames"], "ball_b")
    else:
        ball_b.hide_viewport = True
        ball_b.hide_render = True

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

    # Daylight through the glazed west wall, aimed along it so the light rakes
    # down the table's length rather than flooding it flat. The table stands
    # against the room's south wall, so this is the only direction real light can
    # come from, and it is what puts a highlight on the marbles and a shadow
    # under them -- the ceiling panel alone is a big soft box directly overhead
    # and leaves both balls sitting in their own shade with no contact shadow to
    # say they are touching the table.
    bpy.ops.object.light_add(type="SUN", location=(-4.0, 1.6, 1.4))
    sun = bpy.context.object
    sun.data.energy = float(lighting.get("sun_energy", 2.6))
    sun.data.angle = math.radians(2.8)
    sun.data.color = (1.0, 0.955, 0.90)
    sun.rotation_euler = (math.radians(66.0), 0.0, math.radians(-104.0))

    add_area_light("window_bounce", (-2.10, 0.55, 0.60),
                   power=float(lighting.get("window_bounce_power", 90.0)), size=2.2,
                   target=(0.05, 0.00, 0.00), color=(0.94, 0.965, 1.0))
    # A small light on the open side, at the height of a person standing at the
    # bar. Without it the marbles' camera-facing halves fall away into the dark
    # wood, because everything else in the room lights them from behind.
    add_area_light("bar_fill", (0.55, 1.25, 0.42),
                   power=float(lighting.get("bar_fill_power", 26.0)), size=0.9,
                   target=(0.00, 0.00, 0.02), color=(1.0, 0.98, 0.95))

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
        "scene": OUTPUT_STEM,
        "fps": fps,
        "frame_start": 1,
        "frame_end": frame_end,
        "physics": {k: v for k, v in physics.items() if k != "frames"},
        "objects": {
            "ball_a": {"object_name": ball_a.name,
                       "radius": physics["objects"]["ball_a"]["radius"],
                       "mass": physics["objects"]["ball_a"]["mass"]},
            "ball_b": {"present": bool(physics["objects"]["ball_b"].get("present", True)),
                       "object_name": ball_b.name,
                       "radius": physics["objects"]["ball_b"]["radius"],
                       "mass": physics["objects"]["ball_b"]["mass"]},
            "table": {"footprint_x": list(TABLE_X), "footprint_y": list(TABLE_Y),
                      "top_z": 0.0, "height_m": TABLE_HEIGHT_M},
        },
        "room": {
            "model": ROOM_GLB.name,
            "scale": ROOM_SCALE,
            "origin_xy_model_units": list(ROOM_ORIGIN_XY),
            "table_top_model_units": ROOM_TABLE_UNITS,
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
    ball_b_present = physics["objects"]["ball_b"].get("present", True)
    by_frame = {int(fr["frame_index"]): fr for fr in physics["frames"]}
    for frame in range(1, frame_end + 1):
        scene.frame_set(frame)
        pf = by_frame[frame]
        entry = {"frame_index": frame, "time_sec": (frame - 1) / float(fps),
                 "camera_matrix_world": [[float(v) for v in row]
                                         for row in camera.matrix_world]}
        pairs = [("ball_a", ball_a)]
        if ball_b_present:
            pairs.append(("ball_b", ball_b))
        for key, obj in pairs:
            entry[key] = {
                "matrix_world": [[float(v) for v in row] for row in obj.matrix_world],
                "linear_velocity": pf[key]["linear_velocity"],
                "angular_velocity": pf[key]["angular_velocity"],
                "speed": pf[key]["speed"],
                "spin_z": pf[key]["spin_z"],
                "on_table": pf[key]["on_table"],
            }
        if not ball_b_present:
            entry["ball_b"] = {"present": False}
        entry["ball_a"]["phase"] = pf["ball_a"]["phase"]
        if ball_b_present:
            entry["ball_b"]["moving"] = pf["ball_b"]["moving"]
        records["frames"].append(entry)
    (out_dir / GROUND_TRUTH_NAME).write_text(json.dumps(records, indent=2), encoding="utf-8")


def render_preview(args: argparse.Namespace) -> None:
    scene = bpy.context.scene
    scene.render.image_settings.file_format = "PNG"
    if args.preview_frames:
        for frame in args.preview_frames:
            scene.frame_set(max(scene.frame_start, min(int(frame), scene.frame_end)))
            scene.render.filepath = str(args.out_dir / f"preview_f{int(frame):03d}.png")
            bpy.ops.render.render(write_still=True)
        return
    scene.frame_set(max(scene.frame_start, min(int(args.preview_frame), scene.frame_end)))
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

    stale = not PROPS_COLLIDER_OBJ.exists()
    if args.mode == "colliders" or args.refresh_colliders or stale:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        import_room()
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
