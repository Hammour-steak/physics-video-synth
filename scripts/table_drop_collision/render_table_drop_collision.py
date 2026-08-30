"""Blender/Cycles renderer for a tennis ball rolled off a coffee table.

Imports assets/models/living_room.glb, calibrates it into the physics world
frame, puts two tennis balls in it -- one on the round coffee table and one on
the floor beside it -- runs simulate_table_drop_collision.py and replays the
result as keyframes.

Run through Blender, not python:

    blender -b --python render_table_drop_collision.py -- --out-dir ...
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

ROOM_GLB = MODELS_DIR / "living_room.glb"
# tennis_ball-3.glb rather than -2: it carries the seam as its own geometry and
# a normal map for the felt, so the seam survives a recolour instead of living
# in a base-colour texture that any hue shift has to be careful of.
TENNIS_GLB = MODELS_DIR / "tennis_ball-3.glb"
PROPS_COLLIDER_OBJ = COLLISION_DIR / "living_room_table_dressing.obj"

OUTPUT_STEM = "table_drop_collision"
BLEND_NAME = f"{OUTPUT_STEM}.blend"
PHYSICS_TEMP_NAME = "physics_transforms.json"
GROUND_TRUTH_NAME = "ground_truth_transforms.json"
SCENARIO_METADATA_NAME = "scenario_metadata.json"


# --- Room model calibration ---------------------------------------------------
#
# assets/models/living_room.glb is a small, tidy model -- 18 meshes -- and unlike
# the other flat in this project its objects at least carry meaningful material
# names, in pinyin: `zuozi` is the table, `diban` the floor, `shafa` the sofa,
# `qian` the wall. The mesh names themselves are still Object_NN, so every
# constant below was measured by raycasting the imported model rather than read
# off a name, and VERIFY_OBJECTS checks each one at import time.
#
# It is not in metres. The scale is fixed on the round coffee table, which is
# what the scene is about and which comes in standard sizes: 900 mm across is
# the common one, and the model's table is 1.2435 units at its widest.
#
# Four independent checks agree, all of them on the sofa, which is the other
# thing in the room whose real dimensions are not a matter of taste:
#   - the seat cushions land at 0.421 m, which is a sofa seat;
#   - the top of the back lands at 0.804 m;
#   - the sofa is 2.209 m wide, a three-seater;
#   - and 0.982 m deep.
# The room comes out 3.49 m floor to ceiling, which is tall -- but the model is a
# double-height space with a 3.15 m glazed wall, and the render shows exactly
# that, so it is the model and not the scale.
ROOM_TABLE_DIAMETER_UNITS = 1.24350
TABLE_DIAMETER_M = 0.900
ROOM_SCALE = TABLE_DIAMETER_M / ROOM_TABLE_DIAMETER_UNITS

# The world frame is the natural one for a scene that happens on two levels:
#   z = 0        is the floor,
#   (0, 0)       is the centre of the round coffee table,
#   +x           runs east,
#   +y           runs north, toward the sofa.
# The simulation works in the same frame, so its output needs no transform at
# all before being keyframed.
ROOM_ORIGIN = (1.34740, -2.80409, 0.41990)

# The table top, in world metres. The circle was fitted by bisecting the rim
# along sixteen headings -- they agree to 0.15 per cent, so it really is round --
# and 335 probe points spread over the surface all came back at one z to five
# decimals, so it really is flat.
TABLE_TOP_Z = 0.48245
TABLE_RADIUS = 0.44840

# Object names this scene depends on, with the world-metre bounding box each is
# expected to occupy. The mesh names carry no meaning, so the box is the real
# identity check: if the model is ever replaced or re-exported these will not
# line up and the render says so instead of quietly putting tennis balls in
# mid-air.
VERIFY_OBJECTS = {
    "Object_34": ("round coffee table", (-0.450, 0.450), (-0.450, 0.450), (-0.000, 0.482)),
    "Object_37": ("floor", (-2.487, 1.111), (-2.467, 2.029), (0.000, 0.000)),
    "Object_36": ("room shell (walls and ceiling)", (-2.487, 1.454), (-2.458, 2.029), (0.000, 3.492)),
    "Object_6": ("glazed west wall", (-1.966, -1.879), (-2.468, 2.029), (0.000, 3.121)),
    "Object_20": ("sofa outer shell", (-1.245, 0.964), (0.806, 1.788), (0.113, 0.611)),
    "Object_24": ("sofa seat cushion (east)", (-0.176, 0.769), (0.777, 1.662), (0.204, 0.421)),
    "Object_4": ("cup", (-0.307, -0.195), (0.137, 0.249), (0.483, 0.691)),
    "Object_28": ("magazine", (-0.110, 0.253), (-0.174, 0.223), (0.483, 0.515)),
    "Object_8": ("reading glasses (frame)", (-0.185, 0.003), (-0.105, 0.024), (0.457, 0.565)),
    "Object_10": ("reading glasses (lenses)", (-0.167, -0.024), (-0.097, 0.002), (0.476, 0.560)),
}
VERIFY_TOLERANCE = 0.010

# Everything already standing on the table. It is exported as its own PyBullet
# body rather than merged into the table, so that touching it is reported as its
# own event: the hero take rolls 44 mm clear of the magazine's east edge, and a
# ball that reaches it has gone somewhere the shot does not intend.
#
# The dressing is *kept*, and it is the reason the launch lane is where it is.
# Hiding it would have handed the ball the whole 0.9 m of table to aim across and
# it was tempting, but it is the only thing in the shot that gives the top a
# scale, and it costs nothing: the lane south of the magazine is 0.657 m long,
# which is more than the shot needs, and it runs along the near edge of the table
# where the ball is closest to the camera anyway.
PROP_COLLIDER_OBJECTS = ("Object_4", "Object_28", "Object_8", "Object_10")

# The floor. Its material ships at roughness 0.154, which is a wet look on what
# is meant to be an oiled board; taken to 0.34 it keeps a sheen and a soft
# reflection of the balls without turning the room upside down underneath them.
FLOOR_OBJECT = "Object_37"
TABLE_OBJECT = "Object_34"

# The glazed west wall -- except there is no glass in it. Object_6 is the mullion
# frame and the panes are open holes, which a ray fired from the floor toward the
# sun confirms: it either hits a bar or leaves the room. So the sun prints the
# whole window grid across the floor, the sofa and the table top, and at this
# elevation the bars fall straight across the lane the ball is rolled down. A
# 67 mm ball crossing a hard shadow edge every few frames reads as the ball
# changing colour rather than as the light, which is the one thing this shot
# cannot afford. Taking the frame out of the sun's path keeps the daylight, the
# raking highlight and every contact shadow, and loses only the pattern.
WINDOW_FRAME_OBJECT = "Object_6"

# The two balls, both cut from assets/models/tennis_ball-2.glb, and the point is
# that they are the *same ball*: ITF Type 2, 67 mm and 57 g each, so the mass
# ratio in the collision is exactly 1 and the closed form has nothing in it to
# tune. What tells them apart on screen is only how old they are -- the one on
# the floor is dulled and greener, which is what a tennis ball actually does
# after a season -- so the difference the viewer can see is deliberately not a
# difference the physics can see.
BALL_DIAMETER = 0.067

# Framing: chosen from a survey; see cam_survey.py. Every candidate stands south
# of the table, which is the only side there is -- the sofa's front face is
# 0.356 m off the table's north rim, so there is no room for a camera behind it.
#
# The shot is laid out to run *across* that camera rather than at it. The ball
# rolls west, leaves by the west lip and everything after it happens to the west,
# which is also the side of the room with somewhere to go: 2.0 m of clear floor
# before the glazed wall, against 0.21 m past the table's rim to the east, where
# the floor plane simply stops. A first pass had the ball rolling south and
# falling toward the lens; it was 67 mm of ball coming straight at a camera two
# metres away, and neither the fall nor the impact read.
CAMERA_LOCATION = (-0.34, -2.05, 0.92)
CAMERA_TARGET = (-0.30, -0.30, 0.20)
CAMERA_LENS_MM = 40.0


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("preview", "animation", "frames", "colliders"),
        default="animation",
        help="'colliders' only rebuilds assets/collision/*.obj and exits, which "
        "is all simulate_table_drop_collision.py needs to run standalone.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resolution", nargs=2, type=int, default=(1280, 720))
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-sec", type=float, default=2.8)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--preview-frame", type=int, default=21)
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
    parser.add_argument("--camera-location", nargs=3, type=float, default=None)
    parser.add_argument("--camera-target", nargs=3, type=float, default=None)
    parser.add_argument("--camera-lens", type=float, default=None)
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
                # A real 24 fps camera has its shutter open about half a frame,
                # and this scene needs that more than most: the ball hits the
                # floor at 3.25 m/s, which is 135 mm between frames against its
                # own 67 mm diameter. Without blur the fall is four disconnected
                # copies of a ball rather than a ball falling.
                "motion_blur": True,
                "motion_blur_shutter": 0.5,
            },
            # Hue and saturation rotate the model's own felt texture rather than
            # replacing the base colour, which would throw away the felt and the
            # seam along with the yellow. The seam is doing real work: a plain
            # sphere is rotationally invariant on screen, and the seam is the
            # only thing that shows these balls are rolling rather than sliding.
            #
            # Deliberately no diameter or mass here. Both are physics
            # parameters, they live in the physics block only, and the rendered
            # balls are sized from what the simulation reports back. Carrying
            # them in two places is how a case ends up simulating a 67 mm ball
            # and rendering an 80 mm one, which no amount of looking at the video
            # would reveal.
            "balls": {
                # The ball off the table: new, optic yellow, as authored.
                "a": {"hue": 0.50, "saturation": 1.10, "value": 1.05},
                # The one on the floor: a season old. Greener, duller, darker.
                "b": {"hue": 0.54, "saturation": 0.72, "value": 0.72},
            },
            "camera": {
                "location": list(CAMERA_LOCATION),
                "target": list(CAMERA_TARGET),
                "lens_mm": CAMERA_LENS_MM,
            },
            # The model ships no lights at all, so the room's light had to be
            # built. It does have one enormous asset to build it from: a 3.15 m
            # glazed wall running the whole west side, which is where all of this
            # comes from and why the balance was found by rendering rather than
            # by reasoning.
            "lighting": {
                "world_strength": 0.42,
                "sun_energy": 1.7,
                "sun_elevation_deg": 34.0,
                "sun_azimuth_deg": -108.0,
                "window_bounce_power": 45.0,
                "front_fill_power": 9.0,
                # Let the window frame print its grid on the room. Off: see
                # WINDOW_FRAME_OBJECT for why the pattern costs more than the
                # interest it adds.
                "window_frame_shadow": False,
            },
            # Tuned in simulate_table_drop_collision.py; see its module comment
            # for what each one is for and its argument help for why it is where
            # it is. Mass and radius are the ITF tennis ball and are not sampled:
            # the balls being identical is the reason the collision's closed form
            # has no free parameter in it.
            "physics": {
                "launch_x": 0.315,
                "launch_y": -0.290,
                "launch_speed": 1.28,
                "launch_heading_deg": 180.0,
                "ball_a_radius": BALL_DIAMETER / 2.0,
                "ball_b_radius": BALL_DIAMETER / 2.0,
                "ball_a_mass": 0.057,
                "ball_b_mass": 0.057,
                "ball_b_x": -0.722,
                # Dead in line with the lane the ball is rolled down, so the
                # exchange has no sideways component at all: the two balls leave
                # 180.0 deg apart and both end the shot on y = -0.290, the line
                # they started on. See the simulation's help for this argument
                # for what that costs and why floor_rolling_friction pays it.
                "ball_b_y": -0.290,
                "ball_a_restitution": 0.86,
                "ball_b_restitution": 0.86,
                "ball_a_friction": 0.62,
                "ball_b_friction": 0.62,
                "table_friction": 0.55,
                "table_restitution": 0.60,
                "table_rolling_friction": 0.0037,
                "table_spinning_friction": 0.004,
                "floor_friction": 0.58,
                "floor_restitution": 0.87,
                "floor_rolling_friction": 0.0110,
                "floor_spinning_friction": 0.004,
                "gravity_z": -9.8,
                # Two-slot presence list, in fixed order (rolling_ball / A,
                # target_ball / B). A PCVE DELETE edit writes 0 at the
                # ball's slot; only slot 1 (target) is actually removable --
                # the whole scene is centred on ball A rolling off the
                # table, so it is always kept.
                "active": [1, 1],
            },
        }
    if args.scenario_overrides_json is not None:
        scenario = deep_merge(
            scenario, json.loads(args.scenario_overrides_json.read_text(encoding="utf-8")),
        )
    for key, value in (("location", args.camera_location),
                       ("target", args.camera_target)):
        if value is not None:
            scenario["camera"][key] = [float(v) for v in value]
    if args.camera_lens is not None:
        scenario["camera"]["lens_mm"] = float(args.camera_lens)
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

    Rather than baking a transform into each of the model's meshes, all of its
    root objects are parented to a single empty carrying the unit conversion and
    the shift onto the table's centre. Parenting with an identity parent-inverse
    means each mesh's own transform composes with the empty's, so the model stays
    internally untouched, the mapping stays inspectable in the saved .blend, and
    -- the point -- the collider export and the render read the same matrices.
    """
    if not ROOM_GLB.exists():
        raise FileNotFoundError(f"Room model not found: {ROOM_GLB}")
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(ROOM_GLB))
    imported = [o for o in bpy.context.scene.objects if o not in before]

    room_root = bpy.data.objects.new("room_root", None)
    bpy.context.scene.collection.objects.link(room_root)
    room_root.location = tuple(-ROOM_SCALE * v for v in ROOM_ORIGIN)
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

    The mesh names are Object_NN and carry no meaning, so a name is no evidence
    that the right mesh has been found. Each one's world bounding box is compared
    against the box it was measured in; a mismatch means the model has been
    replaced or re-exported and every constant in this file needs re-deriving.
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

    table = bpy.data.objects.get(TABLE_OBJECT)
    if table is not None:
        mn, mx = world_bbox(table)
        if abs(mx.z - TABLE_TOP_Z) > 0.002:
            print(f"[WARN] The coffee table's top is at z = {mx.z:+.4f}, not "
                  f"{TABLE_TOP_Z:+.4f}. ROOM_SCALE or ROOM_ORIGIN is wrong.")
        print(f"[INFO] Coffee table: {mx.x - mn.x:.3f} m across, top at "
              f"z = {mx.z:+.4f}; scale {ROOM_SCALE:.7f} "
              f"({TABLE_DIAMETER_M:.3f} m / {ROOM_TABLE_DIAMETER_UNITS} units).")
    floor = bpy.data.objects.get(FLOOR_OBJECT)
    if floor is not None:
        mn, mx = world_bbox(floor)
        if abs(mx.z) > 0.002:
            print(f"[WARN] The floor is at z = {mx.z:+.4f}, not 0.")


def temper_surfaces() -> None:
    """Take the wet look off the floor.

    The floor ships at roughness 0.154, which on an oiled board reads as a
    puddle and hangs a mirror image of the sofa under everything standing on it.
    Taken to 0.34 it keeps a sheen and a soft reflection of the balls without
    turning the room upside down. The base colour is dealt with separately, in
    ``unbake_floor``; the material is copied first, because the datablocks are
    shared and rewriting one in place would take the rest of the room with it.

    The table top is deliberately *not* adjusted, and the reason is worth
    recording because it is a trap that fails silently. Its material drives
    Roughness and Metallic from a packed texture through a Math node, and
    assigning to ``default_value`` on a socket that already has a link does
    nothing at all -- no error, no warning, and a render that looks exactly like
    the one before it. Anything wanting to change it has to go through the link,
    and the top does not need changing: the texture already gives it the satin
    of finished ash. ``report_driven_inputs`` says which sockets are in that
    state, so the next person does not have to find out the hard way.
    """
    obj = bpy.data.objects.get(FLOOR_OBJECT)
    if obj is None or not obj.data.materials or obj.data.materials[0] is None:
        print("[WARN] The floor carries no material to adjust.")
        return
    material = obj.data.materials[0].copy()
    material.name = "floor_boards"
    obj.data.materials[0] = material
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is None:
        print("[WARN] The floor has no Principled BSDF to adjust.")
        return
    for socket, value in (("Roughness", 0.34), ("Specular", 0.42)):
        inp = bsdf.inputs.get(socket)
        if inp is None:
            continue
        if inp.is_linked:
            print(f"[WARN] The floor's {socket} is driven by a "
                  f"{inp.links[0].from_node.type} node, so setting it here would "
                  "have done nothing; left as authored.")
            continue
        inp.default_value = value
        print(f"[INFO] floor: {socket} -> {value}")
    unbake_floor(material)
    report_driven_inputs()


# The floor's albedo, measured from its own texture rather than chosen. Baked
# light only ever subtracts, so over the atlas region the floor actually uses
# the unshaded boards are the bright end of the distribution: the 95th centile
# and above average to this, in linear terms. The shading below it runs to 40
# per cent darker and 10 per cent of the region is flat black.
FLOOR_ALBEDO = (0.4645, 0.4579, 0.4496, 1.0)


def unbake_floor(material) -> None:
    """Take the painted-on daylight off the floor.

    What looks like a board texture is a lightmap. The model is a Sketchfab
    capture and its floor atlas has a whole afternoon baked into it: hard
    diagonal bands from a blind, a soft blot where the sofa stands, a ring under
    a rug that is not in the scene, and a black rectangle. None of it agrees
    with this scene's light, and it cannot -- it was painted for a sun that is
    not the one being rendered, so it does not move, does not soften, and is not
    occluded by anything the render puts on top of it. It was read as floor
    boards for a long time because it is grey-on-grey and does look like grain
    until the sun is turned off and the bands are still there.

    It is worth being clear about what is lost. Under the shading the atlas is
    flat: over the floor's own region the 50th to 99th centile spans six per
    cent, so there is no grain to keep. Replacing the texture with the one
    colour it is actually made of loses nothing and lets Cycles light the floor.
    """
    tree = material.node_tree
    base = tree.nodes.get("Principled BSDF").inputs["Base Color"]
    if not base.is_linked:
        print("[INFO] floor: base colour is already flat; nothing to unbake.")
        return
    for link in list(base.links):
        tree.links.remove(link)
    base.default_value = FLOOR_ALBEDO
    print(f"[INFO] floor: baked lighting removed from the base colour, which is "
          f"now the flat albedo {FLOOR_ALBEDO[:3]}.")


def report_driven_inputs() -> None:
    """List the shader inputs this scene cannot set by assignment."""
    for name, label in ((TABLE_OBJECT, "coffee table"),):
        obj = bpy.data.objects.get(name)
        if obj is None or not obj.data.materials or obj.data.materials[0] is None:
            continue
        bsdf = obj.data.materials[0].node_tree.nodes.get("Principled BSDF")
        if bsdf is None:
            continue
        driven = [s.name for s in bsdf.inputs
                  if s.is_linked and s.name in ("Base Color", "Roughness",
                                                "Metallic", "Specular")]
        if driven:
            print(f"[INFO] {label}: {', '.join(driven)} are texture-driven and "
                  "cannot be changed by assignment; left as authored.")


# --- Balls --------------------------------------------------------------------

def import_ball(name: str, diameter: float, hue: float, saturation: float,
                value: float):
    """Import the tennis ball model, size it, centre it, and age it.

    The colour is changed by rotating the hue of the model's own felt texture
    rather than by overwriting the base colour, which would throw away the felt
    and the seam along with the yellow -- and the seam is the only thing on a
    sphere that shows it is rolling. Each ball gets its own copy of the material;
    otherwise both would share one datablock and ageing the second would age the
    first.
    """
    if not TENNIS_GLB.exists():
        raise FileNotFoundError(f"Tennis ball model not found: {TENNIS_GLB}")

    scene = bpy.context.scene
    before = set(scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(TENNIS_GLB))
    imported = [o for o in scene.objects if o not in before]
    meshes = [o for o in imported if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"{TENNIS_GLB.name} imported no mesh.")

    # The model arrives as more than one mesh -- in this one the green body and
    # the white seam laid over it are separate pieces -- and every piece is part
    # of the ball. Keeping only the biggest mesh, as this used to, threw the
    # seam away and left a plain sphere; on the previous model the two pieces
    # were the same size and it silently kept the wrong one. Join them all.
    for obj in meshes:
        matrix = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = matrix
    for obj in imported:
        if obj.type != "MESH":
            bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    ball = meshes[0]
    bpy.context.view_layer.objects.active = ball
    if len(meshes) > 1:
        bpy.ops.object.join()
    ball.name = name

    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = ball
    ball.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # Scale to the physical diameter, then put the origin at the sphere's centre.
    # The simulation hands over centre positions and spin quaternions, so the
    # object's origin has to *be* the centre or the ball visibly orbits its own
    # offset as it rolls.
    mn, mx = world_bbox(ball)
    span = max(mx.z - mn.z, 1e-9)
    ball.scale = (float(diameter) / span,) * 3
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    ball.location = (0.0, 0.0, 0.0)
    ball.select_set(False)

    if not ball.data.materials:
        raise RuntimeError(f"{TENNIS_GLB.name} carries no material to recolour.")

    # Every slot on the joined ball gets the same treatment. Ageing only the
    # body would leave the fuzz over it at the colour it was authored, which
    # reads as a new ball wearing an old one's shadow.
    for index, authored in enumerate(ball.data.materials):
        if authored is None:
            continue
        material = authored.copy()
        material.name = f"{name}_felt_{index}"
        ball.data.materials[index] = material
        nodes, links = material.node_tree.nodes, material.node_tree.links
        bsdf = nodes.get("Principled BSDF")
        if bsdf is None:
            print(f"[WARN] {name}: slot {index} has no Principled BSDF to age.")
            continue
        base = bsdf.inputs.get("Base Color")
        if base is None:
            continue
        hue_sat = nodes.new("ShaderNodeHueSaturation")
        hue_sat.inputs["Hue"].default_value = float(hue)
        hue_sat.inputs["Saturation"].default_value = float(saturation)
        hue_sat.inputs["Value"].default_value = float(value)
        if base.is_linked:
            links.new(base.links[0].from_socket, hue_sat.inputs["Color"])
        else:
            # A layer that carries a flat glTF baseColorFactor instead of a
            # texture still has to age by the same amount as the one that
            # doesn't, so feed the flat colour through the same node.
            hue_sat.inputs["Color"].default_value = base.default_value
        links.new(hue_sat.outputs["Color"], base)
        # Felt. The export leaves it at the glTF default, which puts a hard
        # highlight on what should be the least shiny thing in the room.
        set_input(bsdf, "Roughness", 0.92)
        set_input(bsdf, "Specular", 0.18)

    for poly in ball.data.polygons:
        poly.use_smooth = True
    ball.rotation_mode = "QUATERNION"
    print(f"[INFO] Ball {name}: {span:.4f} model units -> {diameter:.3f} m, "
          f"hue {hue}, saturation {saturation}, value {value}")
    return ball


# --- Colliders ----------------------------------------------------------------

def export_collider(path: Path, object_names) -> Path:
    """Write what stands on the table as a metre-scale OBJ for PyBullet.

    Handing over the props' own triangles through the same transform the render
    uses is the only way the two can be guaranteed to agree about where the
    magazine's edge is, and the launch lane clears it by 44 mm.

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
        python, str(Path(__file__).with_name("simulate_table_drop_collision.py")),
        "--out", str(out),
        "--props-collider", str(PROPS_COLLIDER_OBJ),
        "--fps", str(int(scenario["render"]["fps"])),
        "--duration-sec", str(float(scenario["render"]["duration_sec"])),
    ]
    for flag, key in (
        ("--ball-a-radius", "ball_a_radius"),
        ("--ball-b-radius", "ball_b_radius"),
        ("--ball-a-mass", "ball_a_mass"),
        ("--ball-b-mass", "ball_b_mass"),
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
        ("--floor-friction", "floor_friction"),
        ("--floor-restitution", "floor_restitution"),
        ("--floor-rolling-friction", "floor_rolling_friction"),
        ("--floor-spinning-friction", "floor_spinning_friction"),
        ("--gravity-z", "gravity_z"),
    ):
        command += [flag, str(float(ph[key]))]

    # Ball A always exists in this scene; only ball B is optional.
    active = ph.get("active", [1, 1])
    command += ["--ball-b-active", str(int(active[1]))]

    subprocess.run(command, check=True)
    data = json.loads(out.read_text(encoding="utf-8"))
    out.unlink(missing_ok=True)

    q, launch = data["quality"], data["launch"]
    imp = data["collision"]["predicted"]
    if not q["left_table"]:
        print("[WARN] The ball never reached the edge of the table.")
    if not q["hit_ball_b"]:
        print("[WARN] The falling ball missed the target. The landing point is "
              "settled at the lip and the target is 67 mm wide, so a few per "
              "cent on the push is enough to do this; move ball_b_y with it.")
    if q["spinning_on_the_spot"]:
        print(f"[WARN] A ball is still turning where it stopped: "
              f"{q['residual_spin']} rad/s. Check the spinning friction.")
    if q["hit_table_dressing"]:
        print(f"[WARN] A ball reached the cup, the magazine or the glasses at "
              f"frame {q['prop_contact_frame']}; the lane is meant to pass clear.")
    if q["ran_off_floor"] is not None:
        print(f"[WARN] {q['ran_off_floor']['ball']} left the modelled floor at "
              f"frame {q['ran_off_floor']['frame']} to the "
              f"{q['ran_off_floor']['edge']}.")
    if q["ball_a_settled_frame"] is None or q["ball_b_settled_frame"] is None:
        print("[WARN] A ball was still moving on the last frame; it never "
              "settles on camera.")

    # The scene's first standing check: the parabola against the ball. Between
    # the lip and the target nothing touches it, so any disagreement here is the
    # solver disagreeing with elementary mechanics and is the single most likely
    # way for this scene to be quietly wrong.
    error = launch["flight_prediction_error"]
    if error is not None and error > 0.005:
        print(f"[WARN] The ball ended up {error * 1000:.0f} mm from where the "
              "projectile formula puts it. Nothing touches it in flight, so "
              "either the lip state or the solver is wrong.")
    flight = launch["predicted_flight"]
    if flight:
        print(f"[INFO] Left the lip at {launch['lip']['speed_h']:.3f} m/s after "
              f"{launch['lip']['roll_distance']:.3f} m of table "
              f"({launch['roll_deceleration']:.3f} m/s^2 of rolling resistance); "
              f"fell {flight['drop']:.3f} m in {flight['fall_time']:.4f} s and "
              f"reached {flight['reach']:.3f} m, landing predicted at "
              f"({flight['landing'][0]:.3f}, {flight['landing'][1]:.3f}) and met "
              f"{error * 1000:.1f} mm.")

    # The second: the line of centres against the solver. The heading is the
    # tight one -- it has no free parameter in it at all -- and the speed carries
    # the floor correction, so they are checked apart.
    if imp and imp.get("measured_b_heading_deg") is not None:
        heading_error = abs((imp["measured_b_heading_deg"] - imp["b_heading_deg"]
                             + 180.0) % 360.0 - 180.0)
        if heading_error > 5.0:
            print(f"[WARN] The struck ball left on {imp['measured_b_heading_deg']:.1f} "
                  f"deg but the line of centres says {imp['b_heading_deg']:.1f} deg. "
                  "That direction is a prediction with nothing to tune in it, so a "
                  "disagreement is real.")
        measured, predicted = (imp["measured_b_horizontal_speed"],
                               imp["b_horizontal_speed"])
        if imp.get("floor_correction_out_of_range"):
            print(f"[INFO] The contact normal is {imp['normal_below_horizontal_deg']:.1f} "
                  "deg below horizontal, which is past where the floor correction "
                  "holds; the predicted speed for the struck ball is not "
                  "meaningful in this case and only the heading is checked.")
        elif abs(measured - predicted) > 0.10 * max(predicted, 0.05) + 0.02:
            print(f"[WARN] The struck ball left at {measured:.3f} m/s, but the "
                  f"line of centres with the floor's share taken out predicts "
                  f"{predicted:.3f} (free-space {imp['b_horizontal_speed_free']:.3f}). "
                  "One of them is wrong.")
        else:
            print(f"[INFO] Contact normal {imp['normal_below_horizontal_deg']:.1f} deg "
                  f"below horizontal; the struck ball left at {measured:.3f} m/s on "
                  f"{imp['measured_b_heading_deg']:.1f} deg against a prediction of "
                  f"{predicted:.3f} on {imp['b_heading_deg']:.1f} deg "
                  f"(free-space {imp['b_horizontal_speed_free']:.3f}; the floor takes "
                  f"{imp['floor_share_of_horizontal'] * 100:.0f} per cent). The two "
                  f"balls left {imp['separation_deg']:.1f} deg apart.")
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

    lighting = scenario.get("lighting", {})

    # World: a sky, not a room. The whole west wall of this flat is glass 3.15 m
    # tall, so what the camera sees through it is outdoors, and an interior HDRI
    # behind it would read as a second room hanging in the air. It is also doing
    # more work here than in most scenes: with no lights in the model and one
    # enormous window, the sky *is* the room's ambient.
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    for node in list(nodes):
        nodes.remove(node)
    output = nodes.new("ShaderNodeOutputWorld")
    bg = nodes.new("ShaderNodeBackground")
    hdri = POLYHAVEN_DIR / "outdoor" / "kloofendal_48d_partly_cloudy_puresky_2k.hdr"
    if hdri.exists():
        tex_coord = nodes.new("ShaderNodeTexCoord")
        mapping = nodes.new("ShaderNodeMapping")
        mapping.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(160.0))
        env = nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(str(hdri), check_existing=True)
        links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], env.inputs["Vector"])
        links.new(env.outputs["Color"], bg.inputs["Color"])
        bg.inputs["Strength"].default_value = float(lighting.get("world_strength", 0.42))
    else:
        print(f"[WARN] HDRI not found at {hdri}; falling back to a flat sky colour.")
        bg.inputs["Color"].default_value = (0.60, 0.68, 0.82, 1.0)
        bg.inputs["Strength"].default_value = float(lighting.get("world_strength", 0.42)) * 1.8
    links.new(bg.outputs["Background"], output.inputs["Surface"])

    import_room()
    temper_surfaces()

    frame = bpy.data.objects.get(WINDOW_FRAME_OBJECT)
    if frame is None:
        print(f"[WARN] {WINDOW_FRAME_OBJECT} is not in the model, so the window "
              "frame's shadow could not be turned off; expect a grid on the floor.")
    elif not bool(lighting.get("window_frame_shadow", False)):
        frame.visible_shadow = False
        print("[INFO] window frame: casts no shadow, so the room keeps the "
              "daylight without the grid.")

    # Sized from what the simulation actually used, not from the scenario, so
    # the two cannot disagree.
    cfg = scenario["balls"]
    ball_a = import_ball("tennis_new",
                         2.0 * float(physics["objects"]["ball_a"]["radius"]),
                         float(cfg["a"]["hue"]), float(cfg["a"]["saturation"]),
                         float(cfg["a"].get("value", 1.0)))
    ball_b = import_ball("tennis_old",
                         2.0 * float(physics["objects"]["ball_b"]["radius"]),
                         float(cfg["b"]["hue"]), float(cfg["b"]["saturation"]),
                         float(cfg["b"].get("value", 1.0)))
    apply_keyframes(ball_a, physics["frames"], "ball_a")
    # DELETE edit: ball_b was removed from the sim (present=false, no per-frame
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

    # Late-afternoon daylight through the glazed west wall, low and raking. It
    # comes from the west because that is the only wall in this flat that light
    # can come through, and it is what puts a highlight on the felt and a hard
    # contact shadow under each ball -- the sky alone is a dome and leaves both
    # balls sitting in flat light with nothing to say they are touching anything.
    bpy.ops.object.light_add(type="SUN", location=(-3.0, 0.4, 1.8))
    sun = bpy.context.object
    sun.data.energy = float(lighting.get("sun_energy", 1.7))
    sun.data.angle = math.radians(2.4)
    sun.data.color = (1.0, 0.955, 0.90)
    sun.rotation_euler = (
        math.radians(90.0 - float(lighting.get("sun_elevation_deg", 34.0))),
        0.0,
        math.radians(float(lighting.get("sun_azimuth_deg", -108.0))),
    )

    # The window as a light in its own right, standing just inside the glass:
    # a 3.1 m tall glazed wall is a huge soft source and the sun lamp alone only
    # models the direct beam through it. It stands well north of the action and
    # high, rather than beside it -- the struck ball comes to rest 0.75 m from the
    # glass, and a light any closer than this blows it out on the last frame,
    # which is the frame the two balls' separation has to be read from.
    add_area_light("window_bounce", (-1.80, 0.15, 1.45),
                   power=float(lighting.get("window_bounce_power", 45.0)), size=3.0,
                   target=(-0.35, -0.32, 0.20), color=(0.94, 0.965, 1.0))
    # A small fill on the camera side at about knee height. Without it the balls'
    # camera-facing halves fall into the dark boards, because everything else in
    # the room lights them from behind and above.
    add_area_light("front_fill", (-0.20, -1.60, 0.55),
                   power=float(lighting.get("front_fill_power", 9.0)), size=1.2,
                   target=(-0.45, -0.34, 0.06), color=(1.0, 0.98, 0.95))

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
                       "mass": physics["objects"]["ball_a"]["mass"],
                       "inertia": physics["objects"]["ball_a"]["inertia"]},
            "ball_b": {"present": bool(physics["objects"]["ball_b"].get("present", True)),
                       "object_name": ball_b.name,
                       "radius": physics["objects"]["ball_b"]["radius"],
                       "mass": physics["objects"]["ball_b"]["mass"],
                       "inertia": physics["objects"]["ball_b"]["inertia"]},
            "table": {"radius": TABLE_RADIUS, "top_z": TABLE_TOP_Z,
                      "diameter_m": TABLE_DIAMETER_M},
            "floor": {"z": 0.0},
        },
        "room": {
            "model": ROOM_GLB.name,
            "scale": ROOM_SCALE,
            "origin_model_units": list(ROOM_ORIGIN),
            "table_diameter_model_units": ROOM_TABLE_DIAMETER_UNITS,
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
        pairs = [("ball_a", ball_a, True)]
        if ball_b_present:
            pairs.append(("ball_b", ball_b, True))
        for key, obj, has_data in pairs:
            entry[key] = {
                "matrix_world": [[float(v) for v in row] for row in obj.matrix_world],
                "linear_velocity": pf[key]["linear_velocity"],
                "angular_velocity": pf[key]["angular_velocity"],
                "speed": pf[key]["speed"],
                "spin_z": pf[key]["spin_z"],
                "height_above_floor": pf[key]["height_above_floor"],
                "on_table": pf[key]["on_table"],
                "airborne": pf[key]["airborne"],
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
