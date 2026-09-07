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
from mathutils import Euler, Quaternion, Vector

WORKSPACE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = WORKSPACE_DIR / "assets" / "models"

ROOM_GLB = MODELS_DIR / "vintage_modern_living_room_with_arcades.glb"
TABLE_GLB = MODELS_DIR / "air_hockey_arcade.glb"

OUTPUT_STEM = "air_hockey_chain"
PHYSICS_TEMP = "physics_transforms.json"
GROUND_TRUTH_NAME = "ground_truth_transforms.json"
SCENARIO_METADATA_NAME = "scenario_metadata.json"

# --- The table, in its own model units ---------------------------------------
# air_hockey_arcade.glb is authored at roughly 1 unit = 2.6 cm with its feet on
# z=0. Everything below was measured off the mesh: a raycast height map over the
# whole table found the playing surface as one flat plane and the only two
# things standing on it, and the mallets' extents came from the vertices.
MODEL_SURFACE_Z = 30.50
MODEL_FIELD = (3.00, 48.50, 3.00, 96.50)      # x0, x1, y0, y1
MODEL_TABLE_CENTRE = (25.75, 49.75)
MODEL_MALLET_A = (31.255, 19.285)             # stands at the foot of an arch post
MODEL_MALLET_B = (19.505, 89.472)             # in the clear -- this one is cloned
MODEL_MALLET_DIAMETER = 4.845
MODEL_MALLET_BASE_Z = 30.595
MODEL_MALLET_TOP_Z = 33.160
# How far to pull the model's back-to-back surface sheets apart, in model
# units: ~0.5 mm once the table is scaled down, far below anything the camera
# or the measured pick regions can see, but far above float precision.
SHEET_SPLIT = 0.02

# --- Placing the table in the room -------------------------------------------
# The room is authored in metres with its floor at z=0.1489. The table stands
# out in the open middle of the room rather than pushed against a wall: on the
# clear floor between the TV on the +Y wall (panel at y 3.40..3.49, its console
# reaching back to y 2.65) and the coffee table in front of the sofa (y
# -1.71..-0.70). That gap is 3.35 m deep, and between the arcade cabinets
# flanking it -- the left pair reaches x -1.74, the right one starts at x 1.95
# -- it is 3.7 m wide. Centring the 1.43 x 2.55 m cabinet in it leaves ~0.4 m
# of floor off each end, with its length already running the way the space does.
FLOOR_Z = 0.1489
TABLE_HEIGHT = 0.78                            # playing surface above the floor
TABLE_SCALE = TABLE_HEIGHT / MODEL_SURFACE_Z
TABLE_CENTRE_XY = (0.30, 0.98)

TABLE_ORIGIN = (
    TABLE_CENTRE_XY[0] - MODEL_TABLE_CENTRE[0] * TABLE_SCALE,
    TABLE_CENTRE_XY[1] - MODEL_TABLE_CENTRE[1] * TABLE_SCALE,
    FLOOR_Z,
)


def tw(mx: float, my: float, mz: float) -> tuple[float, float, float]:
    """Table-model coordinates -> world."""
    return (
        TABLE_ORIGIN[0] + mx * TABLE_SCALE,
        TABLE_ORIGIN[1] + my * TABLE_SCALE,
        TABLE_ORIGIN[2] + mz * TABLE_SCALE,
    )


SURFACE_Z = TABLE_ORIGIN[2] + MODEL_SURFACE_Z * TABLE_SCALE
MALLET_RADIUS = MODEL_MALLET_DIAMETER * TABLE_SCALE / 2.0
MALLET_HEIGHT = (MODEL_MALLET_TOP_Z - MODEL_MALLET_BASE_Z) * TABLE_SCALE

# --- Geometry (must match simulate_air_hockey_chain.py) ----------------------
TABLE_LEN = (MODEL_FIELD[3] - MODEL_FIELD[2]) * TABLE_SCALE
TABLE_WIDTH = (MODEL_FIELD[1] - MODEL_FIELD[0]) * TABLE_SCALE
# The three mallets sit on the quarter, half and three-quarter points of the
# table's length -- as far apart as a three-mallet relay can be spread while
# leaving the last one room to run out. Wide gaps are what make each handoff
# legible: a clear stretch of coasting between hits rather than three impacts
# on top of each other. The chain steps down -x, from the end away from the
# camera back towards it, so the struck disc always comes at the viewer and
# each handoff lands nearer the lens and larger in frame than the one before.
RELAY_DIRECTION = -1.0
START_X = 3.0 * TABLE_LEN / 4.0
SPACING = RELAY_DIRECTION * TABLE_LEN / 4.0
RELAY_Y = 0.0

# The sim frame runs x down the table's length, y across it, z up from the
# playing surface. The table's length lies along world Y, so the mapping is a
# +90 degree yaw about Z followed by a translation onto the field.
SIM_YAW_DEG = 90.0
FIELD_CENTRE_X = TABLE_ORIGIN[0] + (MODEL_FIELD[0] + MODEL_FIELD[1]) / 2.0 * TABLE_SCALE
FIELD_FAR_Y = TABLE_ORIGIN[1] + MODEL_FIELD[2] * TABLE_SCALE

# The master mallet is lifted out of the table mesh by position. The z window is
# what keeps the pick honest: mallet A stands at the foot of one of the arch
# posts, and without an upper bound the pick would swallow the post as well.
MALLET_PICK_RADIUS = MALLET_RADIUS + 0.012
MALLET_PICK_Z = (
    TABLE_ORIGIN[2] + (MODEL_MALLET_BASE_Z - 0.15) * TABLE_SCALE,
    TABLE_ORIGIN[2] + (MODEL_MALLET_TOP_Z + 0.15) * TABLE_SCALE,
)

CAMERA_LENS_MM = 55.0
CAMERA_FSTOP = 4.5
# Where along the table's length the camera looks, as a fraction of it. Short
# of the middle on purpose: the near end of the cabinet is a metre closer to
# the lens than the far end, so it subtends more, and aiming at the geometric
# centre runs it off the bottom of frame.
CAMERA_AIM_ALONG_TABLE = 0.40

# mallet_0 is the one that gets pushed and the momentum passes down the list,
# so this is the order of the relay: blue, standing at the end of the table
# away from the camera, strikes red in the middle, red strikes white at the
# near end, and white runs out to the near rail.
MALLET_COLOURS = (
    ("blue", (0.020, 0.070, 0.420)),
    ("red", (0.520, 0.030, 0.030)),
    ("white", (0.800, 0.800, 0.810)),
)

# Where the room's own lights actually are, measured off the room mesh. Lamps
# placed on these keep the shading consistent with what is in shot.
CEILING_LAMP_XY = (0.0, 0.0)          # the red pendant, its bulb at z 3.08
CEILING_LAMP_Z = 3.02
WINDOW_XY = (-2.94, 0.0)              # the window pane on the -X wall
WINDOW_Z = 2.20

# The two materials carrying the table's printed art, playfield and side
# panels. Named so they can be found again after the glTF import renames them.
PLAYFIELD_MATERIALS = ("Color_A08", "Color_B02")
PLAYFIELD_ROUGHNESS = 0.22

# Lamp powers, in watts. The balance is deliberately lamp-heavy: the pendant
# carries the shot and the other two only keep the shadows from going black,
# which is what a room lit by one bulb after dark actually looks like. Pushing
# the fill up instead (tried at roughly double these) flattens the tiled floor
# out and the cabinet stops sitting in the room.
PENDANT_POWER = 1100.0
WINDOW_POWER = 350.0
BOUNCE_POWER = 150.0
WORLD_STRENGTH = 0.22


def sw(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Sim-frame coordinates -> world coordinates."""
    yaw = math.radians(SIM_YAW_DEG)
    return (
        FIELD_CENTRE_X + x * math.cos(yaw) - y * math.sin(yaw),
        FIELD_FAR_Y + x * math.sin(yaw) + y * math.cos(yaw),
        SURFACE_Z + z,
    )


def sim_quat_to_world(quat_xyzw) -> Quaternion:
    x, y, z, w = quat_xyzw
    yaw = Euler((0.0, 0.0, math.radians(SIM_YAW_DEG)), "XYZ").to_quaternion()
    return yaw @ Quaternion((w, x, y, z))


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preview", "animation"), default="animation")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resolution", nargs=2, type=int, default=(1280, 720))
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-sec", type=float, default=3.0)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--preview-frame", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=41)
    # 0.8 m/s, not the 1.0 the relay used when it was packed into one half of
    # the table. Over the wider gaps this is what lands both handoffs on
    # theory -- each striker keeps 3-4% of its speed against the (1-e)/2 =
    # 4.9% the pair restitution predicts -- and it still walks the last mallet
    # into the near rail at 1.96 s, leaving a beat to rest there. Slower is
    # not better: at 0.75 and 0.7 the second handoff degrades to 8% and 10%.
    parser.add_argument("--push-speed", type=float, default=0.8)
    parser.add_argument("--mallet-restitution", type=float, default=0.95)
    parser.add_argument("--surface-friction", type=float, default=0.06)
    parser.add_argument("--middle-mass-scale", type=float, default=1.0)
    parser.add_argument(
        "--scenario-overrides-json",
        type=Path,
        default=None,
        help="JSON file merged recursively onto the scenario built from the "
        "flags above. This is how the PCVE suite applies a single edit: it "
        "writes {\"physics\": {...}} touching one parameter and leaves "
        "everything else, including the camera and the seed, identical to the "
        "source video.",
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
    must stay in sync with both simulate_air_hockey_chain.py's defaults and
    edit_vocab.BASELINE_PHYSICS -- an edit's "from" value is read out of the
    vocabulary, so a drift here silently mislabels every physics diff.
    Every mallet property is a three-element list in relay order (blue, red,
    white) so that an edit can name one disc without disturbing the others.
    """
    # Mirrors MALLET_MASS in simulate_air_hockey_chain.py.
    mallet_mass = 0.120
    scenario = {
        "schema_version": 2,
        "seed": int(args.seed),
        "environment": "living_room_arcade_air_hockey",
        "render": {
            "fps": int(args.fps),
            "duration_sec": float(args.duration_sec),
            "resolution": [int(args.resolution[0]), int(args.resolution[1])],
            "samples": int(args.samples),
            "device": str(args.device),
            "mode": str(args.mode),
        },
        "physics": {
            "push_speed": float(args.push_speed),
            "mallet_masses": [
                mallet_mass,
                mallet_mass * float(args.middle_mass_scale),
                mallet_mass,
            ],
            "mallet_restitutions": [float(args.mallet_restitution)] * 3,
            # The mallets' own friction tracks the surface's by default: the
            # two are multiplied into one effective coefficient, and the CLI
            # sweeps have always moved them together as a single "air cushion"
            # dial. A PCVE edit can still move one without the other.
            "mallet_frictions": [float(args.surface_friction)] * 3,
            "mallet_active": [1, 1, 1],
            "surface_friction": float(args.surface_friction),
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


# --- Asset handling ----------------------------------------------------------

def world_bbox(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    return Vector((min(xs), min(ys), min(zs))), Vector((max(xs), max(ys), max(zs)))


def make_active(obj: bpy.types.Object) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def import_glb(path: Path) -> list[bpy.types.Object]:
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    return [o for o in bpy.context.scene.objects if o not in before]


def join_meshes(objs: list[bpy.types.Object], name: str) -> bpy.types.Object:
    """Bake the glTF parent transforms into the vertices and join into one
    object, so the measured pick regions apply directly to vertex coordinates."""
    meshes = [o for o in objs if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh found for {name}")
    # Names, not references: join() deletes the objects it merged, and any
    # Python reference to one of them is dead the moment it does.
    empty_names = [o.name for o in objs if o.type != "MESH"]
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    if len(meshes) > 1:
        bpy.ops.object.join()
    joined = bpy.context.view_layer.objects.active
    joined.name = name
    for empty_name in empty_names:
        leftover = bpy.data.objects.get(empty_name)
        if leftover is not None:
            bpy.data.objects.remove(leftover, do_unlink=True)
    return joined


def polish_playfield(objs: list[bpy.types.Object]) -> int:
    """Give the table's printed surfaces the gloss a laminate actually has.

    The glTF import lands every material on roughness 0.6, which is chalk. That
    matters more than it sounds: the discs are only 7 cm tall, so a lamp hung
    at ceiling height throws a shadow a couple of centimetres long that hides
    underneath the disc itself -- no shadow the camera can see. What ties a
    disc to the surface at this angle is the surface reflecting it back, which
    a matte surface will not do. These are the two materials carrying the
    printed art, on the playfield and on the cabinet's side panels alike.
    """
    touched = 0
    for obj in objs:
        for slot in getattr(obj, "material_slots", []):
            mat = slot.material
            if not mat or mat.name.split(".")[0] not in PLAYFIELD_MATERIALS:
                continue
            if not mat.use_nodes:
                continue
            for node in mat.node_tree.nodes:
                if node.type == "BSDF_PRINCIPLED":
                    node.inputs["Roughness"].default_value = PLAYFIELD_ROUGHNESS
                    touched += 1
    return touched


def split_double_sided_sheets(objs: list[bpy.types.Object]) -> None:
    """Pull the table's two coincident surface sheets apart.

    air_hockey_arcade.glb is a zero-thickness double-sided export: essentially
    every surface is modelled as two faces in exactly the same place pointing
    opposite ways, the outward one carrying the artwork and the inward one a
    plain colour -- 9049 of the 9051 coincident face pairs in the file are
    back-to-back like this. Every material sets use_backface_culling, so the
    tool it was authored in only ever drew the outward face, but that flag is
    EEVEE-only. Cycles hits both, they are exactly coplanar, and which one wins
    is settled per BVH region, which is what painted the playing surface in
    stripes. Nudging every vertex along its own normal moves each sheet the way
    it faces, so the artwork ends up just outside the plain sheet and wins from
    every direction. Culling backfaces in the shader instead does not work: on
    the playfield it is the artwork sheet Blender counts as the back one, so
    that erases the surface rather than the duplicate.
    """
    for obj in objs:
        if obj.type != "MESH" or not obj.data.polygons:
            continue
        # Snapshot first: moving a vertex invalidates the cached normals.
        normals = [v.normal.copy() for v in obj.data.vertices]
        for vert, normal in zip(obj.data.vertices, normals):
            vert.co += normal * SHEET_SPLIT


def select_verts(obj: bpy.types.Object, predicate) -> int:
    count = 0
    for vert in obj.data.vertices:
        hit = predicate(vert.co)
        vert.select = hit
        count += int(hit)
    for edge in obj.data.edges:
        edge.select = False
    for poly in obj.data.polygons:
        poly.select = False
    return count


def in_disc(co: Vector, centre_world, radius: float) -> bool:
    if not (MALLET_PICK_Z[0] < co.z < MALLET_PICK_Z[1]):
        return False
    return (co.x - centre_world[0]) ** 2 + (co.y - centre_world[1]) ** 2 < radius ** 2


def import_table_and_mallet() -> bpy.types.Object:
    """Place the table in the room and lift one of its two moulded-in mallets
    out into its own object, deleting the other. Cloning the table's own mallet
    keeps the three in the relay identical to each other -- the condition the
    velocity swap rests on -- and identical in style to the table they slide
    on, which a procedural stand-in would not be."""
    parts = import_glb(TABLE_GLB)
    split_double_sided_sheets(parts)
    polished = polish_playfield(parts)
    table = join_meshes(parts, "air_hockey_table")
    table.scale = (TABLE_SCALE, TABLE_SCALE, TABLE_SCALE)
    table.location = TABLE_ORIGIN
    make_active(table)
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=True)

    centre_b = tw(*MODEL_MALLET_B, MODEL_SURFACE_Z)[:2]
    centre_a = tw(*MODEL_MALLET_A, MODEL_SURFACE_Z)[:2]

    picked = select_verts(table, lambda co: in_disc(co, centre_b, MALLET_PICK_RADIUS))
    if picked < 32:
        raise RuntimeError(
            f"Only {picked} vertices picked for the master mallet; the pick "
            "region no longer matches the model."
        )
    make_active(table)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_mode(type="VERT")
    bpy.ops.mesh.separate(type="SELECTED")
    bpy.ops.object.mode_set(mode="OBJECT")
    master = next(o for o in bpy.context.selected_objects if o is not table)
    master.name = "mallet_master"

    removed = select_verts(table, lambda co: in_disc(co, centre_a, MALLET_PICK_RADIUS))
    if removed:
        make_active(table)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_mode(type="VERT")
        bpy.ops.mesh.delete(type="VERT")
        bpy.ops.object.mode_set(mode="OBJECT")

    mn, mx = world_bbox(master)
    diameter = max(mx.x - mn.x, mx.y - mn.y)
    height = mx.z - mn.z
    print(
        f"[INFO] Table placed: surface z={SURFACE_Z:.4f}, field "
        f"{TABLE_WIDTH:.3f} x {TABLE_LEN:.3f} m. Mallet master: "
        f"{len(master.data.vertices)} verts, diameter {diameter:.4f} m, height "
        f"{height:.4f} m, base z {mn.z:.4f}; erased {removed} verts of the "
        f"second mallet; polished {polished} printed surfaces."
    )
    if abs(diameter - 2 * MALLET_RADIUS) > 0.010 or abs(height - MALLET_HEIGHT) > 0.010:
        print(
            f"[WARN] The extracted mallet does not match the physics cylinder "
            f"(r={MALLET_RADIUS:.4f}, h={MALLET_HEIGHT:.4f}); the pick region "
            "probably caught table geometry."
        )
    if abs(mn.z - SURFACE_Z) > 0.004:
        print(
            f"[WARN] The mallet's base sits at {mn.z:.4f} but the surface is at "
            f"{SURFACE_Z:.4f}; the clones will float or sink by the difference."
        )

    bpy.context.scene.cursor.location = (
        (mn.x + mx.x) / 2.0, (mn.y + mx.y) / 2.0, (mn.z + mx.z) / 2.0,
    )
    make_active(master)
    bpy.ops.object.origin_set(type="ORIGIN_CURSOR")
    master.rotation_mode = "QUATERNION"
    return master


def build_mallets(
    master: bpy.types.Object, active: list[int]
) -> list[bpy.types.Object | None]:
    """Clone the extracted mallet three times and colour the copies apart.

    The three are geometrically identical -- that is the condition the velocity
    swap rests on -- so nothing but colour may distinguish them. Painting them
    blue, red and white in the order the relay runs lets a viewer follow which
    disc carries the momentum, which reading three identical white discs does
    not allow. The clones each get their own mesh copy, otherwise they would
    share the master's material slots and all three take the last colour set.

    A mallet a DELETE edit switched off is never cloned, so it is absent from
    the render rather than hidden: its slot in the returned list is None and
    every consumer skips it. The other two keep their index, colour and start
    position, so the edit reads as "that disc is gone", not "the scene was
    rearranged".
    """
    mallets: list[bpy.types.Object | None] = []
    for index, (name, colour) in enumerate(MALLET_COLOURS):
        if not int(active[index]):
            mallets.append(None)
            continue
        clone = master.copy()
        clone.data = master.data.copy()
        clone.name = f"mallet_{index}"
        bpy.context.scene.collection.objects.link(clone)
        clone.rotation_mode = "QUATERNION"

        material = bpy.data.materials.new(f"mallet_{name}")
        material.use_nodes = True
        bsdf = material.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (*colour, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.28
        bsdf.inputs["Specular"].default_value = 0.5
        clone.data.materials.clear()
        clone.data.materials.append(material)
        # The mesh came out of the table, so its faces still point at whatever
        # slot indices the table used; without this only some faces repaint.
        for poly in clone.data.polygons:
            poly.material_index = 0
        mallets.append(clone)
    bpy.data.objects.remove(master, do_unlink=True)
    return mallets


# --- Physics -----------------------------------------------------------------

def run_physics(args: argparse.Namespace, scenario: dict) -> dict:
    physics_python = WORKSPACE_DIR.parent / "miniconda" / "envs" / "physics" / "bin" / "python"
    python = str(physics_python) if physics_python.exists() else (
        shutil.which("python3") or shutil.which("python")
    )
    if not python:
        raise RuntimeError("Cannot find python3/python for the PyBullet physics simulation.")
    script = Path(__file__).with_name("simulate_air_hockey_chain.py")
    out = args.out_dir / PHYSICS_TEMP
    physics = scenario["physics"]

    def triple(key: str) -> list[str]:
        values = physics[key]
        if len(values) != 3:
            raise ValueError(f"scenario physics {key!r} must have 3 entries, got {values!r}")
        return [str(float(v)) for v in values]

    subprocess.run(
        [
            python, str(script),
            "--out", str(out),
            "--fps", str(int(args.fps)),
            "--duration-sec", str(float(args.duration_sec)),
            "--push-speed", str(float(physics["push_speed"])),
            "--surface-friction", str(float(physics["surface_friction"])),
            "--table-restitution", str(float(physics["table_restitution"])),
            "--gravity-z", str(float(physics["gravity_z"])),
            "--mallet-masses", *triple("mallet_masses"),
            "--mallet-restitutions", *triple("mallet_restitutions"),
            "--mallet-frictions", *triple("mallet_frictions"),
            "--mallet-active", *[str(int(v)) for v in physics["mallet_active"]],
        ],
        check=True,
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    out.unlink(missing_ok=True)
    return data


def apply_keyframes(mallets: list[bpy.types.Object | None], physics: dict) -> None:
    for record in physics["frames"]:
        frame = int(record["frame_index"])
        for index, obj in enumerate(mallets):
            if obj is None:                      # deleted by an edit
                continue
            data = record[f"mallet_{index}"]
            obj.location = sw(*data["location"])
            obj.rotation_quaternion = sim_quat_to_world(data["quaternion_xyzw"])
            obj.keyframe_insert(data_path="location", frame=frame)
            obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)
    for obj in mallets:
        if obj is None:
            continue
        if obj.animation_data and obj.animation_data.action:
            for fcurve in obj.animation_data.action.fcurves:
                for key in fcurve.keyframe_points:
                    key.interpolation = "LINEAR"


# --- Scene -------------------------------------------------------------------

def look_at(obj, target) -> None:
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


def enable_gpu() -> None:
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for dev_type in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = dev_type
                break
            except TypeError:
                continue
        prefs.get_devices()
        used = 0
        for device in prefs.devices:
            device.use = device.type != "CPU"
            used += int(device.use)
        print(f"[INFO] Cycles compute: {prefs.compute_device_type}, {used} GPU device(s)")
    except Exception as e:  # noqa: BLE001 - GPU setup is best-effort
        print(f"[WARN] GPU device setup failed ({e}); Cycles will fall back.")


def build_scene(args: argparse.Namespace, physics: dict, scenario: dict):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.resolution_x = args.resolution[0]
    scene.render.resolution_y = args.resolution[1]
    scene.render.fps = args.fps
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.samples
    scene.cycles.device = "GPU" if args.device == "auto" else "CPU"
    if args.device == "auto":
        enable_gpu()
    scene.cycles.max_bounces = 8
    scene.cycles.use_denoising = True
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.view_settings.exposure = 0.0

    # A dim, slightly cool world. Both GLBs carry emission as a proper mask --
    # Base Color and Emission are different textures -- so the arcade cabinets,
    # the lava lamp and the ceiling fixture glow while everything else stays
    # unlit. That glow is the room's whole character, so it is left alone and
    # the lamps below only add what a fixture over the table would give.
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    for node in list(nodes):
        nodes.remove(node)
    background = nodes.new(type="ShaderNodeBackground")
    background.inputs["Color"].default_value = (0.32, 0.36, 0.48, 1.0)
    background.inputs["Strength"].default_value = WORLD_STRENGTH
    output = nodes.new(type="ShaderNodeOutputWorld")
    links.new(background.outputs["Background"], output.inputs["Surface"])

    join_meshes(import_glb(ROOM_GLB), "living_room")
    master = import_table_and_mallet()
    mallets = build_mallets(master, scenario["physics"]["mallet_active"])
    apply_keyframes(mallets, physics)

    # What the camera and the lamps are aimed at. Now that the relay is spread
    # over the quarter, half and three-quarter points it runs the whole length
    # of the table and ends against the far rail, so the middle of the table is
    # also the middle of the action -- back when the mallets were packed into
    # the far half this had to be pulled off centre to keep them from crowding
    # one side of frame.
    relay_centre = sw(TABLE_LEN / 2.0, RELAY_Y, 0.0)
    relay_aim = (relay_centre[0], relay_centre[1], SURFACE_Z)

    # Key: the room's own pendant. Standing the lamp where the fixture actually
    # hangs, instead of on a rig of its own directly over the table, is what
    # makes the shading read as this room's: every shadow in the shot runs away
    # from the one light source the shot can see, and they rake along the table
    # rather than dropping straight down, which is what a single bulb hung over
    # a room does. Still small for its power, so each mallet throws a defined
    # shadow onto the playing surface -- at this camera height those shadows
    # are what tie the discs to the surface instead of leaving them looking
    # pasted onto it.
    add_area_light(
        "ceiling_pendant", (*CEILING_LAMP_XY, CEILING_LAMP_Z),
        power=PENDANT_POWER, size=0.28, target=relay_aim,
        color=(1.00, 0.89, 0.74),
    )

    # Second motivated source: daylight at the window on the -X wall, raking
    # across the room between the two arcade cabinets that flank it. Being cool
    # against the warm pendant, it is what separates the far side of the
    # cabinet from the near side instead of leaving the whole thing one tone.
    add_area_light(
        "window", (WINDOW_XY[0] + 0.12, WINDOW_XY[1], WINDOW_Z),
        power=WINDOW_POWER, size=1.5, target=(0.30, 0.40, 1.00),
        color=(0.62, 0.75, 1.00),
    )

    # A wide, weak bounce for what the tiled floor and the near wall throw back.
    # Under the pendant alone the face of the cabinet turned towards camera sits
    # in its own shadow and goes to near-black.
    add_area_light(
        "floor_bounce", (2.20, -2.40, 1.10), power=BOUNCE_POWER, size=3.0,
        target=(0.30, 0.70, 0.90), color=(0.95, 0.90, 0.82),
    )

    # Camera: a three-quarter view from the room's -Y/+X corner, 3.9 m back on
    # a 42 mm lens, which puts the cabinet across most of the frame and still
    # leaves floor visible all the way around it -- that ring of open floor is
    # what says the table is standing in the middle of the room rather than
    # shoved against something. This is the corner to shoot from because the
    # room model only carries walls on -X and +Y: looking that way stands the
    # Metal Slug cabinet and the TV behind the table and keeps the missing +X
    # wall, which renders as a grey void, behind the lens. Raised ~15 degrees
    # to see over the near rail onto the playing surface the discs slide on.
    cam_loc = (2.44, -2.41, 2.00)
    framing_centre = sw(TABLE_LEN * CAMERA_AIM_ALONG_TABLE, RELAY_Y, 0.0)
    cam_target = (framing_centre[0], framing_centre[1], SURFACE_Z + 0.03)
    bpy.ops.object.camera_add(location=cam_loc)
    camera = bpy.context.object
    camera.data.lens = CAMERA_LENS_MM
    camera.data.sensor_width = 36.0
    scene.camera = camera
    target = bpy.data.objects.new("camera_target", None)
    scene.collection.objects.link(target)
    target.location = cam_target
    constraint = camera.constraints.new("TRACK_TO")
    constraint.target = target
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"
    camera.data.dof.use_dof = True
    camera.data.dof.focus_object = target
    camera.data.dof.aperture_fstop = CAMERA_FSTOP

    frame_end = int(physics["frame_end"])
    scene.frame_start = 1
    scene.frame_end = frame_end
    scene.frame_set(1)
    return mallets, camera


def export_ground_truth(out_dir: Path, mallets, camera, physics: dict, scenario: dict) -> None:
    scene = bpy.context.scene
    fps = int(physics["fps"])
    frame_end = int(physics["frame_end"])
    records = {
        "schema_version": 1,
        "fps": fps,
        "frame_start": 1,
        "frame_end": frame_end,
        "sim_to_world": {
            "yaw_deg": SIM_YAW_DEG,
            "origin": [FIELD_CENTRE_X, FIELD_FAR_Y, SURFACE_Z],
        },
        "table": {
            "scale": TABLE_SCALE,
            "origin": list(TABLE_ORIGIN),
            "surface_z": SURFACE_Z,
            "field": [TABLE_WIDTH, TABLE_LEN],
        },
        "physics": {k: v for k, v in physics.items() if k != "frames"},
        # A deleted mallet keeps its slot with present=false rather than
        # vanishing from the mapping, so a consumer diffing an edit against its
        # source sees three slots on both sides.
        "objects": {
            f"mallet_{i}": (
                {"present": False, "object_name": None} if o is None
                else {"present": True, "object_name": o.name}
            )
            for i, o in enumerate(mallets)
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
    by_frame = {int(f["frame_index"]): f for f in physics["frames"]}
    for frame in range(1, frame_end + 1):
        scene.frame_set(frame)
        source = by_frame[frame]
        entry = {"frame_index": frame, "time_sec": (frame - 1) / float(fps)}
        for index, obj in enumerate(mallets):
            if obj is None:
                entry[f"mallet_{index}"] = {"present": False}
                continue
            entry[f"mallet_{index}"] = {
                "present": True,
                "matrix_world": [[float(v) for v in row] for row in obj.matrix_world],
                "linear_velocity": source[f"mallet_{index}"]["linear_velocity"],
                "angular_velocity": source[f"mallet_{index}"]["angular_velocity"],
            }
        entry["camera_matrix_world"] = [
            [float(v) for v in row] for row in camera.matrix_world
        ]
        records["frames"].append(entry)
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
    (args.out_dir / SCENARIO_METADATA_NAME).write_text(
        json.dumps(scenario, indent=2), encoding="utf-8"
    )

    physics = run_physics(args, scenario)
    mallets, camera = build_scene(args, physics, scenario)
    export_ground_truth(args.out_dir, mallets, camera, physics, scenario)

    if args.mode == "preview":
        render_preview(args)
    else:
        render_animation(args)

    bpy.ops.wm.save_as_mainfile(filepath=str(args.out_dir / f"{OUTPUT_STEM}.blend"))
    print(f"[INFO] Render complete. Output: {args.out_dir}")


if __name__ == "__main__":
    main()
