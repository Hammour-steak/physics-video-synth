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
from mathutils import Matrix, Vector

WORKSPACE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = WORKSPACE_DIR / "assets" / "models"
POLYHAVEN_DIR = WORKSPACE_DIR / "assets" / "polyhaven"
AMBIENTCG_DIR = WORKSPACE_DIR / "assets" / "ambientcg"

CAR_GLB = MODELS_DIR / "mini_cooper_s.glb"
ROOM_GLB = MODELS_DIR / "dining_room__kichen_baked.glb"
VASE_GLB = MODELS_DIR / "flowers_in_vase.glb"
BALL_GLB = MODELS_DIR / "pixar_ball.glb"
BOOK_STACK_GLB = MODELS_DIR / "harry_potter_books_stack.glb"

OUTPUT_STEM = "car_gap_jump"
PHYSICS_TEMP = "physics_transforms.json"
GROUND_TRUTH_NAME = "ground_truth_transforms.json"
SCENARIO_METADATA_NAME = "scenario_metadata.json"

# --- Geometry (must match simulate_car_gap_jump.py) --------------------------
# A 1:24 die-cast toy Mini Cooper S (~16 cm long) doing an indoor tabletop stunt
# inside the baked dining-room environment: it rolls along a stack of hardback
# books at the edge of a dining table, launches off the front of the stack,
# and jumps the gap to a second table of the same height (or falls short to
# the room floor).
CAR_HALF_LENGTH = 0.080    # along travel (+X)
CAR_HALF_WIDTH = 0.036
CAR_HALF_HEIGHT = 0.030
CAR_TARGET_LENGTH = 2.0 * CAR_HALF_LENGTH  # 16.0 cm = 3.85 m Mini Cooper S at 1:24

DECK_TOP_Z = 0.0           # launch tabletop height in the sim frame
TABLE_THICKNESS = 0.04
DECK_WIDTH = 0.60
APPROACH_LEN = 1.05        # launch table length; its front edge is at x=0

# The launch pad is assets/models/harry_potter_books_stack.glb -- four real
# hardbacks. The scanned stack is a shallow wedge -- its base and its top cover
# are about 2 degrees out of parallel -- so resting it flat and having a level
# top are not both achievable by rotating it. The rotation below levels its
# measured BASE plane (rest it on the raw bounding box instead and it teeters
# on one corner with daylight underneath), and the remaining wedge is then
# taken out of the mesh itself, so the cover ends up level too. See
# straighten_stack_cover().
STACK_BASE_LEVEL_ROT_X_DEG = -0.6698
STACK_BASE_LEVEL_ROT_Y_DEG = 1.3374
BOOK_STACK_LEN = 0.241
BOOK_STACK_WIDTH = 0.162
BOOK_STACK_H = 0.1490
BOOK_STACK_SETBACK = 0.045

FAR_LEN = 1.50
CHASM_DEPTH = 0.74         # launch tabletop height above the room floor

# The physics runs in a "sim frame" with the launch tabletop at z=0 and the
# car travelling along +X. The dining-room environment is authored around its
# own origin (floor top at z=0.01), so the whole stunt is shifted into an
# empty stretch of the room: sim (0,0,0) maps to world (0, -1.6, 0.75),
# putting the launch table at a real 75 cm height with the room floor 74 cm
# below it.
WORLD_OFFSET = (0.0, -1.6, 0.75)
# The room GLB is yawed so its long empty half surrounds the run and the
# dining set + kitchen wall become the backdrop behind it (+Y side).
ROOM_YAW_DEG = 90.0
# The room GLB is a *baked* model: every material feeds the same baked texture
# into both Base Color and Emission, so the room lights itself and needs no
# lamps at all. That is why props standing on the tables looked pasted on -- a
# uniformly self-lit room casts no directional light, so nothing in the scene
# had a contact shadow, no matter where the lamps were put. Scaling the
# emission down lets the lamps below actually shape the scene; the baked
# texture stays on Base Color, so no detail is lost. Not zero: a little
# residual emission keeps the far side of the room from going black, which is
# roughly what bounced daylight would do anyway.
ROOM_EMISSION_SCALE = 0.18

TABLE_LEG_SIZE = 0.045
TABLE_LEG_INSET = 0.055
APRON_HEIGHT = 0.07
APRON_THICKNESS = 0.02

CAMERA_LENS_MM = 35.0
CAMERA_FSTOP = 2.8


def sw(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Sim-frame coordinates -> world coordinates."""
    return (x + WORLD_OFFSET[0], y + WORLD_OFFSET[1], z + WORLD_OFFSET[2])


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preview", "animation"), default="animation")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resolution", nargs=2, type=int, default=(1280, 720))
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-sec", type=float, default=1.6)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--preview-frame", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--launch-speed", type=float, default=2.0)
    parser.add_argument("--gap-width", type=float, default=0.28)
    parser.add_argument(
        "--car-friction", type=float, default=0.45,
        help="Forwarded to simulate_car_gap_jump.py; see there for defaults.",
    )
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


def create_wood_material(
    name: str,
    base_dir: Path,
    color_file: str,
    rough_file: str,
    normal_file: str,
    *,
    scale: float = 2.0,
    value_mult: float = 1.0,
    roughness_boost: float = 0.0,
    fallback_color=(0.35, 0.22, 0.12, 1.0),
) -> bpy.types.Material:
    """PBR wood from image textures, box-projected in object space so plain
    cube geometry (tabletops, legs) needs no UV unwrapping."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    set_input_default(bsdf, "Roughness", 0.6)
    set_input_default(bsdf, "Metallic", 0.0)

    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    mapping = nodes.new(type="ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (scale, scale, scale)
    links.new(tex_coord.outputs["Object"], mapping.inputs["Vector"])

    try:
        def img(fname, cs):
            image = bpy.data.images.load(str(require_path(base_dir, fname)), check_existing=True)
            image.colorspace_settings.name = cs
            tex = nodes.new(type="ShaderNodeTexImage")
            tex.image = image
            tex.projection = "BOX"
            tex.projection_blend = 0.25
            links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
            return tex

        color = img(color_file, "sRGB")
        if value_mult != 1.0:
            hsv = nodes.new(type="ShaderNodeHueSaturation")
            hsv.inputs["Value"].default_value = value_mult
            links.new(color.outputs["Color"], hsv.inputs["Color"])
            links.new(hsv.outputs["Color"], bsdf.inputs["Base Color"])
        else:
            links.new(color.outputs["Color"], bsdf.inputs["Base Color"])
        rough = img(rough_file, "Non-Color")
        if roughness_boost:
            # Under a directional key these scans go glossy enough to throw a
            # broad specular sheen across the tabletop that washes the grain
            # out to near-white. Pushing roughness up spreads that highlight
            # back down into the material.
            boost = nodes.new(type="ShaderNodeMath")
            boost.operation = "ADD"
            boost.use_clamp = True
            boost.inputs[1].default_value = roughness_boost
            links.new(rough.outputs["Color"], boost.inputs[0])
            links.new(boost.outputs["Value"], bsdf.inputs["Roughness"])
        else:
            links.new(rough.outputs["Color"], bsdf.inputs["Roughness"])
        nor = img(normal_file, "Non-Color")
        nmap = nodes.new(type="ShaderNodeNormalMap")
        nmap.inputs["Strength"].default_value = 0.6
        links.new(nor.outputs["Color"], nmap.inputs["Color"])
        links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
    except FileNotFoundError as e:
        print(f"[WARN] {e}; using flat fallback color for {name}")
        set_input_default(bsdf, "Base Color", fallback_color)
    return mat


def create_tabletop_material(value_mult: float = 1.0) -> bpy.types.Material:
    return create_wood_material(
        f"tabletop_wood_{value_mult:.2f}",
        POLYHAVEN_DIR / "wood_table",
        "wood_table_diff_4k.jpg",
        "wood_table_rough_4k.jpg",
        "wood_table_nor_gl_4k.jpg",
        scale=1.6,
        value_mult=value_mult,
        roughness_boost=0.30,
    )


def create_leg_material() -> bpy.types.Material:
    return create_wood_material(
        "table_leg_pine",
        POLYHAVEN_DIR / "stained_pine",
        "stained_pine_diff_4k.jpg",
        "stained_pine_rough_4k.jpg",
        "stained_pine_nor_gl_4k.jpg",
        scale=2.5,
        value_mult=0.85,
        roughness_boost=0.25,
        fallback_color=(0.22, 0.13, 0.07, 1.0),
    )


# --- Mesh builders -----------------------------------------------------------

def add_box(name, center, dims, material, rotation_euler=(0.0, 0.0, 0.0),
            bevel: float = 0.0) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=center, rotation=rotation_euler)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dims
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel > 0.0:
        mod = obj.modifiers.new("bevel", "BEVEL")
        mod.width = bevel
        mod.segments = 2
    if material is not None:
        obj.data.materials.append(material)
    return obj


def build_table(
    name: str, x0: float, x1: float, top_mat, leg_mat, top_z: float = 0.0,
) -> None:
    """A plain rectangular wooden table in the sim frame: tabletop top face at
    z=top_z, legs down to the room floor at z=-CHASM_DEPTH."""
    length = x1 - x0
    cx = (x0 + x1) / 2.0
    add_box(f"{name}_top", sw(cx, 0.0, top_z - TABLE_THICKNESS / 2.0),
            (length, DECK_WIDTH, TABLE_THICKNESS), top_mat, bevel=0.004)

    leg_len = CHASM_DEPTH + top_z - TABLE_THICKNESS
    leg_z = top_z - TABLE_THICKNESS - leg_len / 2.0
    leg_xs = (x0 + TABLE_LEG_INSET, x1 - TABLE_LEG_INSET)
    leg_ys = (-DECK_WIDTH / 2.0 + TABLE_LEG_INSET, DECK_WIDTH / 2.0 - TABLE_LEG_INSET)
    for i, lx in enumerate(leg_xs):
        for j, ly in enumerate(leg_ys):
            add_box(f"{name}_leg_{i}{j}", sw(lx, ly, leg_z),
                    (TABLE_LEG_SIZE, TABLE_LEG_SIZE, leg_len), leg_mat, bevel=0.003)

    # Apron rails under the top, connecting the legs.
    apron_z = top_z - TABLE_THICKNESS - APRON_HEIGHT / 2.0
    apron_len_x = (x1 - x0) - 2.0 * TABLE_LEG_INSET + TABLE_LEG_SIZE
    for j, ly in enumerate(leg_ys):
        add_box(f"{name}_apron_y{j}", sw(cx, ly, apron_z),
                (apron_len_x, APRON_THICKNESS, APRON_HEIGHT), leg_mat)
    apron_len_y = DECK_WIDTH - 2.0 * TABLE_LEG_INSET + TABLE_LEG_SIZE
    for i, lx in enumerate(leg_xs):
        add_box(f"{name}_apron_x{i}", sw(lx, 0.0, apron_z),
                (APRON_THICKNESS, apron_len_y, APRON_HEIGHT), leg_mat)


# --- Asset import ------------------------------------------------------------

def world_bbox(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    return Vector((min(xs), min(ys), min(zs))), Vector((max(xs), max(ys), max(zs)))


def import_glb_meshes(glb_path: Path) -> tuple[list[bpy.types.Object], list[str]]:
    if not glb_path.exists():
        raise FileNotFoundError(f"Model not found: {glb_path}")
    existing = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(glb_path))
    imported = [o for o in bpy.context.scene.objects if o not in existing]
    mesh_objs = [o for o in imported if o.type == "MESH"]
    if not mesh_objs:
        raise RuntimeError(f"No mesh found in {glb_path}")
    return mesh_objs, [o.name for o in imported]


def bake_and_join(mesh_objs: list[bpy.types.Object], imported_names: list[str], new_name: str) -> bpy.types.Object:
    bpy.ops.object.select_all(action="DESELECT")
    for o in mesh_objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objs[0]
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    if len(mesh_objs) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for o in mesh_objs:
            o.select_set(True)
        bpy.context.view_layer.objects.active = mesh_objs[0]
        bpy.ops.object.join()
    merged = bpy.context.view_layer.objects.active
    for nm in imported_names:
        obj = bpy.data.objects.get(nm)
        if obj is not None and obj is not merged:
            bpy.data.objects.remove(obj, do_unlink=True)
    merged.name = new_name
    return merged


def import_room() -> None:
    """The baked dining-room/kitchen interior, yawed so the empty half of the
    room hosts the stunt and the dining set + kitchen become the backdrop."""
    if not ROOM_GLB.exists():
        raise FileNotFoundError(f"Room model not found: {ROOM_GLB}")
    existing = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(ROOM_GLB))
    imported = [o for o in bpy.context.scene.objects if o not in existing]
    rot = Matrix.Rotation(math.radians(ROOM_YAW_DEG), 4, "Z")
    imported_set = set(imported)
    for obj in imported:
        if obj.parent is None or obj.parent not in imported_set:
            obj.matrix_world = rot @ obj.matrix_world

    dimmed = set()
    for obj in imported:
        for slot in getattr(obj, "material_slots", ()):
            mat = slot.material
            if mat is None or mat.name in dimmed or not mat.use_nodes:
                continue
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            strength = bsdf.inputs.get("Emission Strength") if bsdf else None
            if strength is not None:
                strength.default_value *= ROOM_EMISSION_SCALE
            dimmed.add(mat.name)
    print(f"[INFO] Room: scaled baked emission to {ROOM_EMISSION_SCALE:g} on "
          f"{len(dimmed)} materials.")


def import_car() -> bpy.types.Object:
    if not CAR_GLB.exists():
        raise FileNotFoundError(f"Car model not found: {CAR_GLB}")
    mesh_objs, imported_names = import_glb_meshes(CAR_GLB)
    obj = bake_and_join(mesh_objs, imported_names, "car")

    # The raw model's length runs along its local Y, nose at -Y. Yaw +90 deg
    # about Z so the nose points along world +X, the direction of travel (a
    # -90 yaw drives the car tail-first).
    obj.rotation_mode = "XYZ"
    obj.rotation_euler = (0.0, 0.0, math.radians(90.0))
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    # Uniformly rescale so the length (now along X) matches a 1:24 die-cast
    # GT-R (~19.6 cm).
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
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
    return obj


def fit_plane(points) -> tuple[float, float, float]:
    """Least-squares z = a x + b y + c through a set of Vectors."""
    n = float(len(points))
    sx = sum(p.x for p in points); sy = sum(p.y for p in points); sz = sum(p.z for p in points)
    sxx = sum(p.x * p.x for p in points); syy = sum(p.y * p.y for p in points)
    sxy = sum(p.x * p.y for p in points)
    sxz = sum(p.x * p.z for p in points); syz = sum(p.y * p.z for p in points)
    rows = [[sxx, sxy, sx, sxz], [sxy, syy, sy, syz], [sx, sy, n, sz]]
    for i in range(3):
        piv = max(range(i, 3), key=lambda r: abs(rows[r][i]))
        rows[i], rows[piv] = rows[piv], rows[i]
        for r in range(3):
            if r != i and rows[i][i]:
                f = rows[r][i] / rows[i][i]
                for col in range(i, 4):
                    rows[r][col] -= f * rows[i][col]
    return tuple(rows[i][3] / rows[i][i] for i in range(3))


def measure_cover(obj: bpy.types.Object) -> list:
    """Sample the stack's top surface -- the cover of its top book. Works off
    the mesh vertices rather than ray_cast, which reads a cached BVH and so
    would keep reporting the geometry as it was before straighten_stack_cover()
    moved anything. Vertices are bucketed on a grid in x/y, the highest vertex
    in each bucket is the surface there, and only buckets that reach the top of
    the stack are kept."""
    verts = obj.data.vertices
    zs = [v.co.z for v in verts]
    lo, hi = min(zs), max(zs)
    xs = [v.co.x for v in verts]
    ys = [v.co.y for v in verts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    steps = 28
    top = {}
    for v in verts:
        i = min(steps - 1, int((v.co.x - x0) / (x1 - x0) * steps))
        j = min(steps - 1, int((v.co.y - y0) / (y1 - y0) * steps))
        cur = top.get((i, j))
        if cur is None or v.co.z > cur.z:
            top[(i, j)] = v.co.copy()
    cover_floor = lo + 0.85 * (hi - lo)
    hits = [p for p in top.values() if p.z > cover_floor]
    if len(hits) < 16:
        raise RuntimeError(f"Could not find the top cover of {BOOK_STACK_GLB}")
    return hits


def measure_base_z(obj: bpy.types.Object) -> float:
    """Height of the stack's underside. Taken as the median of the lowest
    vertex in each x/y bucket that reaches the bottom of the stack, not the
    single lowest vertex: the scan's underside is bumpy by a few mm and resting
    it on one stray low point would leave the rest of the stack hovering."""
    verts = obj.data.vertices
    zs = [v.co.z for v in verts]
    lo, hi = min(zs), max(zs)
    xs = [v.co.x for v in verts]
    ys = [v.co.y for v in verts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    steps = 28
    bottom = {}
    for v in verts:
        i = min(steps - 1, int((v.co.x - x0) / (x1 - x0) * steps))
        j = min(steps - 1, int((v.co.y - y0) / (y1 - y0) * steps))
        cur = bottom.get((i, j))
        if cur is None or v.co.z < cur:
            bottom[(i, j)] = v.co.z
    floor = lo + 0.15 * (hi - lo)
    lows = sorted(z for z in bottom.values() if z < floor)
    return lows[len(lows) // 2] if lows else lo


def straighten_stack_cover(obj: bpy.types.Object) -> None:
    """Take the wedge out of the stack so its top cover is level once its base
    is. The base is already horizontal at this point, but the scan's books fan
    out, leaving the cover a couple of degrees off. Rotating that out would
    just tilt the base back, so instead the slope is removed from the mesh:
    each vertex is pulled down by the cover's slope at its own x/y, scaled by
    how far up the stack it sits. The base (weight 0) does not move, the cover
    (weight 1) comes level, and the books between straighten proportionally --
    a ~8 mm correction over a 250 mm stack, invisible in the book shapes."""
    hits = measure_cover(obj)
    a, b, _c = fit_plane(hits)
    base_z = measure_base_z(obj)
    thickness = sum(h.z for h in hits) / len(hits) - base_z
    for vert in obj.data.vertices:
        weight = min(1.0, max(0.0, (vert.co.z - base_z) / thickness))
        vert.co.z -= (a * vert.co.x + b * vert.co.y) * weight
    obj.data.update()


def import_book_stack() -> None:
    """The launch pad. The stack is rotated so its measured base plane is
    horizontal, the residual wedge is taken out of the mesh so its top cover is
    horizontal too, and it is then positioned by the surface the car actually
    uses: the midpoint of the cover's front edge goes to sim
    (-BOOK_STACK_SETBACK, 0, BOOK_STACK_H), the top-front edge of the physics
    box. BOOK_STACK_H is the stack's own thickness, so its base lands on the
    tabletop; that is checked here rather than assumed."""
    mesh_objs, imported_names = import_glb_meshes(BOOK_STACK_GLB)
    merged = bake_and_join(mesh_objs, imported_names, "book_stack")
    # bake_and_join leaves objects in quaternion rotation mode, where assigning
    # rotation_euler is silently ignored.
    merged.rotation_mode = "XYZ"
    merged.rotation_euler = (
        math.radians(STACK_BASE_LEVEL_ROT_X_DEG),
        math.radians(STACK_BASE_LEVEL_ROT_Y_DEG),
        0.0,
    )
    bpy.ops.object.select_all(action="DESELECT")
    merged.select_set(True)
    bpy.context.view_layer.objects.active = merged
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    straighten_stack_cover(merged)

    hits = measure_cover(merged)
    a, b, c = fit_plane(hits)
    residual = math.degrees(math.atan(math.hypot(a, b)))
    front_x = max(h.x for h in hits)
    mid = [h for h in hits if h.x < front_x - 0.02]
    cover_y = (min(h.y for h in mid) + max(h.y for h in mid)) / 2.0
    front_z = a * front_x + b * cover_y + c
    thickness = front_z - measure_base_z(merged)
    print(f"[INFO] Book stack: cover levelled to {residual:.2f} deg residual, "
          f"thickness {thickness:.4f} m; placing its front edge at "
          f"({-BOOK_STACK_SETBACK:.3f}, 0, {BOOK_STACK_H:.4f}).")
    if abs(thickness - BOOK_STACK_H) > 0.0015:
        print(f"[WARN] BOOK_STACK_H is {BOOK_STACK_H:.4f} but the stack is "
              f"{thickness:.4f} m thick, so its base will sit "
              f"{(BOOK_STACK_H - thickness) * 1000:.1f} mm off the tabletop.")

    merged.location = sw(-front_x - BOOK_STACK_SETBACK, -cover_y, BOOK_STACK_H - front_z)


def import_vase(location_sim: tuple[float, float, float], height: float = 0.35) -> None:
    """Static decor: a vase of flowers, origin recentred on its base so placing
    it by location sits the vase flush on the tabletop."""
    try:
        mesh_objs, imported_names = import_glb_meshes(VASE_GLB)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[WARN] Skipping vase: {e}")
        return
    merged = bake_and_join(mesh_objs, imported_names, "flowers_in_vase")
    mn, mx = world_bbox(merged)
    scale = height / (mx.z - mn.z)
    merged.scale = (scale, scale, scale)
    bpy.ops.object.select_all(action="DESELECT")
    merged.select_set(True)
    bpy.context.view_layer.objects.active = merged
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    mn, mx = world_bbox(merged)
    bpy.context.scene.cursor.location = ((mn.x + mx.x) / 2.0, (mn.y + mx.y) / 2.0, mn.z)
    bpy.ops.object.origin_set(type="ORIGIN_CURSOR")
    merged.location = sw(*location_sim)


def import_ball(location_sim: tuple[float, float, float], radius: float = 0.05) -> None:
    """Static decor: a toy ball on the room floor, off to the camera side of
    the flight path so it dresses the gap without colliding with the car."""
    try:
        mesh_objs, imported_names = import_glb_meshes(BALL_GLB)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[WARN] Skipping ball: {e}")
        return
    merged = bake_and_join(mesh_objs, imported_names, "toy_ball")
    mn, mx = world_bbox(merged)
    raw_radius = max((mx - mn)[i] for i in range(3)) / 2.0
    scale = radius / raw_radius
    merged.scale = (scale, scale, scale)
    bpy.ops.object.select_all(action="DESELECT")
    merged.select_set(True)
    bpy.context.view_layer.objects.active = merged
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    mn, mx = world_bbox(merged)
    bpy.context.scene.cursor.location = ((mn.x + mx.x) / 2.0, (mn.y + mx.y) / 2.0, (mn.z + mx.z) / 2.0)
    bpy.ops.object.origin_set(type="ORIGIN_CURSOR")
    merged.rotation_euler = (0.0, 0.0, math.radians(40.0))
    merged.location = sw(*location_sim)


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
            "--car-friction", str(float(args.car_friction)),
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
        obj.location = sw(*d["location"])
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


def add_sun(name, energy, angle_deg, rotation_deg, color=(1.0, 1.0, 1.0)) -> bpy.types.Object:
    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 3.0))
    sun = bpy.context.object
    sun.name = name
    sun.data.energy = float(energy)
    sun.data.angle = math.radians(angle_deg)
    sun.data.color = color
    sun.rotation_euler = tuple(math.radians(a) for a in rotation_deg)
    return sun


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
        for d in prefs.devices:
            d.use = d.type != "CPU"
            used += int(d.use)
        print(f"[INFO] Cycles compute: {prefs.compute_device_type}, {used} GPU device(s)")
    except Exception as e:  # noqa: BLE001 - GPU setup is best-effort
        print(f"[WARN] GPU device setup failed ({e}); Cycles will fall back.")


def build_scene(args: argparse.Namespace, physics: dict) -> tuple[bpy.types.Object, bpy.types.Object]:
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
    # The room's baked textures are pale/high-albedo (same GLB as the
    # dining_chain scene); a negative color-management exposure calms it down.
    scene.view_settings.exposure = 0.30

    gap_width = float(args.gap_width)

    # World: plain warm fill only -- the room geometry is the visible
    # background; area lights below do the real lighting.
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    wn = world.node_tree.nodes
    wl = world.node_tree.links
    for n in list(wn):
        wn.remove(n)
    bg = wn.new(type="ShaderNodeBackground")
    bg.inputs["Color"].default_value = (0.88, 0.85, 0.80, 1.0)
    bg.inputs["Strength"].default_value = 0.45
    out = wn.new(type="ShaderNodeOutputWorld")
    wl.new(bg.outputs["Background"], out.inputs["Surface"])

    # The dining-room environment.
    import_room()

    # Materials.
    top_mat_near = create_tabletop_material(1.0)
    top_mat_far = create_tabletop_material(0.88)
    leg_mat = create_leg_material()

    # Launch table: front edge at x=0, the book stack sits on it.
    build_table("launch_table", -APPROACH_LEN, 0.0, top_mat_near, leg_mat)
    # Landing table across the gap, the same height as the launch table -- the
    # car's whole drop is the book-stack height. Slightly darker top so the two
    # read as separate pieces of furniture rather than one surface.
    build_table("landing_table", gap_width, gap_width + FAR_LEN,
                top_mat_far, leg_mat)

    # The launch pad itself.
    import_book_stack()

    # Set dressing so the room reads as lived-in: a vase of flowers on the
    # landing table and a toy ball on the floor under the gap. The vase's x is
    # fixed rather than measured from the landing table's near edge, because it
    # is pinned between two limits: much further along the table and it runs
    # out of frame at the right, much nearer and it crowds where the car comes
    # to rest. It is set back in Y as well, clear of the car's centre line.
    # Neither prop carries a collision proxy.
    import_vase((1.20, 0.14, 0.0), height=0.32)
    import_ball((0.16, 0.02, -CHASM_DEPTH + 0.05))

    # Lighting. The key is deliberately small for its power: broad soft sources
    # from every side wash out the darkening right where an object meets the
    # tabletop, and without that darkening the books, the vase and the car all
    # read as pasted onto the surface rather than resting on it. A compact key
    # set well behind the run throws every prop's contact shadow forward, toward
    # the camera, where the grazing view of the tabletop can actually see it --
    # a key off to the side drops those shadows behind the props themselves. The rest are fills: a cool
    # daylight wash from the window wall (world -X), a warm rim from the
    # dining-room side, and a weak front fill -- weak on purpose, since it is
    # the one that would otherwise erase those contact shadows.
    # The key is a sun, not a lamp. An area light close enough to throw a long
    # shadow also burns a hot spot into the tabletop under it and washes the
    # wood out; a sun has no falloff, so it lights both tables evenly and every
    # prop gets the same shadow direction and length. Its elevation is what
    # matters: at ~35 degrees each prop's shadow reaches about 1.4x its own
    # height across the table, toward the camera, which is what makes the books
    # and the vase read as sitting on the surface instead of pasted onto it.
    add_sun("key_sun", energy=3.2, angle_deg=1.5, rotation_deg=(-55.0, 0.0, -20.0),
            color=(1.0, 0.95, 0.86))
    add_area_light("window_fill", (-3.2, 0.6, 1.7), power=185.0, size=2.5,
                   target=sw(0.1, 0.1, 0.0), color=(0.85, 0.90, 1.00))
    add_area_light("back_rim", sw(0.4, 1.5, 1.15), power=85.0, size=1.6,
                   target=sw(0.2, 0.0, 0.0), color=(1.0, 0.90, 0.80))
    # The key is behind the run, so the faces turned toward the camera -- the
    # book spines above all, which are dark covers to begin with -- get nothing
    # from it and crush to a flat dark mass. This fill rakes across them from
    # the camera side. It is broad and soft so it adds no competing shadow of
    # its own, and stays well below the key so the contact shadows the key
    # throws forward survive it.
    add_area_light("front_fill", sw(0.1, -1.7, 0.85), power=45.0, size=2.0,
                   target=sw(-0.1, 0.0, 0.08), color=(1.0, 0.95, 0.88))

    # Camera: low three-quarter view a little above the launch height, yawed
    # off perpendicular so the book stack and the depth of the gap read (a
    # dead-side view flattens both). Framed to span the room floor -- where a
    # fall-short car ends up -- up to comfortably above the launch. The view
    # onto the tabletop is deliberately grazing, which keeps the arc reading
    # against the room instead of against the wood; props are set forward of
    # the back edge so some tabletop still shows behind them.
    action_center_x = gap_width * 0.5 + 0.18
    cam_loc = sw(action_center_x - 0.58, -1.95, 0.36)
    cam_tgt = sw(action_center_x - 0.04, 0.0, -0.18)
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
    camera.data.dof.use_dof = True
    camera.data.dof.focus_object = tgt
    camera.data.dof.aperture_fstop = CAMERA_FSTOP

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
        "world_offset": list(WORLD_OFFSET),
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
        "environment": "indoor_dining_room_tabletop",
        "car_scale": "1:24",
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
