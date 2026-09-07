"""Scene-agnostic edit DSL for PCVE physics-edit datasets.

Grammar
-------
    Edit       := DeleteStmt | SetStmt | AddStmt
    DeleteStmt := "DELETE" ObjectId [Timing] ["HINT" QUOTED_STRING]
    SetStmt    := "SET" ObjectId "." Property Change [Timing]
                  ["HINT" QUOTED_STRING]
    Change     := "TIMES" Scalar | "TO" Value
    AddStmt    := "ADD" ObjectId "BETWEEN" ObjectId "AND" ObjectId
                  "AT" Where ["HINT" QUOTED_STRING]
    Where      := "MIDPOINT" | Int "/" Int "FROM" ObjectId
    Timing     := "AT" "FRAME" Int
    Value      := Scalar | "(" Scalar "," Scalar "," Scalar ")"

Three operations: delete an object, set a property, or add an object.

When an edit takes effect
-------------------------
An edit with no ``AT FRAME`` clause holds for the whole clip: the simulator is
started with the edited value, so the very first frame already shows it. That
is what every edit in this benchmark used to be, and the prompts now say so
out loud ("from the very first frame ...") rather than leaving it implied.

``AT FRAME n`` makes the edit land partway through instead. The clip runs on
the baseline physics up to frame ``n``, the change is applied there, and the
rest of the clip runs on. The point is that both halves of the video are then
informative: the frames before ``n`` establish what was going to happen, and
the frames after show what the edit did to it.

    SET red_ball.friction TIMES 3 AT FRAME 18
    DELETE red_stone AT FRAME 30

A timed edit is worth writing when it changes what a body *already in motion*
does next, and the comparison that decides this is against the whole-clip
version of the same edit rather than against the source video. ``DELETE x AT
FRAME n`` differs from ``DELETE x`` by exactly what ``x`` did in frames
1..n-1: taking away a body that has been sitting still and has touched nothing
yet leaves every other trajectory in the clip bit-identical, and the two videos
differ only in a motionless object being visible early on.

``n`` counts rendered frames from 1, so it is a number a viewer can find by
scrubbing. Frame 1 is not a legal target: an edit that lands on the first
frame is a whole-clip edit, and is written without the clause. The upper bound
is the scene's ``total_frames``, which the vocabulary has to declare before it
can carry a timed edit.

Timing is not offered on ADD. An ADD resolves its position from where the two
endpoint bodies sit *at the start*, and partway through the clip those bodies
have moved, so the same statement would no longer name the same point.

SET states the change as a multiple of the property's baseline, not as a pair
of absolute numbers:

    SET cue_ball.mass TIMES 5

reads "make the cue ball five times as heavy as it is". The baseline lives in
the scene's vocabulary, so the DSL never repeats it and the prompts never have
to quote a number a viewer cannot see -- "five times heavier" is something the
eye can check against the source video, "0.85 kg" is not. Factors below 1 are
reductions: ``TIMES 0.25`` is a quarter of the baseline. Factors are written
and spoken as decimals -- every one in the suite is a value a decimal writes
exactly -- so no prompt ever asks a reader to divide.

The absolute form, ``SET stones.restitution TO 0.95``, is the escape hatch for
the one case a factor cannot express: a baseline of exactly 0, which no
multiple can move off. Prefer ``TIMES`` everywhere else.

ADD states the new object's position entirely in relative terms -- no scene
coordinates appear in the DSL or in the prompts it generates. The position is
a division point of the straight line between two objects already in the
scene:

    ADD yellow_ball BETWEEN cue_ball AND target_ball AT MIDPOINT
    ADD yellow_ball BETWEEN cue_ball AND target_ball AT 1/4 FROM cue_ball

reads "put the yellow ball at the midpoint of the cue-ball-to-target-ball
line", and "...a quarter of the way along that line, measured from the cue
ball". Both halves are relative: the line is named by two objects rather than
an axis, and the position is a division point of that line rather than a
length in metres, so the same statement means the same thing in a scene whose
bodies are centimetres apart and one whose bodies are metres apart. The anchor
must be one of the two endpoints, and ``k/n`` is measured from it as a
fraction of the whole line -- ``1/4 FROM cue_ball`` and ``3/4 FROM
target_ball`` name the same point.

Per-scene setup
---------------
Each scene declares a ``Vocabulary`` that says:
  * which object ids exist and their zh/en display names;
  * which properties exist (name, kind, unit, zh/en);
  * how each (object, property) pair maps to the scene's sim parameter dict
    (``sim_bindings``), and how DELETE and ADD map to the sim parameter dict
    (``delete_bindings``, ``add_bindings``);
  * where each object's centre and radius live in the physics dict
    (``geometry``), which is what lets ADD resolve a relative position into
    the coordinates the simulator wants;
  * the baseline physics dict, which is now load-bearing twice over: it is
    where SET reads the value a factor multiplies, and where physics_diff
    reads the "from" side of every edit.

Once the vocabulary is defined, the whole pipeline is data-driven:
    parse(dsl_str, vocab)                -> Edit
    render(edit, vocab, lang, precise)   -> natural-language prompt
    to_physics_override(edit, vocab)     -> {sim_param: new_value, ...}
    to_scenario_override(edit, vocab)    -> the physics block the render takes
                                            (the same values, wrapped in a
                                            timed_edits schedule when the edit
                                            lands partway through)
    starts_at_frame(edit)                -> 1, or the frame the edit lands on
    baseline_value_for(edit, vocab)      -> baseline value (for physics_diff)
    resolved_to_value(edit, vocab)       -> value the factor resolves to
    add_location(edit, vocab)            -> resolved centre of an ADDed body
    add_position_diff(edit, vocab)       -> physics_diff's `position` block
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Union

Scalar = float
Vector = tuple[float, float, float]
Value = Union[Scalar, Vector]


# ---------------------------------------------------------------- data types


@dataclass(frozen=True)
class DeleteEdit:
    """Take one body out of the scene.

    ``at_frame`` is None for the usual whole-clip delete (the body is never
    built at all) and a frame number for a delete that lands partway through:
    the body is there, moving as it does in the source, and then it is gone.
    """
    object_id: str
    at_frame: int | None = None


@dataclass(frozen=True)
class SetEdit:
    """Move one property off its baseline.

    ``factor`` is the multiple the DSL was written in (``TIMES 5``), and is
    None only for the absolute ``TO`` form. ``from_value`` and ``to_value``
    are the two ends the factor resolves to; both are filled in at parse time
    when a vocabulary is supplied, and stay None when it is not, because the
    baseline a factor multiplies lives in the vocabulary rather than in the
    edit string.
    """
    object_id: str
    property_name: str
    factor: float | None = None
    from_value: Value | None = None
    to_value: Value | None = None
    hint: str | None = None
    # None: the simulator starts on the edited value, so the whole clip shows
    # it. A frame number: the clip runs on the baseline up to that frame and
    # the property is changed there.
    at_frame: int | None = None

    @property
    def is_relative(self) -> bool:
        return self.factor is not None


@dataclass(frozen=True)
class AddEdit:
    """Put a new body at a division point of the line between two existing ones.

    ``numerator``/``denominator`` are the ``k/n`` of the DSL, measured from
    ``anchor_id`` as a fraction of the whole line; the anchor must be either
    ``endpoint_a`` or ``endpoint_b``. The midpoint form carries ``1/2`` with
    ``anchor_id`` None, because a midpoint has no near end to measure from.
    """
    object_id: str
    endpoint_a: str
    endpoint_b: str
    numerator: int
    denominator: int
    anchor_id: str | None = None
    hint: str | None = None

    @property
    def is_midpoint(self) -> bool:
        return self.anchor_id is None

    @property
    def other_id(self) -> str | None:
        """The endpoint the fraction is *not* measured from."""
        if self.anchor_id is None:
            return None
        return self.endpoint_b if self.anchor_id == self.endpoint_a else self.endpoint_a

    @property
    def fraction_from_a(self) -> float:
        """How far along endpoint_a -> endpoint_b the new body sits, in [0, 1]."""
        t = self.numerator / self.denominator
        if self.anchor_id is None or self.anchor_id == self.endpoint_a:
            return t
        return 1.0 - t


Edit = Union[DeleteEdit, SetEdit, AddEdit]


@dataclass(frozen=True)
class PropertySpec:
    kind: str          # "scalar" or "vector"
    unit: str          # "kg", "m/s", "" ...
    zh: str
    en: str


@dataclass(frozen=True)
class SimBinding:
    """Where writing a SET edit lands in the scene's physics dict.

    ``key`` is the top-level physics parameter name (e.g. ``"ball_mass"``).
    If ``index`` is not None, the parameter is a list and this binding
    writes at that index (other slots are copied from ``baseline_physics``).
    """
    key: str
    index: int | None = None


@dataclass(frozen=True)
class CompoundBinding:
    """One editable property that scales several sim parameters together.

    The scene has multiple physical knobs that behave as a single thing to a
    viewer -- most commonly lateral friction plus rolling friction, which the
    edit vocabulary collapses into one ``friction`` property. Editing
    ``friction`` from ``x`` to ``y`` multiplies every component's baseline by
    ``y / x``; the value seen in the DSL and in prompts is ``components[0]``'s
    baseline (the "display" component).

    ``components[0]`` must have a non-zero baseline, or the ratio has no
    meaning. Non-zero baselines on later components carry their proportional
    share; a baseline of exactly 0 is left at 0 (scaling nothing by anything
    is nothing), which is the intended behaviour for a body that carries no
    rolling friction of its own.
    """
    components: tuple[SimBinding, ...]


@dataclass(frozen=True)
class DeleteBinding:
    """Where writing a DELETE edit lands in the scene's physics dict.

    The physics parameter at ``key`` is a list; the entry at ``index`` is
    set to ``value_when_removed`` (usually 0). Baseline is inspected to fill
    the other slots.
    """
    key: str
    index: int
    value_when_removed: Any = 0


@dataclass(frozen=True)
class GeometryAnchor:
    """Where an object's baseline centre and radius live in the physics dict.

    ADD places the new body on the line between two objects already in the
    scene, so the vocabulary has to say where those objects *are*.
    ``location_key`` names a 3-vector physics parameter (or, with
    ``location_index``, one entry of a list of them); ``radius_key`` names the
    scalar the scene keeps that object's radius in. Bodies that share one
    radius parameter -- a rack of identical billiard balls -- all point at the
    same key.

    Centres are read in whatever frame the scene's own parameters use, and the
    resolved point is written back in that same frame. For pool that is table
    coordinates with z measured up from the felt, which is why interpolating
    all three components is exactly "on the line".
    """
    location_key: str
    radius_key: str
    location_index: int | None = None
    radius_index: int | None = None


@dataclass(frozen=True)
class AddBinding:
    """Where writing an ADD edit lands in the scene's physics dict.

    DELETE's mirror image. The body must already exist in the scene's
    parameter set as a slot that is *off* at baseline: the physics parameter
    at ``presence_key`` is a list whose entry at ``presence_index`` reads 0 in
    ``baseline_physics``, and ADD sets it to ``value_when_present``. The centre
    the edit resolves to is written at ``location_key``.

    ``defaults`` carries any other physics keys that have to come on with the
    body -- its mass, its friction, and so on, for scenes that keep those in
    separate parameters. Each pair is written verbatim.
    """
    presence_key: str
    presence_index: int
    location_key: str
    value_when_present: Any = 1
    location_index: int | None = None
    defaults: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class Vocabulary:
    objects: dict[str, dict[str, str]]                    # obj_id -> {zh, en}
    properties: dict[str, PropertySpec]                   # prop_name -> spec
    sim_bindings: dict[tuple[str, str], SimBinding | CompoundBinding]  # (obj, prop) -> where
    delete_bindings: dict[str, DeleteBinding]             # obj -> where
    baseline_physics: dict[str, Any]
    # Both default to empty so every scene written before ADD existed keeps
    # constructing unchanged; a scene without them simply has no ADD edits.
    geometry: dict[str, GeometryAnchor] = field(default_factory=dict)
    add_bindings: dict[str, AddBinding] = field(default_factory=dict)
    # How many frames the suite's videos are, which is what bounds an
    # ``AT FRAME n`` clause and what turns that n into "early / midway /
    # late in the clip" for the vague prompt. None until a scene declares it;
    # a scene that has not is simply one that cannot carry a timed edit yet.
    total_frames: int | None = None

    def check_object(self, name: str) -> None:
        if name not in self.objects:
            raise ValueError(f"Unknown object: {name!r}. Allowed: {sorted(self.objects)}")

    def check_geometry(self, name: str) -> None:
        self.check_object(name)
        if name not in self.geometry:
            raise ValueError(
                f"Object {name!r} has no geometry anchor, so it cannot be used "
                f"to position an ADD. Allowed: {sorted(self.geometry)}"
            )

    def check_property(self, name: str) -> None:
        if name not in self.properties:
            raise ValueError(f"Unknown property: {name!r}. Allowed: {sorted(self.properties)}")

    def check_frame(self, frame: int) -> None:
        """A timed edit has to land on a frame the clip actually has.

        Frame 1 is rejected on purpose: an edit that lands there is a
        whole-clip edit, and writing it two different ways would put two
        spellings of the same thing in the benchmark.
        """
        if self.total_frames is None:
            raise ValueError(
                "This scene's vocabulary does not declare total_frames, so it "
                "cannot say whether a frame number is inside the clip. Add "
                "total_frames=<frames in the suite's videos> to the Vocabulary."
            )
        if frame < 2:
            raise ValueError(
                f"AT FRAME {frame} is not a partway edit. An edit that holds "
                "from the first frame is written without the AT FRAME clause."
            )
        if frame > self.total_frames:
            raise ValueError(
                f"AT FRAME {frame} is past the end of the clip, which is "
                f"{self.total_frames} frames long."
            )


# --------------------------------------------------------------------- parse

_VEC = re.compile(r"\(\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\)")
# TIMES takes a bare scalar, optionally written as the fraction the prompt
# will read back ("TIMES 1/4"); TO takes a scalar or a 3-vector.
_SET_TIMES = re.compile(
    r"^SET\s+(\w+)\.(\w+)\s+TIMES\s+(\d+\s*/\s*\d+|[-\d.eE+]+)"
    r"(?:\s+HINT\s+\"([^\"]*)\")?\s*$"
)
_SET_TO = re.compile(
    r"^SET\s+(\w+)\.(\w+)\s+TO\s+(\S.*?)"
    r"(?:\s+HINT\s+\"([^\"]*)\")?\s*$"
)
_DEL = re.compile(r"^DELETE\s+(\w+)\s*$")
# The timing clause sits after the statement and before any HINT, and is
# lifted off before the statement itself is matched -- which is what keeps the
# three statement patterns below exactly as they were. Anchoring on the
# optional trailing HINT is what stops an "AT FRAME 30" written *inside* a
# hint string from being read as timing.
_TIMING = re.compile(
    r"^(?P<body>.*?)\s+AT\s+FRAME\s+(?P<frame>\d+)"
    r"(?P<hint>(?:\s+HINT\s+\"[^\"]*\")?)\s*$"
)
_ADD = re.compile(
    r"^ADD\s+(\w+)\s+BETWEEN\s+(\w+)\s+AND\s+(\w+)"
    r"\s+AT\s+(?:(MIDPOINT)|(\d+)\s*/\s*(\d+)\s+FROM\s+(\w+))"
    r"(?:\s+HINT\s+\"([^\"]*)\")?\s*$"
)


def _parse_value(s: str) -> Value:
    s = s.strip()
    m = _VEC.fullmatch(s)
    if m:
        return (float(m.group(1)), float(m.group(2)), float(m.group(3)))
    return float(s)


def _parse_factor(s: str) -> float:
    """``"5"``, ``"0.35"`` and ``"1/4"`` all read as a plain multiplier."""
    s = s.strip()
    if "/" in s:
        num, den = s.split("/", 1)
        den_v = float(den)
        if den_v == 0.0:
            raise ValueError(f"Factor {s!r} divides by zero")
        return float(num) / den_v
    return float(s)


def _round(x: float) -> float:
    """Trim the float noise a multiplication leaves behind.

    0.343 * 5 lands on 1.7149999999999999, which would then be written into
    the scenario overrides and quoted in the manifest's physics_diff. Ten
    significant digits is far more precision than any of these parameters
    carry and enough to make the product read as the number it is.
    """
    return float(f"{x:.10g}")


def _scale(value: Any, factor: float) -> Any:
    if isinstance(value, (list, tuple)):
        return type(value)(_round(float(c) * factor) for c in value)
    return _round(float(value) * factor)


def _check_kind(prop: str, v: Value, kind: str) -> None:
    if kind == "scalar" and not isinstance(v, float):
        raise ValueError(f"{prop!r} expects scalar, got {v!r}")
    if kind == "vector" and not (isinstance(v, tuple) and len(v) == 3):
        raise ValueError(f"{prop!r} expects 3-vector, got {v!r}")


def _split_timing(text: str) -> tuple[str, int | None]:
    """Lift an ``AT FRAME n`` clause off a statement, if it carries one."""
    m = _TIMING.match(text)
    if not m:
        return text, None
    return (m.group("body") + m.group("hint")).strip(), int(m.group("frame"))


def parse(text: str, vocab: Vocabulary | None = None) -> Edit:
    text, at_frame = _split_timing(text.strip())
    if at_frame is not None and vocab:
        vocab.check_frame(at_frame)
    m = _DEL.match(text)
    if m:
        obj = m.group(1)
        if vocab:
            vocab.check_object(obj)
            if obj not in vocab.delete_bindings:
                raise ValueError(f"Object {obj!r} cannot be deleted in this scene")
        return DeleteEdit(object_id=obj, at_frame=at_frame)
    m = _ADD.match(text)
    if m:
        if at_frame is not None:
            raise ValueError(
                "ADD cannot be timed. It places the new body on the line "
                "between two others as they sit at the start of the clip, and "
                "partway through they are somewhere else."
            )
        obj, end_a, end_b, midpoint, num_s, den_s, anchor, hint = m.groups()
        if end_a == end_b:
            raise ValueError(
                f"ADD needs two different endpoints to define a line, got "
                f"{end_a!r} twice"
            )
        if midpoint:
            num, den, anchor = 1, 2, None
        else:
            num, den = int(num_s), int(den_s)
            if den < 2:
                raise ValueError(
                    f"ADD divides the line into {den} parts; a division point "
                    "needs at least 2"
                )
            if not 0 < num < den:
                raise ValueError(
                    f"ADD at {num}/{den} is not a division point strictly "
                    f"between {end_a!r} and {end_b!r}"
                )
            if anchor not in (end_a, end_b):
                raise ValueError(
                    f"ADD measures from one of its own endpoints; {anchor!r} "
                    f"is neither {end_a!r} nor {end_b!r}"
                )
        edit = AddEdit(obj, end_a, end_b, num, den, anchor, hint or None)
        if vocab:
            vocab.check_object(obj)
            if obj not in vocab.add_bindings:
                raise ValueError(f"Object {obj!r} cannot be added in this scene")
            vocab.check_geometry(end_a)
            vocab.check_geometry(end_b)
            vocab.check_geometry(obj)
            # Resolving now is the point: an ADD that would drop the new body
            # inside one of its own endpoints is a bad edit, and this is the
            # cheapest place to find that out.
            add_location(edit, vocab)
        return edit
    m = _SET_TIMES.match(text)
    if m:
        obj, prop, factor_s, hint = m.groups()
        factor = _parse_factor(factor_s)
        edit = SetEdit(obj, prop, factor=factor, hint=hint or None,
                       at_frame=at_frame)
        if not vocab:
            return edit
        _check_set(obj, prop, vocab, at_frame)
        from_v = baseline_value_for(edit, vocab)
        if isinstance(from_v, (int, float)) and float(from_v) == 0.0:
            raise ValueError(
                f"{obj}.{prop} has a baseline of 0, which no factor can move "
                f"off. Write it as an absolute edit: SET {obj}.{prop} TO <value>"
            )
        to_v = _scale(from_v, factor)
        spec = vocab.properties[prop]
        _check_kind(prop, to_v, spec.kind)
        return SetEdit(obj, prop, factor, from_v, to_v, hint or None, at_frame)

    m = _SET_TO.match(text)
    if m:
        obj, prop, to_s, hint = m.groups()
        to_v = _parse_value(to_s)
        edit = SetEdit(obj, prop, to_value=to_v, hint=hint or None,
                       at_frame=at_frame)
        if not vocab:
            return edit
        _check_set(obj, prop, vocab, at_frame)
        _check_kind(prop, to_v, vocab.properties[prop].kind)
        return SetEdit(obj, prop, None, baseline_value_for(edit, vocab),
                       to_v, hint or None, at_frame)
    raise ValueError(f"Not a valid edit: {text!r}")


def _check_set(obj: str, prop: str, vocab: Vocabulary,
               at_frame: int | None = None) -> None:
    vocab.check_object(obj)
    vocab.check_property(prop)
    if (obj, prop) not in vocab.sim_bindings:
        raise ValueError(f"Property {prop!r} is not defined on {obj!r}")
    # A launch speed is a starting condition, not a knob the scene carries
    # around: "change the initial speed at frame 30" has nothing to change,
    # the launch already happened.
    if at_frame is not None and prop.startswith("initial_"):
        raise ValueError(
            f"{prop!r} is a starting condition and cannot be timed. It is set "
            "when the body is launched, which is frame 1 by definition."
        )


# ------------------------------------------------------------------ serialize


def _fmt_num(x: float) -> str:
    return f"{x:g}"


def _fmt_value(v: Value) -> str:
    if isinstance(v, tuple):
        return "(" + ", ".join(_fmt_num(c) for c in v) + ")"
    return _fmt_num(v)


def starts_at_frame(edit: Edit) -> int:
    """The first frame the edit is visible on -- 1 for a whole-clip edit."""
    return getattr(edit, "at_frame", None) or 1


def is_timed(edit: Edit) -> bool:
    """Whether the edit lands partway through rather than holding throughout."""
    return getattr(edit, "at_frame", None) is not None


def _timing_suffix(edit: Edit) -> str:
    frame = getattr(edit, "at_frame", None)
    return f" AT FRAME {frame}" if frame is not None else ""


def serialize(edit: Edit) -> str:
    if isinstance(edit, DeleteEdit):
        return f"DELETE {edit.object_id}{_timing_suffix(edit)}"
    if isinstance(edit, AddEdit):
        where = ("MIDPOINT" if edit.is_midpoint
                 else f"{edit.numerator}/{edit.denominator} FROM {edit.anchor_id}")
        parts = [
            f"ADD {edit.object_id}",
            f"BETWEEN {edit.endpoint_a} AND {edit.endpoint_b}",
            f"AT {where}",
        ]
        if edit.hint:
            parts.append(f'HINT "{edit.hint}"')
        return " ".join(parts)
    change = (f"TIMES {_fmt_num(edit.factor)}" if edit.is_relative
              else f"TO {_fmt_value(edit.to_value)}")
    parts = [f"SET {edit.object_id}.{edit.property_name}", change]
    if edit.at_frame is not None:
        parts.append(f"AT FRAME {edit.at_frame}")
    if edit.hint:
        parts.append(f'HINT "{edit.hint}"')
    return " ".join(parts)


# ------------------------------------------------ relative geometry (for ADD)


def _anchor_centre(object_id: str, vocab: Vocabulary) -> Vector:
    anchor = vocab.geometry[object_id]
    value = vocab.baseline_physics[anchor.location_key]
    if anchor.location_index is not None:
        value = value[anchor.location_index]
    if len(value) != 3:
        raise ValueError(
            f"Geometry for {object_id!r} points at {anchor.location_key!r}, "
            f"which is not a 3-vector: {value!r}"
        )
    return (float(value[0]), float(value[1]), float(value[2]))


def _anchor_radius(object_id: str, vocab: Vocabulary) -> float:
    anchor = vocab.geometry[object_id]
    value = vocab.baseline_physics[anchor.radius_key]
    if anchor.radius_index is not None:
        value = value[anchor.radius_index]
    radius = float(value)
    if radius <= 0.0:
        raise ValueError(
            f"Geometry for {object_id!r} reads radius {radius} from "
            f"{anchor.radius_key!r}; a radius has to be positive"
        )
    return radius


def add_fraction(edit: AddEdit, vocab: Vocabulary | None = None) -> float:
    """How far along the endpoint_a -> endpoint_b segment the new body sits.

    0 puts it on top of ``endpoint_a`` and 1 on top of ``endpoint_b``. Since
    the DSL now names a division point directly, this is pure arithmetic on
    ``k/n`` and needs no scene at all; ``vocab`` is accepted and ignored so
    older callers keep working.
    """
    return edit.fraction_from_a


def anchor_centre(object_id: str, vocab: Vocabulary) -> Vector:
    """Public read of an object's baseline centre, in the scene's own frame.

    The evaluator needs the two endpoints of an ADD's line in scene
    coordinates: identifying them by name downstream does not work, because
    the DSL's object ids and the render's mesh names are different vocabularies
    (`red_stone` is `stone_0` in curling's ground truth). Coordinates project
    through the camera without any name matching at all.
    """
    vocab.check_geometry(object_id)
    return _anchor_centre(object_id, vocab)


def _where(edit: AddEdit) -> str:
    """How the edit reads, for error messages."""
    if edit.is_midpoint:
        return "the midpoint"
    return f"{edit.numerator}/{edit.denominator} from {edit.anchor_id}"


def add_location(edit: AddEdit, vocab: Vocabulary) -> Vector:
    """Resolve an ADD's relative position into the scene's own coordinates.

    The new centre sits ``k/n`` of the way from the anchor towards the other
    endpoint. Both endpoints are checked for room: an edit that would seat the
    new body inside one of them is rejected here rather than handed to the
    simulator, which would answer with an explosive overlap resolution instead
    of an error.
    """
    vocab.check_geometry(edit.endpoint_a)
    vocab.check_geometry(edit.endpoint_b)
    vocab.check_geometry(edit.object_id)

    a_centre = _anchor_centre(edit.endpoint_a, vocab)
    b_centre = _anchor_centre(edit.endpoint_b, vocab)
    span = math.dist(a_centre, b_centre)
    if span < 1e-9:
        raise ValueError(
            f"{edit.endpoint_a!r} and {edit.endpoint_b!r} are at the same "
            "place, so there is no line between them"
        )

    r_a = _anchor_radius(edit.endpoint_a, vocab)
    r_b = _anchor_radius(edit.endpoint_b, vocab)
    r_new = _anchor_radius(edit.object_id, vocab)
    t = edit.fraction_from_a

    if t * span < r_a + r_new:
        raise ValueError(
            f"ADD {edit.object_id} at {_where(edit)} puts it {t * span:.4f} m "
            f"from {edit.endpoint_a} centre to centre, but they need at least "
            f"{r_a + r_new:.4f} m not to overlap (the whole line is "
            f"{span:.4f} m)"
        )
    if (1.0 - t) * span < r_b + r_new:
        raise ValueError(
            f"ADD {edit.object_id} at {_where(edit)} leaves only "
            f"{(1.0 - t) * span:.4f} m to {edit.endpoint_b}, which needs at "
            f"least {r_b + r_new:.4f} m (the whole line is {span:.4f} m)"
        )

    return tuple(
        _round(a + t * (b - a)) for a, b in zip(a_centre, b_centre)
    )  # type: ignore[return-value]


def add_position_diff(edit: AddEdit, vocab: Vocabulary) -> dict[str, Any]:
    """The ``position`` block a scene's physics_diff records for an ADD.

    Both halves are kept: the relative statement the edit was written in, and
    the absolute centre it resolves to, so a consumer can score a prediction
    without re-running the resolver. The two endpoints come along in scene
    coordinates because the evaluator projects them to find the line on
    screen -- matching the DSL's object ids against the render's mesh names
    does not work, they are different vocabularies.
    """
    return {
        "on_line_between": [edit.endpoint_a, edit.endpoint_b],
        "at": "midpoint" if edit.is_midpoint
              else f"{edit.numerator}/{edit.denominator}",
        "measured_from": edit.anchor_id,
        "fraction_from_endpoint_a": edit.fraction_from_a,
        "resolved_location": list(add_location(edit, vocab)),
        "endpoint_locations": [
            list(anchor_centre(edit.endpoint_a, vocab)),
            list(anchor_centre(edit.endpoint_b, vocab)),
        ],
    }


# ------------------------------------------------------ physics adapter (generic)


def to_physics_override(edit: Edit, vocab: Vocabulary) -> dict[str, Any]:
    """Return the physics sub-dict to merge onto the scene's baseline."""
    if isinstance(edit, DeleteEdit):
        vocab.check_object(edit.object_id)
        db = vocab.delete_bindings[edit.object_id]
        current = list(vocab.baseline_physics[db.key])
        current[db.index] = db.value_when_removed
        return {db.key: current}

    if isinstance(edit, AddEdit):
        vocab.check_object(edit.object_id)
        ab = vocab.add_bindings[edit.object_id]
        out: dict[str, Any] = dict(ab.defaults)
        presence = list(vocab.baseline_physics[ab.presence_key])
        presence[ab.presence_index] = ab.value_when_present
        out[ab.presence_key] = presence
        location = list(add_location(edit, vocab))
        if ab.location_index is None:
            out[ab.location_key] = location
        else:
            current = list(vocab.baseline_physics[ab.location_key])
            current[ab.location_index] = location
            out[ab.location_key] = current
        return out

    vocab.check_object(edit.object_id)
    vocab.check_property(edit.property_name)
    sb = vocab.sim_bindings[(edit.object_id, edit.property_name)]
    if isinstance(sb, CompoundBinding):
        return _compound_override(sb, edit, vocab)
    to_value = resolved_to_value(edit, vocab)
    if sb.index is None:
        return {sb.key: to_value}
    current = list(vocab.baseline_physics[sb.key])
    current[sb.index] = to_value
    return {sb.key: current}


# The physics key a timed edit's schedule is written under. The renderer
# forwards it to the simulator, which applies each entry's params at the top of
# that frame and carries on from the state the baseline had reached.
TIMED_EDITS_KEY = "timed_edits"


def to_scenario_override(edit: Edit, vocab: Vocabulary) -> dict[str, Any]:
    """The ``physics`` block a case's scenario_overrides.json carries.

    A whole-clip edit is exactly the parameter overrides -- the simulator is
    built with them and runs. A timed edit leaves the parameters at baseline
    and ships a schedule instead, so the run up to the edit frame is the source
    video's own physics, bit for bit, and only then diverges.
    """
    params = to_physics_override(edit, vocab)
    frame = getattr(edit, "at_frame", None)
    if frame is None:
        return params
    return {TIMED_EDITS_KEY: [{"frame": int(frame), "params": params}]}


def timing_diff(edit: Edit, vocab: Vocabulary) -> dict[str, Any]:
    """The ``timing`` block a case's physics_diff records.

    Recorded for every edit, not just the timed ones: a consumer scoring a
    prediction needs to know that frames 1..n-1 are supposed to match the
    source video, and for a whole-clip edit that range is empty rather than
    unspecified.
    """
    return {
        "applies_from_frame": starts_at_frame(edit),
        "total_frames": vocab.total_frames,
        "unedited_frames": (
            [1, starts_at_frame(edit) - 1] if is_timed(edit) else []
        ),
    }


def _read_baseline(component: SimBinding, vocab: Vocabulary) -> Any:
    val = vocab.baseline_physics[component.key]
    if component.index is not None:
        val = val[component.index]
    return val


def _write_override(
    component: SimBinding, new_value: Any, vocab: Vocabulary, out: dict[str, Any]
) -> None:
    if component.index is None:
        out[component.key] = new_value
        return
    current = out.get(component.key)
    if current is None:
        current = list(vocab.baseline_physics[component.key])
    current[component.index] = new_value
    out[component.key] = current


def _compound_override(
    binding: CompoundBinding, edit: SetEdit, vocab: Vocabulary
) -> dict[str, Any]:
    if not binding.components:
        raise ValueError("CompoundBinding requires at least one component")
    display_baseline = float(_read_baseline(binding.components[0], vocab))
    if display_baseline == 0.0:
        raise ValueError(
            f"CompoundBinding display component {binding.components[0].key!r} has "
            "a zero baseline; the scaling ratio is undefined"
        )
    # A compound property is a ratio by construction, which is exactly what
    # the TIMES form already carries; the absolute form has to divide its way
    # back to one.
    ratio = (edit.factor if edit.is_relative
             else float(edit.to_value) / display_baseline)
    out: dict[str, Any] = {}
    for component in binding.components:
        baseline = float(_read_baseline(component, vocab))
        # A zero baseline stays zero: 0 * anything is 0, and this is how a body
        # with no rolling friction of its own remains a pure lateral edit.
        _write_override(component, _round(baseline * ratio), vocab, out)
    return out


def baseline_value_for(edit: Edit, vocab: Vocabulary) -> Any:
    """Baseline value of the property this edit targets (for physics_diff)."""
    if isinstance(edit, DeleteEdit):
        return "present"
    if isinstance(edit, AddEdit):
        return "absent"
    sb = vocab.sim_bindings[(edit.object_id, edit.property_name)]
    if isinstance(sb, CompoundBinding):
        return _read_baseline(sb.components[0], vocab)
    val = vocab.baseline_physics[sb.key]
    if sb.index is not None:
        val = val[sb.index]
    return val


def resolved_to_value(edit: SetEdit, vocab: Vocabulary) -> Any:
    """The absolute value this edit lands on.

    For the ``TO`` form that is what the edit already said; for ``TIMES`` it
    is the baseline scaled by the factor. Parsing with a vocabulary fills
    ``to_value`` in, so this is normally a read -- it recomputes only for an
    edit that was parsed without one.
    """
    if edit.to_value is not None:
        return edit.to_value
    if edit.factor is None:
        raise ValueError(f"{edit!r} carries neither a factor nor a value")
    return _scale(baseline_value_for(edit, vocab), edit.factor)


# ------------------------------------------------------------- NL generation


_VERBS = {
    "increase": {"zh": "调大", "en": "increase"},
    "decrease": {"zh": "调小", "en": "decrease"},
    "change":   {"zh": "改动", "en": "change"},
}

# Chinese counters for the "n 等分点" phrasing. A line is never divided into
# more parts than a viewer could count off by eye, so the table stops where
# counting does.
_ZH_NUM = {2: "二", 3: "三", 4: "四", 5: "五", 6: "六",
           7: "七", 8: "八", 9: "九", 10: "十"}

# English fraction names, for "one quarter of the way along the line".
_EN_DEN = {2: "half", 3: "third", 4: "quarter", 5: "fifth", 6: "sixth",
           7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth"}
_EN_NUM = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
           6: "six", 7: "seven", 8: "eight", 9: "nine"}


def _unit(unit: str) -> str:
    return f" {unit}" if unit else ""


def _poss(name: str) -> str:
    """Possessive of an English display name, plurals included.

    Object names are written the way a caption would read them, and one of
    them ("the two stones") is plural, which takes a bare apostrophe.
    """
    return f"{name}'" if name.endswith("s") else f"{name}'s"


def _factor_text(factor: float, lang: str) -> str:
    """How a multiplier reads in a prompt: as a decimal, always.

    Fractions were tried and dropped. "2/3" and "10/3" are exact but they read
    as arithmetic rather than as an instruction, and mixing them with the
    decimals the other cases need makes one benchmark speak two dialects. Every
    factor in the suite is a value a decimal writes exactly, so the decimal is
    the whole vocabulary.
    """
    return _fmt_num(factor)


def _direction(edit: SetEdit, vocab: Vocabulary) -> str:
    """Which way the edit moves the property, for the vague prompt."""
    def mag(v: Any) -> float:
        if isinstance(v, (list, tuple)):
            return math.sqrt(sum(float(c) * float(c) for c in v))
        return abs(float(v))

    if edit.is_relative:
        if edit.factor == 0.0:
            return "deactivate"
        return "increase" if edit.factor > 1.0 else (
            "decrease" if edit.factor < 1.0 else "change")
    a = mag(edit.from_value if edit.from_value is not None
            else baseline_value_for(edit, vocab))
    b = mag(edit.to_value)
    if a < 1e-12 <= b:
        return "activate"
    if b < 1e-12 <= a:
        return "deactivate"
    if b > a:
        return "increase"
    if b < a:
        return "decrease"
    return "change"


def _render_add(edit: AddEdit, vocab: Vocabulary, lang: str, precise: bool) -> str:
    vocab.check_object(edit.object_id)
    new = vocab.objects[edit.object_id][lang]
    a = vocab.objects[edit.endpoint_a][lang]
    b = vocab.objects[edit.endpoint_b][lang]
    hint = edit.hint or ""

    if precise:
        text = _render_add_precise(edit, vocab, lang, new, a, b)
        if hint:
            text = text[:-1] + {"zh": f"({hint})。", "en": f" ({hint})."}[lang]
        return text

    # Vague: keep the position relative but drop the division point. A third
    # of the way is the cutoff -- nearer than that and the eye reads the new
    # body as sitting beside one of the endpoints rather than between them.
    t = edit.fraction_from_a
    # Inclusive at a third: the 1/3 point is one a viewer reads as sitting
    # nearer that end, not as the middle.
    if t <= 1.0 / 3.0 + 1e-9:
        where = {"zh": f"靠近{a}的地方", "en": f"closer to {a}"}[lang]
    elif t >= 2.0 / 3.0 - 1e-9:
        where = {"zh": f"靠近{b}的地方", "en": f"closer to {b}"}[lang]
    else:
        where = {"zh": "大致中间的位置", "en": "about midway between them"}[lang]
    if lang == "zh":
        text = f"在{a}和{b}的连线上、{where},放一个{new}。"
    else:
        text = f"Place {new} on the line between {a} and {b}, {where}."
    if hint:
        text = text[:-1] + {"zh": f"({hint})。", "en": f" ({hint})."}[lang]
    return text


def _render_add_precise(
    edit: AddEdit, vocab: Vocabulary, lang: str, new: str, a: str, b: str
) -> str:
    """The division point in words.

    ``k/n`` past halfway is flipped to ``(n-k)/n`` from the other endpoint
    before it is spoken: "the quarter point nearest the target ball" is a
    thing to look for, "the third of four counting from the cue ball" is
    arithmetic. Which endpoint the DSL measured from does not survive that
    flip, and does not need to -- both name the same point on the same line.
    """
    if edit.is_midpoint or edit.numerator * 2 == edit.denominator:
        if lang == "zh":
            return f"在{a}和{b}连线的中点放一个{new}。"
        return f"Place {new} at the midpoint of the line between {a} and {b}."

    n = edit.denominator
    k = edit.numerator
    anchor_id = edit.anchor_id
    if k * 2 > n:                      # past halfway: measure from the far end
        k = n - k
        anchor_id = edit.other_id
    anchor = vocab.objects[anchor_id][lang]

    if lang == "zh":
        parts = _ZH_NUM.get(n, str(n))
        if k == 1:
            return f"在{a}和{b}连线的{parts}等分点上放一个{new},取靠近{anchor}的那个。"
        return (f"在{a}和{b}连线的{parts}等分点上放一个{new},"
                f"取从{anchor}数起的第{_ZH_NUM.get(k, str(k))}个。")

    if k == 1:
        frac = f"one {_EN_DEN.get(n, f'{n}th')}"
    else:
        frac = f"{_EN_NUM.get(k, str(k))} {_EN_DEN.get(n, f'{n}th')}s"
    other = vocab.objects[edit.endpoint_b if anchor_id == edit.endpoint_a
                          else edit.endpoint_a][lang]
    return (f"Place {new} on the line between {a} and {b}, {frac} of the way "
            f"from {anchor} to {other}.")


def _render_set(edit: SetEdit, vocab: Vocabulary, lang: str, precise: bool) -> str:
    vocab.check_object(edit.object_id)
    vocab.check_property(edit.property_name)
    spec = vocab.properties[edit.property_name]
    obj = vocab.objects[edit.object_id][lang]
    prop = spec.zh if lang == "zh" else spec.en
    hint = edit.hint or ""
    dir_ = _direction(edit, vocab)

    if precise:
        if not edit.is_relative:
            # The absolute escape hatch: a property whose baseline is 0 has no
            # multiple to name, so the prompt says the number it does have.
            unit = _unit(spec.unit)
            fv = _fmt_value(edit.from_value if edit.from_value is not None
                            else baseline_value_for(edit, vocab)) + unit
            tv = _fmt_value(edit.to_value) + unit
            if lang == "zh":
                return f"把{obj}的{prop}从 {fv} 改成 {tv}。"
            return f"Change {_poss(obj)} {prop} from {fv} to {tv}."
        if dir_ == "deactivate":
            return _zero_text(spec, obj, prop, lang)
        k = _factor_text(edit.factor, lang)
        if lang == "zh":
            verb = "增加" if edit.factor > 1.0 else "减少"
            return f"把{obj}的{prop}{verb}到原来的 {k} 倍。"
        verb = "Increase" if edit.factor > 1.0 else "Reduce"
        return f"{verb} {_poss(obj)} {prop} to {k} times its original value."

    if dir_ == "activate":
        hint_zh = f",方向{hint}" if hint else ""
        hint_en = f" ({hint})" if hint else ""
        article = "an" if prop[:1].lower() in "aeiou" else "a"
        return {"zh": f"给{obj}一个{prop}{hint_zh}。",
                "en": f"Give {obj} {article} {prop}{hint_en}."}[lang]
    if dir_ == "deactivate":
        return _zero_text(spec, obj, prop, lang)
    verb = _VERBS[dir_][lang]
    if lang == "zh":
        return f"把{obj}的{prop}{verb}一些。"
    return f"{verb.capitalize()} {_poss(obj)} {prop}."


def _zero_text(spec: PropertySpec, obj: str, prop: str, lang: str) -> str:
    """Taking a property to zero, said the way the scene reads.

    A speed of zero is not "a zeroed initial speed", it is a body that does
    not move, so a velocity property gets its own wording; everything else
    keeps the generic one.
    """
    if spec.unit == "m/s":
        return {"zh": f"让{obj}保持静止。",
                "en": f"Leave {obj} at rest."}[lang]
    return {"zh": f"把{obj}的{prop}清零。",
            "en": f"Set {_poss(obj)} {prop} to zero."}[lang]


def _when_text(edit: Edit, vocab: Vocabulary, lang: str, precise: bool) -> str:
    """When the edit takes effect, as the clause a prompt opens with.

    Every prompt carries one, whole-clip edits included: "increase the mass"
    and "increase the mass from frame 41 on" are different instructions, and
    the benchmark should never leave a reader to guess which one it meant.

    The precise flavour names the frame, which is a number a viewer can scrub
    to. The vague flavour places it in the clip the way an eye would -- early,
    about halfway, late -- on the same thirds the vague ADD prompt uses.
    """
    frame = getattr(edit, "at_frame", None)
    if frame is None:
        if precise:
            return {"zh": "从第 1 帧开始", "en": "From frame 1 onwards"}[lang]
        return {"zh": "从视频一开始", "en": "From the start of the clip"}[lang]
    if precise:
        return {"zh": f"从第 {frame} 帧开始",
                "en": f"From frame {frame} onwards"}[lang]
    total = vocab.total_frames
    if not total:
        raise ValueError(
            "A timed edit needs the scene's total_frames to say where in the "
            "clip it lands"
        )
    t = (frame - 1) / float(total - 1) if total > 1 else 0.0
    if t <= 1.0 / 3.0 + 1e-9:
        return {"zh": "在视频前段", "en": "Early in the clip"}[lang]
    if t >= 2.0 / 3.0 - 1e-9:
        return {"zh": "在视频后段", "en": "Late in the clip"}[lang]
    return {"zh": "在视频进行到一半左右时",
            "en": "About halfway through the clip"}[lang]


def _with_when(text: str, edit: Edit, vocab: Vocabulary, lang: str,
               precise: bool) -> str:
    when = _when_text(edit, vocab, lang, precise)
    if lang == "zh":
        return f"{when},{text}"
    # Every generated English prompt opens with its verb, so the sentence
    # joins onto the clause by lowercasing that one letter.
    return f"{when}, {text[:1].lower()}{text[1:]}"


def render(edit: Edit, vocab: Vocabulary, *, lang: str = "zh", precise: bool) -> str:
    if isinstance(edit, DeleteEdit):
        vocab.check_object(edit.object_id)
        name = vocab.objects[edit.object_id][lang]
        text = {"zh": f"把{name}从场景中删除。",
                "en": f"Remove {name} from the scene."}[lang]
    elif isinstance(edit, AddEdit):
        text = _render_add(edit, vocab, lang, precise)
    else:
        text = _render_set(edit, vocab, lang, precise)
    return _with_when(text, edit, vocab, lang, precise)


def editable_objects(vocab: Vocabulary) -> list[dict[str, Any]]:
    """What a viewer of the source video is allowed to edit, and how.

    An object that only ADD can bring in is not in the source video at all, so
    it is reported with ``in_video`` false: naming it in a description of the
    baseline would be describing something that is not on screen.
    """
    out: list[dict[str, Any]] = []
    for object_id, names in vocab.objects.items():
        props = sorted(p for (o, p) in vocab.sim_bindings if o == object_id)
        out.append({
            "object_id": object_id,
            "zh": names["zh"],
            "en": names["en"],
            "in_video": object_id not in vocab.add_bindings,
            "properties": props,
            "can_delete": object_id in vocab.delete_bindings,
            "can_add": object_id in vocab.add_bindings,
        })
    return out


def _join(items: list[str], lang: str) -> str:
    if len(items) == 1:
        return items[0]
    if lang == "zh":
        return "、".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def render_roster(vocab: Vocabulary, lang: str = "zh", *,
                  detailed: bool = True) -> str:
    """One sentence naming the editable objects in the source video.

    Only objects that are actually on screen are named. An object the scene can
    ADD is not in the source video, and listing it here would be describing the
    clip by what it might become rather than by what it shows.

    ``detailed`` chooses which of the two source prompts this is for. The
    quantitative one names each object's editable properties -- per object,
    because not every body carries every property, and implying otherwise would
    invite edits the vocabulary rejects. The vague one names the objects only.
    """
    present = []
    for obj in editable_objects(vocab):
        if not obj["in_video"]:
            continue
        name = obj[lang]
        if not detailed:
            present.append(name)
            continue
        props = [(vocab.properties[p].zh if lang == "zh" else vocab.properties[p].en)
                 for p in obj["properties"]]
        if lang == "zh":
            if props and obj["can_delete"]:
                present.append(f"{name}(可改{_join(props, lang)},可删除)")
            elif props:
                present.append(f"{name}(可改{_join(props, lang)})")
            else:
                present.append(f"{name}(仅可删除)" if obj["can_delete"] else name)
        else:
            if props and obj["can_delete"]:
                present.append(f"{name} ({_join(props, lang)}, removable)")
            elif props:
                present.append(f"{name} ({_join(props, lang)})")
            else:
                present.append(f"{name} (removable)" if obj["can_delete"] else name)

    if lang == "zh":
        return "画面里可编辑的物体:" + _join(present, lang) + "。"
    return "Editable objects in this video: " + _join(present, lang) + "."


def make_prompts(edit: Edit, vocab: Vocabulary) -> dict[str, dict[str, str]]:
    """Convenience: precise + vague, in zh + en."""
    return {
        "vague":        {"zh": render(edit, vocab, lang="zh", precise=False),
                         "en": render(edit, vocab, lang="en", precise=False)},
        "quantitative": {"zh": render(edit, vocab, lang="zh", precise=True),
                         "en": render(edit, vocab, lang="en", precise=True)},
    }
