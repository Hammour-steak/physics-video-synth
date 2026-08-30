"""Scene-agnostic edit DSL for PCVE physics-edit datasets.

Grammar
-------
    Edit       := DeleteStmt | SetStmt
    DeleteStmt := "DELETE" ObjectId
    SetStmt    := "SET" ObjectId "." Property
                  "FROM" Value "TO" Value ["HINT" QUOTED_STRING]
    Value      := Scalar | "(" Scalar "," Scalar "," Scalar ")"

Two operations only: delete an object, or set a scalar/vector property from
its baseline value to a new value.

Per-scene setup
---------------
Each scene declares a ``Vocabulary`` that says:
  * which object ids exist and their zh/en display names;
  * which properties exist (name, kind, unit, zh/en);
  * how each (object, property) pair maps to the scene's sim parameter dict
    (``sim_bindings``), and how DELETE maps to the sim parameter dict
    (``delete_bindings``);
  * the baseline physics dict, used to look up "from" values when generating
    physics_diff and to preserve neighbouring slots when writing into a list.

Once the vocabulary is defined, the whole pipeline is data-driven:
    parse(dsl_str, vocab)                -> Edit
    render(edit, vocab, lang, precise)   -> natural-language prompt
    to_physics_override(edit, vocab)     -> {sim_param: new_value, ...}
    baseline_value_for(edit, vocab)      -> baseline value (for physics_diff)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Union

Scalar = float
Vector = tuple[float, float, float]
Value = Union[Scalar, Vector]


# ---------------------------------------------------------------- data types


@dataclass(frozen=True)
class DeleteEdit:
    object_id: str


@dataclass(frozen=True)
class SetEdit:
    object_id: str
    property_name: str
    from_value: Value
    to_value: Value
    hint: str | None = None


Edit = Union[DeleteEdit, SetEdit]


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
class Vocabulary:
    objects: dict[str, dict[str, str]]                    # obj_id -> {zh, en}
    properties: dict[str, PropertySpec]                   # prop_name -> spec
    sim_bindings: dict[tuple[str, str], SimBinding | CompoundBinding]  # (obj, prop) -> where
    delete_bindings: dict[str, DeleteBinding]             # obj -> where
    baseline_physics: dict[str, Any]

    def check_object(self, name: str) -> None:
        if name not in self.objects:
            raise ValueError(f"Unknown object: {name!r}. Allowed: {sorted(self.objects)}")

    def check_property(self, name: str) -> None:
        if name not in self.properties:
            raise ValueError(f"Unknown property: {name!r}. Allowed: {sorted(self.properties)}")


# --------------------------------------------------------------------- parse

_VEC = re.compile(r"\(\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\)")
_SET = re.compile(
    r"^SET\s+(\w+)\.(\w+)\s+FROM\s+(\S.*?)\s+TO\s+(\S.*?)"
    r"(?:\s+HINT\s+\"([^\"]*)\")?\s*$"
)
_DEL = re.compile(r"^DELETE\s+(\w+)\s*$")


def _parse_value(s: str) -> Value:
    s = s.strip()
    m = _VEC.fullmatch(s)
    if m:
        return (float(m.group(1)), float(m.group(2)), float(m.group(3)))
    return float(s)


def _check_kind(prop: str, v: Value, kind: str) -> None:
    if kind == "scalar" and not isinstance(v, float):
        raise ValueError(f"{prop!r} expects scalar, got {v!r}")
    if kind == "vector" and not (isinstance(v, tuple) and len(v) == 3):
        raise ValueError(f"{prop!r} expects 3-vector, got {v!r}")


def parse(text: str, vocab: Vocabulary | None = None) -> Edit:
    text = text.strip()
    m = _DEL.match(text)
    if m:
        obj = m.group(1)
        if vocab:
            vocab.check_object(obj)
            if obj not in vocab.delete_bindings:
                raise ValueError(f"Object {obj!r} cannot be deleted in this scene")
        return DeleteEdit(object_id=obj)
    m = _SET.match(text)
    if m:
        obj, prop, from_s, to_s, hint = m.groups()
        from_v, to_v = _parse_value(from_s), _parse_value(to_s)
        if vocab:
            vocab.check_object(obj)
            vocab.check_property(prop)
            spec = vocab.properties[prop]
            _check_kind(prop, from_v, spec.kind)
            _check_kind(prop, to_v, spec.kind)
            if (obj, prop) not in vocab.sim_bindings:
                raise ValueError(f"Property {prop!r} is not defined on {obj!r}")
        return SetEdit(obj, prop, from_v, to_v, hint or None)
    raise ValueError(f"Not a valid edit: {text!r}")


# ------------------------------------------------------------------ serialize


def _fmt_num(x: float) -> str:
    return f"{x:g}"


def _fmt_value(v: Value) -> str:
    if isinstance(v, tuple):
        return "(" + ", ".join(_fmt_num(c) for c in v) + ")"
    return _fmt_num(v)


def serialize(edit: Edit) -> str:
    if isinstance(edit, DeleteEdit):
        return f"DELETE {edit.object_id}"
    parts = [
        f"SET {edit.object_id}.{edit.property_name}",
        f"FROM {_fmt_value(edit.from_value)}",
        f"TO {_fmt_value(edit.to_value)}",
    ]
    if edit.hint:
        parts.append(f'HINT "{edit.hint}"')
    return " ".join(parts)


# ------------------------------------------------------ physics adapter (generic)


def to_physics_override(edit: Edit, vocab: Vocabulary) -> dict[str, Any]:
    """Return the physics sub-dict to merge onto the scene's baseline."""
    if isinstance(edit, DeleteEdit):
        vocab.check_object(edit.object_id)
        db = vocab.delete_bindings[edit.object_id]
        current = list(vocab.baseline_physics[db.key])
        current[db.index] = db.value_when_removed
        return {db.key: current}

    vocab.check_object(edit.object_id)
    vocab.check_property(edit.property_name)
    sb = vocab.sim_bindings[(edit.object_id, edit.property_name)]
    if isinstance(sb, CompoundBinding):
        return _compound_override(sb, edit, vocab)
    if sb.index is None:
        return {sb.key: edit.to_value}
    current = list(vocab.baseline_physics[sb.key])
    current[sb.index] = edit.to_value
    return {sb.key: current}


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
    from_v = float(edit.from_value)
    to_v = float(edit.to_value)
    if from_v == 0.0:
        raise ValueError(
            f"Edit on {edit.object_id}.{edit.property_name} has FROM 0; cannot "
            "form a scaling ratio"
        )
    ratio = to_v / from_v
    out: dict[str, Any] = {}
    for component in binding.components:
        baseline = float(_read_baseline(component, vocab))
        # A zero baseline stays zero: 0 * anything is 0, and this is how a body
        # with no rolling friction of its own remains a pure lateral edit.
        _write_override(component, baseline * ratio, vocab, out)
    return out


def baseline_value_for(edit: Edit, vocab: Vocabulary) -> Any:
    """Baseline value of the property this edit targets (for physics_diff)."""
    if isinstance(edit, DeleteEdit):
        return "present"
    sb = vocab.sim_bindings[(edit.object_id, edit.property_name)]
    if isinstance(sb, CompoundBinding):
        return _read_baseline(sb.components[0], vocab)
    val = vocab.baseline_physics[sb.key]
    if sb.index is not None:
        val = val[sb.index]
    return val


# ------------------------------------------------------------- NL generation


def _direction(from_v: Value, to_v: Value) -> str:
    def mag(v: Value) -> float:
        return math.sqrt(sum(c * c for c in v)) if isinstance(v, tuple) else abs(v)
    a, b = mag(from_v), mag(to_v)
    if a < 1e-12 <= b:
        return "activate"
    if b < 1e-12 <= a:
        return "deactivate"
    if b > a:
        return "increase"
    if b < a:
        return "decrease"
    return "change"


_VERBS = {
    "increase": {"zh": "调大", "en": "increase"},
    "decrease": {"zh": "调小", "en": "decrease"},
    "change":   {"zh": "改动", "en": "change"},
}


def _unit(unit: str) -> str:
    return f" {unit}" if unit else ""


def render(edit: Edit, vocab: Vocabulary, *, lang: str = "zh", precise: bool) -> str:
    if isinstance(edit, DeleteEdit):
        vocab.check_object(edit.object_id)
        name = vocab.objects[edit.object_id][lang]
        return {"zh": f"把{name}从场景中删除。",
                "en": f"Remove {name} from the scene."}[lang]

    vocab.check_object(edit.object_id)
    vocab.check_property(edit.property_name)
    spec = vocab.properties[edit.property_name]
    obj = vocab.objects[edit.object_id][lang]
    prop = spec.zh if lang == "zh" else spec.en
    unit = _unit(spec.unit)
    hint = edit.hint or ""

    if precise:
        fv = _fmt_value(edit.from_value) + unit
        tv = _fmt_value(edit.to_value) + unit
        if lang == "zh":
            return f"把{obj}的{prop}从 {fv} 改成 {tv}。"
        return f"Change {obj}'s {prop} from {fv} to {tv}."

    dir_ = _direction(edit.from_value, edit.to_value)
    if dir_ == "activate":
        hint_zh = f",方向{hint}" if hint else ""
        hint_en = f" ({hint})" if hint else ""
        return {"zh": f"给{obj}一个{prop}{hint_zh}。",
                "en": f"Give {obj} an {prop}{hint_en}."}[lang]
    if dir_ == "deactivate":
        return {"zh": f"把{obj}的{prop}清零。",
                "en": f"Set {obj}'s {prop} to zero."}[lang]
    verb = _VERBS[dir_][lang]
    if lang == "zh":
        return f"把{obj}的{prop}{verb}一些。"
    return f"{verb.capitalize()} {obj}'s {prop}."


def make_prompts(edit: Edit, vocab: Vocabulary) -> dict[str, dict[str, str]]:
    """Convenience: precise + vague, in zh + en."""
    return {
        "vague":        {"zh": render(edit, vocab, lang="zh", precise=False),
                         "en": render(edit, vocab, lang="en", precise=False)},
        "quantitative": {"zh": render(edit, vocab, lang="zh", precise=True),
                         "en": render(edit, vocab, lang="en", precise=True)},
    }
