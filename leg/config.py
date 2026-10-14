"""Configuration: dataclasses, YAML loading and validation, and preset write-back.

``leg_config.yaml`` is the single source of truth for geometry, masses, sections, actuators, gaits and presets.
Everything is converted to SI here (m, kg, N, N*m, rad, Pa).  Keys that carry degrees say so in their name
(``limits_deg``, ``q_deg``, ``rpy_deg``).

Structure of the file (see the comments in ``leg_config.yaml`` for the full field list):

    robot / gravity / n_stations / *_support_fraction   scalars
    joints                ordered pelvis -> foot; each joint carries the link distal to it
    foot                  contact point, sole rectangle, beam end
    actuators             ratings (output side), reflected rotor inertia
    actuator_options      alternative actuator models for named groups (e.g. the ankle motors)
    transmissions         joint space -> actuator space, one group per coupled set of joints
    gaits                 keyframed stance phases (walking, running)
    presets               named poses + load cases (MUST remain the last section: the UI appends to it)

Validation collects *all* problems and raises one ``ConfigError`` whose message lists each offending YAML path, so a
file with several typos is fixed in one pass.  Values tagged ``# PLACEHOLDER`` on their line are counted so the UI
can show a banner while any remain (``find_placeholders``).
"""
from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


class ConfigError(Exception):
    """Raised with a readable, multi-line message when the config is invalid."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("Invalid leg config:\n  - " + "\n  - ".join(errors))


# =========================================================================== data model


@dataclass
class PointMass:
    """A lumped mass (actuator body, ankle rod, ...) fixed in a link frame."""
    name: str
    mass: float
    pos: np.ndarray            # in the link frame [m]
    kind: str = "actuator"     # "actuator" | "structure" | "linkage": only changes how it is drawn


@dataclass
class Section:
    """Cross-section properties in the *beam frame* (x along the beam, y from ``y_ref``, z = x cross y)."""
    kind: str
    A: float                   # area [m^2]
    Iy: float                  # second moment about beam-y (resists bending in the z direction) [m^4]
    Iz: float                  # second moment about beam-z (resists bending in the y direction) [m^4]
    J: float                   # torsion constant [m^4]  (reported with the loads, NOT used in the stress)
    cy: float                  # extreme-fibre distance along beam-y [m]
    cz: float                  # extreme-fibre distance along beam-z [m]
    yield_strength: float      # [Pa]
    modulus: float             # [Pa]
    kt: float = 1.0            # stress-concentration factor applied to the total stress
    y_ref: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))   # beam-y, in the link frame


@dataclass
class LinkSpec:
    """The rigid body distal to a joint.  Its frame is the joint frame (after the joint rotation)."""
    name: str
    mass: float                # structure mass at ``com`` [kg]; actuators are separate ``point_masses``
    com: np.ndarray            # structure COM in the link frame (``com: center`` = half-way along the beam)
    point_masses: list[PointMass]
    inertia: np.ndarray | None  # 3x3 about the COM in the link frame; None -> thin rod (see dynamics.py)
    section: Section | None    # first segment's section (its y_ref orients the beam frame)
    two_force: bool            # report axial load only and flag any transverse load as a design error
    segments: list = field(default_factory=list)   # [(from_m, to_m, Section)], contiguous, covering the beam
    beam: tuple | None = None  # optional (start, end) in the link frame; default: joint origin -> next joint origin


@dataclass
class JointSpec:
    name: str
    axis: np.ndarray           # unit vector in the joint frame (identical before and after the rotation)
    origin_xyz: np.ndarray     # fixed offset parent link frame -> joint frame [m]
    origin_R: np.ndarray       # fixed rotation parent link frame -> joint frame
    limits: tuple[float, float]  # rad
    link: LinkSpec


@dataclass
class FootSpec:
    contact_point: np.ndarray  # sole point under the ankle, in the last link frame [m]
    beam_end: np.ndarray       # end of the last link's structural beam, last link frame [m]
    toe_x: float               # sole extent along the foot x axis [m] (heel_x < 0 < toe_x)
    heel_x: float
    half_width: float          # sole half-width along y [m]
    thickness: float = 0.02    # plate thickness measured up from the sole [m] (drawing only)


@dataclass
class ActuatorSpec:
    """Ratings are OUTPUT-side (after any integrated gearbox), so the matching transmission ratio is 1."""
    name: str
    kind: str                  # "rotary" (N*m, rad/s) or "linear" (N, m/s)
    continuous: float          # rated torque / force
    peak: float
    max_speed: float           # no-load speed
    rotor_inertia: float       # kg m^2 referred to the OUTPUT shaft (N^2 x datasheet inertia); kg for linear
    model: str = ""            # e.g. "X8-150" (labels only)
    rated_speed: float = 0.0   # speed at rated torque, for the torque-speed envelope; 0 -> 0.8 x max_speed


@dataclass
class TransmissionSpec:
    """Maps a group of joints to a group of actuators.  ``qdot_act = T(q) qdot_joint`` (see actuators.py)."""
    joints: list[int]
    actuators: list[int]
    kind: str                  # ratio | matrix | table | function
    ratio: float | None = None
    matrix: np.ndarray | None = None
    table_q: np.ndarray | None = None
    table_ratio: np.ndarray | None = None
    function: str | None = None    # "package.module:function"
    params: dict = field(default_factory=dict)   # keyword arguments for ``function``


@dataclass
class ActuatorOption:
    """Alternative actuator models for a named group (e.g. the two ankle motors), selectable in the UI.

    ``choices[<model>]`` holds continuous, peak, max_speed, rated_speed, rotor_inertia, mass and note.  The
    ``actuators:`` entries in the YAML must equal the ``default`` choice (checked on load)."""
    key: str
    label: str
    applies_to: list[str]      # actuator names whose ratings are replaced
    mass_points: list[str]     # lumped-mass names (in any link) whose mass is replaced
    default: str
    choices: dict


@dataclass
class Preset:
    """A named pose + load case.  ``load_fraction`` x robot weight is applied vertically at the contact point."""
    name: str
    q: np.ndarray              # rad
    load_fraction: float
    support: str               # "single" | "double": only selects the default fraction in the UI
    motion: dict | None = None  # optional {type: gait, gait: <name>, phase: <0..1>}  (see gait.py)
    description: str = ""


@dataclass
class GaitKey:
    phase: float               # 0 = foot strike .. 1 = toe-off (fraction of the stance duration)
    name: str
    q: np.ndarray              # rad, joint order
    force_bw: np.ndarray       # ground reaction on the foot in body weights: [fore-aft (+ fwd), lateral (+ left), vertical]
    cop_x: float               # centre of pressure ahead of the ankle along the foot [m]


@dataclass
class Gait:
    """One stance phase as keyframes over phase 0..1.  Joint speeds come from the angles / ``stance_time``."""
    keys: list
    description: str
    stance_time: float         # s, foot-down duration
    stance_fraction: float     # share of the whole gait cycle spent in stance (only for the cycle-RMS estimate)
    name: str = ""


@dataclass
class Config:
    robot_total_mass: float            # kg: sets "fraction x robot weight"
    gravity: float
    n_stations: int                    # evenly spaced load-diagram stations per link
    joints: list[JointSpec]
    foot: FootSpec
    actuators: list[ActuatorSpec]
    transmissions: list[TransmissionSpec]
    presets: dict[str, Preset]
    double_support_fraction: float     # default load fraction for the UI's double-support toggle
    single_support_fraction: float
    path: Path | None = None
    placeholder_lines: list[tuple[int, str]] = field(default_factory=list)
    gaits: dict = field(default_factory=dict)              # name -> Gait
    actuator_options: dict = field(default_factory=dict)   # key -> ActuatorOption
    active_choices: dict = field(default_factory=dict)     # key -> selected model (after ``with_actuator_choices``)

    @property
    def n(self) -> int:
        return len(self.joints)

    @property
    def joint_names(self) -> list[str]:
        return [j.name for j in self.joints]


# =========================================================================== validation helpers


class _Check:
    """Accumulates readable errors keyed by YAML path and offers typed accessors."""

    def __init__(self) -> None:
        self.errors: list[str] = []

    def err(self, path: str, msg: str) -> None:
        self.errors.append(f"{path}: {msg}")

    def get(self, d: Any, key: str, path: str, required: bool = True, default: Any = None):
        """``d[key]`` with a readable error if ``d`` is not a mapping or the key is missing."""
        if not isinstance(d, dict):
            self.err(path, "expected a mapping")
            return default
        if key not in d:
            if required:
                self.err(f"{path}.{key}" if path else key, "missing required key")
            return default
        return d[key]

    def num(self, v: Any, path: str, lo: float | None = None, hi: float | None = None,
            allow_none: bool = False) -> float:
        """A finite number within [lo, hi]."""
        if v is None and allow_none:
            return math.nan
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            self.err(path, f"expected a finite number, got {v!r}")
            return math.nan
        if lo is not None and v < lo:
            self.err(path, f"must be >= {lo}, got {v}")
        if hi is not None and v > hi:
            self.err(path, f"must be <= {hi}, got {v}")
        return float(v)

    def vec(self, v: Any, n: int, path: str) -> np.ndarray:
        """A list of ``n`` finite numbers as an array."""
        if (not isinstance(v, (list, tuple)) or len(v) != n
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                       for x in v)):
            self.err(path, f"expected a list of {n} finite numbers, got {v!r}")
            return np.zeros(n)
        return np.array(v, dtype=float)


def _rpy_to_R(rpy: np.ndarray) -> np.ndarray:
    """Rotation matrix Rz(yaw) Ry(pitch) Rx(roll) from roll-pitch-yaw [rad]."""
    r, p, y = rpy
    Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# =========================================================================== sections and links


def _section(c: _Check, s: Any, path: str) -> Section | None:
    """Build a ``Section`` from its YAML mapping.  Kinds and their dimensions (all in the beam frame):

    tube            outer_diameter, wall
    rect_tube       width (y), height (z), wall  -or-  wall_y (walls normal to y) + wall_z
    parallel_plates thickness, depth (z), gap (clear space between the plates, along y)
    i_beam          flange_width, flange_thickness, web_thickness, depth (z)
    rect            width (y), height (z)         solid rectangle
    generic         area, Iy, Iz, J, c_y, c_z

    plus yield_strength, modulus, optional stress_concentration (Kt) and y_ref.  Hollow / open-section torsion
    constants are thin-wall approximations; ``parallel_plates`` uses the open-section value (conservative) and treats
    the plate pair as acting compositely in bending across the gap (parallel-axis term), which is right for a pure
    moment (an axial couple in the plates) but not for lateral shear.
    """
    if s is None:
        return None
    kind = c.get(s, "kind", path)
    mat_y = c.num(c.get(s, "yield_strength", path), f"{path}.yield_strength", lo=1.0)
    mat_E = c.num(c.get(s, "modulus", path), f"{path}.modulus", lo=1.0)
    kt = c.num(s.get("stress_concentration", 1.0), f"{path}.stress_concentration", lo=1.0)
    y_ref = c.vec(s.get("y_ref", [0, 1, 0]), 3, f"{path}.y_ref")
    common = dict(yield_strength=mat_y, modulus=mat_E, kt=kt, y_ref=y_ref)
    P = lambda k, **kw: c.num(c.get(s, k, path), f"{path}.{k}", **kw)  # noqa: E731
    if kind == "tube":
        D, t = P("outer_diameter", lo=1e-6), P("wall", lo=1e-6)
        if not (t < D / 2):
            c.err(path, "wall must be < outer_diameter/2")
            return None
        d = D - 2 * t
        I = math.pi / 64 * (D**4 - d**4)
        return Section("tube", math.pi / 4 * (D**2 - d**2), I, I, 2 * I, D / 2, D / 2, **common)
    if kind == "rect_tube":
        b, h = P("width", lo=1e-6), P("height", lo=1e-6)
        if "wall" in s:
            ty = tz = P("wall", lo=1e-6)
        else:
            ty, tz = P("wall_y", lo=1e-6), P("wall_z", lo=1e-6)
        if not (2 * ty < b and 2 * tz < h):
            c.err(path, "walls must be thinner than half the width / height")
            return None
        bi, hi = b - 2 * ty, h - 2 * tz
        Am = (b - ty) * (h - tz)
        J = 4 * Am**2 / (2 * (b - ty) / tz + 2 * (h - tz) / ty)      # Bredt thin-wall closed section
        return Section("rect_tube", b * h - bi * hi, (b * h**3 - bi * hi**3) / 12,
                       (h * b**3 - hi * bi**3) / 12, J, b / 2, h / 2, **common)
    if kind == "parallel_plates":
        t, w, g = P("thickness", lo=1e-6), P("depth", lo=1e-6), P("gap", lo=0.0)
        d = g / 2 + t / 2                                   # plate centre-line offset from the mid-plane
        return Section("parallel_plates", 2 * t * w, 2 * t * w**3 / 12, 2 * (w * t**3 / 12 + t * w * d**2),
                       2 * w * t**3 / 3, g / 2 + t, w / 2, **common)
    if kind == "i_beam":
        b, tf, tw, h = P("flange_width", lo=1e-6), P("flange_thickness", lo=1e-6), P("web_thickness", lo=1e-6), P("depth", lo=1e-6)
        if not (2 * tf < h and tw < b):
            c.err(path, "need 2*flange_thickness < depth and web_thickness < flange_width")
            return None
        hw = h - 2 * tf
        return Section("i_beam", 2 * b * tf + hw * tw, (b * h**3 - (b - tw) * hw**3) / 12,
                       (2 * tf * b**3 + hw * tw**3) / 12, (2 * b * tf**3 + hw * tw**3) / 3, b / 2, h / 2, **common)
    if kind == "rect":
        b, h = P("width", lo=1e-6), P("height", lo=1e-6)
        a_, c_ = max(b, h), min(b, h)
        J = a_ * c_**3 * (1 / 3 - 0.21 * (c_ / a_) * (1 - c_**4 / (12 * a_**4)))   # standard rectangle torsion constant
        return Section("rect", b * h, b * h**3 / 12, h * b**3 / 12, J, b / 2, h / 2, **common)
    if kind == "generic":
        return Section("generic", P("area", lo=1e-12), P("Iy", lo=1e-18), P("Iz", lo=1e-18),
                       P("J", lo=1e-18), P("c_y", lo=1e-9), P("c_z", lo=1e-9), **common)
    c.err(f"{path}.kind", f"unknown section kind {kind!r} (use tube | rect_tube | parallel_plates | i_beam | rect | generic)")
    return None


def _link(c: _Check, d: Any, path: str) -> LinkSpec:
    """Parse one link: masses, optional inertia, optional beam override, and a section or list of section segments."""
    name = str(c.get(d, "name", path, default="?"))
    mass = c.num(c.get(d, "mass", path), f"{path}.mass", lo=0.0)
    raw_com = d.get("com") if isinstance(d, dict) else None
    # "center" = half-way along the link's beam; resolved in ``_resolve_links`` once the next joint is known
    com = np.full(3, np.nan) if raw_com == "center" else c.vec(c.get(d, "com", path), 3, f"{path}.com")
    pms = []
    for i, pm in enumerate(d.get("point_masses", []) or []):
        p = f"{path}.point_masses[{i}]"
        pms.append(PointMass(str(c.get(pm, "name", p, default=f"pm{i}")),
                             c.num(c.get(pm, "mass", p), f"{p}.mass", lo=0.0),
                             c.vec(c.get(pm, "pos", p), 3, f"{p}.pos"), str(pm.get("kind", "actuator"))))
        if pms[-1].kind not in ("actuator", "structure", "linkage"):
            c.err(f"{p}.kind", "must be 'actuator', 'structure' or 'linkage'")
    inertia = None
    if d.get("inertia") is not None:
        raw = d["inertia"]
        if isinstance(raw, list) and len(raw) == 3 and all(isinstance(x, (int, float)) for x in raw):
            inertia = np.diag(np.array(raw, dtype=float))  # [Ixx, Iyy, Izz] about the COM
        elif isinstance(raw, list) and len(raw) == 3 and all(isinstance(x, list) for x in raw):
            inertia = np.array(raw, dtype=float)
            if inertia.shape != (3, 3):
                c.err(f"{path}.inertia", "expected [Ixx,Iyy,Izz] or a 3x3 matrix")
                inertia = None
        else:
            c.err(f"{path}.inertia", "expected [Ixx,Iyy,Izz] or a 3x3 matrix")
    two_force = bool(d.get("two_force", False))
    beam = None
    if isinstance(d, dict) and d.get("beam") is not None:
        bm = d["beam"]
        beam = (c.vec(bm.get("from") if isinstance(bm, dict) else None, 3, f"{path}.beam.from"),
                c.vec(bm.get("to") if isinstance(bm, dict) else None, 3, f"{path}.beam.to"))
    raw_sec = d.get("section") if isinstance(d, dict) else None
    segments: list = []
    if isinstance(raw_sec, list):               # several sections along the link: from / to in m from the beam start
        for k, sg in enumerate(raw_sec):
            pth = f"{path}.section[{k}]"
            if not isinstance(sg, dict):
                c.err(pth, "expected a mapping")
                continue
            s0 = c.num(sg.get("from", 0.0), f"{pth}.from", lo=0.0)
            to = sg.get("to", "end")
            s1 = None if to == "end" else c.num(to, f"{pth}.to", lo=0.0)
            sec_k = _section(c, {k_: v for k_, v in sg.items() if k_ not in ("from", "to")}, pth)
            if sec_k is not None:
                segments.append((s0, s1, sec_k))
    elif raw_sec is not None:
        sec_1 = _section(c, raw_sec, f"{path}.section")
        if sec_1 is not None:
            segments = [(0.0, None, sec_1)]
    return LinkSpec(name, mass, com, pms, inertia, segments[0][2] if segments else None, two_force, segments, beam)


# =========================================================================== section parsers (one per YAML block)


def _parse_scalars(c: _Check, raw: dict) -> tuple[float, float, int, float, float]:
    robot = c.get(raw, "robot", "")
    total_mass = c.num(c.get(robot, "total_mass", "robot"), "robot.total_mass", lo=1e-6)
    gravity = c.num(raw.get("gravity", 9.81), "gravity", lo=0.0)
    n_stations = int(c.num(raw.get("n_stations", 5), "n_stations", lo=2))
    ds = c.num(raw.get("double_support_fraction", 0.5), "double_support_fraction", lo=0.0)
    ss = c.num(raw.get("single_support_fraction", 1.0), "single_support_fraction", lo=0.0)
    return total_mass, gravity, n_stations, ds, ss


def _parse_joints(c: _Check, raw: dict) -> list[JointSpec]:
    joints: list[JointSpec] = []
    jl = c.get(raw, "joints", "")
    if not isinstance(jl, list) or not jl:
        c.err("joints", "expected a non-empty list (ordered pelvis -> foot)")
        return joints
    names: set[str] = set()
    for i, j in enumerate(jl):
        p = f"joints[{i}]"
        name = str(c.get(j, "name", p, default=f"joint{i}"))
        if name in names:
            c.err(f"{p}.name", f"duplicate joint name {name!r}")
        names.add(name)
        axis = c.vec(c.get(j, "axis", p), 3, f"{p}.axis")
        nrm = np.linalg.norm(axis)
        if nrm < 1e-9:
            c.err(f"{p}.axis", "axis must be non-zero")
            axis = np.array([1.0, 0, 0])
        else:
            axis = axis / nrm
        org = j.get("origin", {}) if isinstance(j, dict) else {}
        xyz = c.vec(org.get("xyz", [0, 0, 0]), 3, f"{p}.origin.xyz")
        rpy = np.radians(c.vec(org.get("rpy_deg", [0, 0, 0]), 3, f"{p}.origin.rpy_deg"))
        lim = c.get(j, "limits_deg", p)
        if (isinstance(lim, list) and len(lim) == 2 and all(isinstance(x, (int, float)) for x in lim)):
            if lim[0] >= lim[1]:
                c.err(f"{p}.limits_deg", f"lower must be < upper, got {lim}")
            limits = (math.radians(lim[0]), math.radians(lim[1]))
        else:
            c.err(f"{p}.limits_deg", f"expected [lo, hi] in degrees, got {lim!r}")
            limits = (-math.pi, math.pi)
        link = _link(c, c.get(j, "link", p, default={}), f"{p}.link")
        joints.append(JointSpec(name, axis, xyz, _rpy_to_R(rpy), limits, link))
    return joints


def _parse_foot(c: _Check, raw: dict) -> FootSpec:
    f = c.get(raw, "foot", "")
    cp = c.vec(c.get(f, "contact_point", "foot"), 3, "foot.contact_point")
    be = c.vec(f.get("beam_end", list(cp)), 3, "foot.beam_end") if isinstance(f, dict) else cp
    sole = f.get("sole", {}) if isinstance(f, dict) else {}
    return FootSpec(cp, be, c.num(sole.get("toe_x", 0.12), "foot.sole.toe_x"),
                    c.num(sole.get("heel_x", -0.06), "foot.sole.heel_x"),
                    c.num(sole.get("half_width", 0.04), "foot.sole.half_width", lo=0.0),
                    c.num(sole.get("thickness", 0.02), "foot.sole.thickness", lo=1e-4))


def _beam_vector(joints: list[JointSpec], foot: FootSpec, i: int) -> np.ndarray:
    """Beam of link i in its own frame: the override if given, else joint origin -> next joint origin."""
    lk = joints[i].link
    if lk.beam:
        return lk.beam[1] - lk.beam[0]
    return joints[i + 1].origin_xyz if i + 1 < len(joints) else foot.beam_end


def _resolve_links(c: _Check, joints: list[JointSpec], foot: FootSpec) -> None:
    """Fill in ``com: center`` and validate / sort the section segments against each link's beam length."""
    for i, j in enumerate(joints):
        if np.isnan(j.link.com[0]):
            start = j.link.beam[0] if j.link.beam else np.zeros(3)
            j.link.com = start + _beam_vector(joints, foot, i) / 2.0
    for i, j in enumerate(joints):
        segs = j.link.segments
        L = float(np.linalg.norm(_beam_vector(joints, foot, i)))
        if not segs or L < 1e-9:
            continue
        segs = sorted(((s0, L if s1 is None else s1, sec) for (s0, s1, sec) in segs), key=lambda t: t[0])
        tol = 1e-4
        if abs(segs[0][0]) > tol or abs(segs[-1][1] - L) > tol or any(
                abs(segs[k][1] - segs[k + 1][0]) > tol or segs[k][1] <= segs[k][0] for k in range(len(segs) - 1)):
            c.err(f"joints[{i}].link.section",
                  f"segments must be contiguous and cover the whole link, 0 .. {L:.4f} m from the joint origin "
                  f"(got {[(round(a_, 4), round(b_, 4)) for a_, b_, _ in segs]})")
        j.link.segments = segs
        j.link.section = segs[0][2]


def _parse_actuators(c: _Check, raw: dict) -> tuple[list[ActuatorSpec], dict[str, int]]:
    acts: list[ActuatorSpec] = []
    anames: dict[str, int] = {}
    al = c.get(raw, "actuators", "")
    if not isinstance(al, list) or not al:
        c.err("actuators", "expected a non-empty list")
        return acts, anames
    for i, a in enumerate(al):
        p = f"actuators[{i}]"
        name = str(c.get(a, "name", p, default=f"act{i}"))
        if name in anames:
            c.err(f"{p}.name", f"duplicate actuator name {name!r}")
        anames[name] = i
        kind = c.get(a, "kind", p, default="rotary")
        if kind not in ("rotary", "linear"):
            c.err(f"{p}.kind", f"must be 'rotary' or 'linear', got {kind!r}")
        cont = c.num(c.get(a, "continuous", p), f"{p}.continuous", lo=1e-12)
        peak = c.num(c.get(a, "peak", p), f"{p}.peak", lo=1e-12)
        if math.isfinite(cont) and math.isfinite(peak) and peak < cont:
            c.err(f"{p}.peak", "peak must be >= continuous")
        acts.append(ActuatorSpec(name, kind, cont, peak,
                                 c.num(c.get(a, "max_speed", p), f"{p}.max_speed", lo=1e-12),
                                 c.num(a.get("rotor_inertia", 0.0), f"{p}.rotor_inertia", lo=0.0),
                                 str(a.get("model", "")),
                                 c.num(a.get("rated_speed", 0.0), f"{p}.rated_speed", lo=0.0)))
    return acts, anames


_CHOICE_FIELDS = ("continuous", "peak", "max_speed", "rated_speed", "rotor_inertia")


def _parse_actuator_options(c: _Check, raw: dict, acts: list[ActuatorSpec], anames: dict[str, int],
                            joints: list[JointSpec]) -> dict[str, ActuatorOption]:
    """``actuator_options:``  key -> {label, applies_to, mass_points, default, choices}.

    The ``default`` choice must agree with the ``actuators:`` entries it replaces (and the lumped masses), so the
    two descriptions of the default hardware cannot drift apart."""
    out: dict[str, ActuatorOption] = {}
    all_pm = {pm.name: pm for j in joints for pm in j.link.point_masses}
    for key, o in (raw.get("actuator_options") or {}).items():
        p = f"actuator_options.{key}"
        if not isinstance(o, dict):
            c.err(p, "expected a mapping")
            continue
        applies = c.get(o, "applies_to", p, default=[])
        bad = [x for x in applies if x not in anames] if isinstance(applies, list) else ["?"]
        if bad:
            c.err(f"{p}.applies_to", f"unknown actuator name(s) {bad}")
            continue
        mps = o.get("mass_points", []) or []
        bad = [x for x in mps if x not in all_pm]
        if bad:
            c.err(f"{p}.mass_points", f"unknown lumped-mass name(s) {bad}")
            continue
        choices: dict[str, dict] = {}
        for cname, spec in (c.get(o, "choices", p, default={}) or {}).items():
            cp = f"{p}.choices.{cname}"
            if not isinstance(spec, dict):
                c.err(cp, "expected a mapping")
                continue
            ch = {k: c.num(spec.get(k, 0.0 if k in ("rated_speed", "rotor_inertia") else None), f"{cp}.{k}", lo=0.0)
                  for k in _CHOICE_FIELDS}
            ch["mass"] = c.num(spec.get("mass"), f"{cp}.mass", lo=0.0) if "mass" in spec else None
            ch["note"] = str(spec.get("note", ""))
            choices[str(cname)] = ch
        default = str(c.get(o, "default", p, default=""))
        if default not in choices:
            c.err(f"{p}.default", f"must be one of {list(choices)}")
            continue
        d = choices[default]
        for an in applies:                                  # default choice == the actuators: entries
            a = acts[anames[an]]
            for k in _CHOICE_FIELDS:
                if not math.isclose(getattr(a, k), d[k], rel_tol=1e-6, abs_tol=1e-12):
                    c.err(f"{p}.choices.{default}.{k}", f"disagrees with actuators entry {an!r} "
                                                        f"({getattr(a, k)} vs {d[k]}); the default choice must match")
        if d["mass"] is not None:
            for mp in mps:
                if not math.isclose(all_pm[mp].mass, d["mass"], rel_tol=1e-6):
                    c.err(f"{p}.choices.{default}.mass", f"disagrees with lumped mass {mp!r} ({all_pm[mp].mass})")
        out[str(key)] = ActuatorOption(str(key), str(o.get("label", key)), list(applies), list(mps), default, choices)
    return out


def _parse_transmissions(c: _Check, raw: dict, joints: list[JointSpec], anames: dict[str, int],
                         n_acts: int) -> list[TransmissionSpec]:
    jidx = {j.name: i for i, j in enumerate(joints)}
    trans: list[TransmissionSpec] = []
    tl = c.get(raw, "transmissions", "")
    if not isinstance(tl, list) or not tl:
        c.err("transmissions", "expected a non-empty list")
        tl = []
    for i, t in enumerate(tl):
        p = f"transmissions[{i}]"
        jn = c.get(t, "joints", p, default=[])
        an = c.get(t, "actuators", p, default=[])
        bad = [x for x in jn if x not in jidx] if isinstance(jn, list) else ["?"]
        if bad:
            c.err(f"{p}.joints", f"unknown joint name(s) {bad}")
            continue
        bad = [x for x in an if x not in anames] if isinstance(an, list) else ["?"]
        if bad:
            c.err(f"{p}.actuators", f"unknown actuator name(s) {bad}")
            continue
        if len(jn) != len(an):
            c.err(p, f"needs as many actuators as joints ({len(jn)} joints, {len(an)} actuators)")
            continue
        kind = c.get(t, "type", p, default="ratio")
        spec = TransmissionSpec([jidx[x] for x in jn], [anames[x] for x in an], kind)
        if kind == "ratio":
            if len(jn) != 1:
                c.err(p, "type 'ratio' is for a single joint; use 'matrix' for coupled joints")
            spec.ratio = c.num(c.get(t, "ratio", p), f"{p}.ratio")
            if spec.ratio == 0:
                c.err(f"{p}.ratio", "must be non-zero")
        elif kind == "matrix":
            m = c.get(t, "matrix", p, default=None)
            try:
                M = np.array(m, dtype=float)
                if M.shape != (len(jn), len(jn)):
                    raise ValueError
                if abs(np.linalg.det(M)) < 1e-14:
                    c.err(f"{p}.matrix", "matrix is singular")
                spec.matrix = M
            except (ValueError, TypeError):
                c.err(f"{p}.matrix", f"expected a {len(jn)}x{len(jn)} numeric matrix")
        elif kind == "table":
            tb = c.get(t, "table", p, default={})
            qd = tb.get("q_deg") if isinstance(tb, dict) else None
            rt = tb.get("ratio") if isinstance(tb, dict) else None
            if (len(jn) != 1 or not isinstance(qd, list) or not isinstance(rt, list)
                    or len(qd) != len(rt) or len(qd) < 2 or list(qd) != sorted(qd)):
                c.err(f"{p}.table", "need a single joint and equal-length ascending q_deg / ratio lists (>=2 points)")
            else:
                spec.table_q = np.radians(np.array(qd, dtype=float))
                spec.table_ratio = np.array(rt, dtype=float)
        elif kind == "function":
            fn = c.get(t, "function", p, default="")
            if not isinstance(fn, str) or not re.fullmatch(r"[\w.]+:\w+", fn):
                c.err(f"{p}.function", "expected 'package.module:function_name'")
            spec.function = fn
            prm = t.get("params", {}) or {}
            if not isinstance(prm, dict):
                c.err(f"{p}.params", "expected a mapping of keyword arguments for the function")
            else:
                spec.params = prm
        else:
            c.err(f"{p}.type", f"unknown type {kind!r} (ratio | matrix | table | function)")
        trans.append(spec)
    cov = sorted(j for t in trans for j in t.joints)
    if cov != list(range(len(joints))):
        c.err("transmissions", "every joint must appear in exactly one transmission group")
    acov = sorted(a for t in trans for a in t.actuators)
    if acov != list(range(n_acts)):
        c.err("transmissions", "every actuator must appear in exactly one transmission group")
    return trans


def _parse_gaits(c: _Check, raw: dict, n_joints: int) -> dict[str, Gait]:
    gaits: dict[str, Gait] = {}
    g_all = raw.get("gaits")
    if g_all is None and raw.get("gait") is not None:      # legacy single-gait block
        g_all = {"walking": raw["gait"]}
    if g_all is not None and not isinstance(g_all, dict):
        c.err("gaits", "expected a mapping of gait name -> {description, stance_time, stance_fraction, keyframes}")
        return gaits
    for gname, g_raw in (g_all or {}).items():
        gp = f"gaits.{gname}"
        kl = g_raw.get("keyframes") if isinstance(g_raw, dict) else None
        if not isinstance(kl, list) or len(kl) < 2:
            c.err(f"{gp}.keyframes", "expected a list of at least 2 keyframes")
            continue
        keys = []
        for ki, k in enumerate(kl):
            pth = f"{gp}.keyframes[{ki}]"
            keys.append(GaitKey(c.num(c.get(k, "phase", pth), f"{pth}.phase", lo=0.0, hi=1.0),
                                str(k.get("name", "")) if isinstance(k, dict) else "",
                                np.radians(c.vec(c.get(k, "q_deg", pth), n_joints, f"{pth}.q_deg")),
                                c.vec(c.get(k, "force_bw", pth), 3, f"{pth}.force_bw"),
                                c.num(k.get("cop_x", 0.0) if isinstance(k, dict) else 0.0, f"{pth}.cop_x")))
        keys.sort(key=lambda g_: g_.phase)
        if (abs(keys[0].phase) > 1e-9 or abs(keys[-1].phase - 1.0) > 1e-9
                or any(b_.phase - a_.phase <= 1e-9 for a_, b_ in zip(keys, keys[1:]))):
            c.err(f"{gp}.keyframes", "phases must be strictly increasing, starting at 0 and ending at 1")
        gaits[str(gname)] = Gait(keys, str(g_raw.get("description", "")),
                                 c.num(g_raw.get("stance_time", 0.6), f"{gp}.stance_time", lo=0.05),
                                 c.num(g_raw.get("stance_fraction", 0.6), f"{gp}.stance_fraction", lo=0.1, hi=1.0), str(gname))
    return gaits


def _parse_motion(c: _Check, motion: Any, p: str, gaits: dict[str, Gait]) -> dict | None:
    """A preset's optional ``motion:`` block.  Only ``gait -> gait, phase`` is supported (landing and stand-speed motions
    were removed: out of scope for the walking / squat / step-up load cases)."""
    if motion is None:
        return None
    if not isinstance(motion, dict) or motion.get("type") != "gait":
        c.err(f"{p}.motion", "expected a mapping with type: gait")
        return None
    gname = str(motion.get("gait", "walking"))
    if gname not in gaits:
        c.err(f"{p}.motion.gait", f"unknown gait {gname!r} (define it under gaits:)")
    return {"type": "gait", "gait": gname, "phase": c.num(motion.get("phase", 0.25), f"{p}.motion.phase", lo=0.0, hi=1.0)}


def _parse_presets(c: _Check, raw: dict, n_joints: int, gaits: dict[str, Gait]) -> dict[str, Preset]:
    presets: dict[str, Preset] = {}
    for name, pr in (raw.get("presets") or {}).items():
        p = f"presets.{name}"
        qd = c.vec(c.get(pr, "q_deg", p), n_joints, f"{p}.q_deg") if n_joints else np.zeros(0)
        lf = c.num(c.get(pr, "load_fraction", p), f"{p}.load_fraction", lo=0.0)
        sup = pr.get("support", "single") if isinstance(pr, dict) else "single"
        if sup not in ("single", "double"):
            c.err(f"{p}.support", "must be 'single' or 'double'")
        motion = _parse_motion(c, pr.get("motion") if isinstance(pr, dict) else None, p, gaits)
        desc = pr.get("description", "") if isinstance(pr, dict) else ""
        if not isinstance(desc, str):
            c.err(f"{p}.description", "must be text")
            desc = ""
        presets[str(name)] = Preset(str(name), np.radians(qd), lf, sup, motion, desc)
    return presets


# =========================================================================== public API


def load_config(source: str | Path | dict) -> Config:
    """Load and validate from a YAML path or an already-parsed dict (the tests build small dicts)."""
    path: Path | None = None
    placeholder_lines: list[tuple[int, str]] = []
    if isinstance(source, dict):
        raw = source
    else:
        path = Path(source)
        if not path.exists():
            raise ConfigError([f"config file not found: {path}"])
        text = path.read_text(encoding="utf-8")
        placeholder_lines = find_placeholders(text)
        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise ConfigError([f"YAML syntax error: {e}"]) from e
    if not isinstance(raw, dict):
        raise ConfigError(["top level must be a mapping"])

    c = _Check()
    total_mass, gravity, n_stations, ds_frac, ss_frac = _parse_scalars(c, raw)
    joints = _parse_joints(c, raw)
    foot = _parse_foot(c, raw)
    _resolve_links(c, joints, foot)
    acts, anames = _parse_actuators(c, raw)
    options = _parse_actuator_options(c, raw, acts, anames, joints)
    trans = _parse_transmissions(c, raw, joints, anames, len(acts))
    gaits = _parse_gaits(c, raw, len(joints))
    presets = _parse_presets(c, raw, len(joints), gaits)
    if c.errors:
        raise ConfigError(c.errors)
    return Config(total_mass, gravity, n_stations, joints, foot, acts, trans, presets, ds_frac, ss_frac, path,
                  placeholder_lines, gaits, options, {k: o.default for k, o in options.items()})


def with_actuator_choices(cfg: Config, selections: dict[str, str] | None = None) -> Config:
    """A deep copy of ``cfg`` with each ``actuator_options`` group set to the selected model.

    ``selections`` maps option key -> model name (missing keys keep the default).  Replaces the ratings, rotor
    inertia and (if given) the lumped-mass weight of the affected actuators; geometry is NOT changed, so a motor
    with a different size needs the geometry revisited by hand (the choice's ``note`` says so)."""
    selections = selections or {}
    new = copy.deepcopy(cfg)
    for key, opt in new.actuator_options.items():
        pick = selections.get(key, opt.default)
        if pick not in opt.choices:
            raise ValueError(f"unknown choice {pick!r} for {key!r}; options: {list(opt.choices)}")
        ch = opt.choices[pick]
        for an in opt.applies_to:
            a = next(a for a in new.actuators if a.name == an)
            a.model = pick
            for k in _CHOICE_FIELDS:
                setattr(a, k, ch[k])
        if ch["mass"] is not None:
            for j in new.joints:
                for pm in j.link.point_masses:
                    if pm.name in opt.mass_points:
                        pm.mass = ch["mass"]
        new.active_choices[key] = pick
    return new


# ------------------------------------------------------------- placeholders / preset write-back

_PLACEHOLDER_RE = re.compile(r"^[^#\n]*\S[^#\n]*#\s*PLACEHOLDER", re.IGNORECASE)


def find_placeholders(text: str) -> list[tuple[int, str]]:
    """Lines that carry a value *and* a trailing ``# PLACEHOLDER`` marker (pure comment lines are ignored)."""
    return [(i + 1, ln.strip()) for i, ln in enumerate(text.splitlines()) if _PLACEHOLDER_RE.match(ln)]


def save_preset(path: str | Path, name: str, q_deg: list[float], load_fraction: float,
                support: str, description: str = "", motion: dict | None = None) -> None:
    """Write a preset into the YAML text, keeping every comment.

    ``presets:`` must be the last top-level section.  Presets are one-line flow mappings, so an existing preset of
    the same name is replaced in place; a new one is appended after the last preset."""
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", name):
        raise ValueError("preset name may only contain letters, digits, '_' and '-'")
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next((i for i, ln in enumerate(lines) if re.match(r"^presets\s*:", ln)), None)
    new = (f"  {name}: {{q_deg: [{', '.join(f'{x:.1f}' for x in q_deg)}], "
           f"load_fraction: {load_fraction:g}, support: {support}")
    if motion:
        new += ", motion: {" + ", ".join(f"{k}: {v:g}" if isinstance(v, (int, float)) else f"{k}: {v}"
                                          for k, v in motion.items()) + "}"
    desc = " ".join(description.split())          # one line; JSON quoting is valid YAML
    if desc:
        new += ", description: " + json.dumps(desc, ensure_ascii=False)
    new += "}"
    if start is None:
        lines += ["", "presets:", new]
    else:
        end = len(lines)
        for i in range(start + 1, len(lines)):
            if lines[i] and not lines[i][0].isspace() and not lines[i].startswith("#"):
                end = i
                break
        hit = next((i for i in range(start + 1, end) if re.match(rf"^\s+{re.escape(name)}\s*:", lines[i])), None)
        if hit is not None:
            lines[hit] = new
        else:
            last = max((i for i in range(start, end) if lines[i].strip()), default=start)
            lines.insert(last + 1, new)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
