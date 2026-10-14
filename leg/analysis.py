"""One call that runs the whole pipeline (A-F) for a pose + load and returns everything the UI / exporter needs.

``analyze`` = forward kinematics -> foot Jacobian + reach check -> static joint torques -> transmission and
actuator limits -> link loads and bearing reactions, plus a list of human-readable warnings
(limits violated, centre of pressure off the sole, actuator over rating, safety factor < 1, ...).
An unavailable transmission (e.g. an unreachable ankle linkage) is reported in ``errors`` and disables only the
actuator results; the rest is still returned.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import actuators, kinematics, rods as rods_mod, statics, structure
from .config import Config
from .names import pretty


@dataclass
class Inputs:
    """Everything the analysis needs besides the config."""
    q: np.ndarray                      # joint angles [rad], joint order
    pelvis_pitch: float = 0.0          # rad; + = forward lean
    pelvis_roll: float = 0.0           # rad
    load: statics.Load = field(default_factory=statics.Load)       # ground reaction (world frame) at the contact point
    cop_offset: tuple[float, float] = (0.0, 0.0)   # m: contact point shift in the foot frame (centre of pressure)
    payload_mass: float = 0.0          # kg (informational here: it only changes the UI's "x body weight" loads)
    qd: np.ndarray | None = None       # joint velocities [rad/s]; None -> no speed demand


@dataclass
class Results:
    inputs: Inputs
    fk: kinematics.FK
    J: np.ndarray                          # 6xn foot Jacobian
    singularity: dict                      # see kinematics.singularity_report
    static: statics.StaticResult           # joint torques and their split
    T: np.ndarray | None                   # transmission matrix (None if unavailable)
    actuators: list[actuators.ActuatorLoad] | None
    shares: list[actuators.JointShare] | None   # per-joint share of its actuator group (see actuators.JointShare)
    rods: list[rods_mod.RodLoad] | None       # ankle rod forces (None: no ankle linkage, or it cannot be solved here)
    links: list[structure.LinkResult]
    flanges: list[structure.FlangeReaction]
    warnings: list[str]
    errors: list[str]


def analyze(cfg: Config, inp: Inputs) -> Results:
    """Run the full model for one pose and load (a few milliseconds)."""
    warnings: list[str] = []
    errors: list[str] = []
    fk = kinematics.forward(cfg, inp.q, inp.pelvis_pitch, inp.pelvis_roll, inp.cop_offset)
    J = kinematics.foot_jacobian(fk)
    sing = kinematics.singularity_report(cfg, fk, J)
    for j, qi in zip(cfg.joints, inp.q):
        if not (j.limits[0] - 1e-9 <= qi <= j.limits[1] + 1e-9):
            warnings.append(f"{pretty(j.name)} = {np.degrees(qi):.1f} deg is outside its limits "
                            f"[{np.degrees(j.limits[0]):.0f}, {np.degrees(j.limits[1]):.0f}] deg.")
    f = cfg.foot
    cx, cy = f.contact_point[0] + inp.cop_offset[0], f.contact_point[1] + inp.cop_offset[1]
    if not (f.heel_x - 1e-9 <= cx <= f.toe_x + 1e-9 and abs(cy) <= f.half_width + 1e-9):
        warnings.append(f"Centre of pressure ({cx * 1000:.0f}, {cy * 1000:.0f}) mm is outside the sole "
                        f"(x {f.heel_x * 1000:.0f}..{f.toe_x * 1000:.0f} mm, |y| <= {f.half_width * 1000:.0f} mm): "
                        "the foot would tip.")
    st = statics.joint_torques(cfg, fk, inp.load)
    T = acts = shares = rods = None
    try:
        T = actuators.transmission_matrix(cfg, inp.q)
        acts = actuators.actuator_loads(cfg, T, st.tau, inp.qd)
        shares = actuators.joint_shares(cfg, T, st.tau)
        for a in acts:
            if a.status == "over_peak":
                warnings.append(f"{pretty(a.name)}: {abs(a.value):.1f} {a.unit} exceeds PEAK ({a.peak:g}).")
            elif a.exceeds_continuous:
                warnings.append(f"{pretty(a.name)}: {abs(a.value):.1f} {a.unit} exceeds CONTINUOUS ({a.continuous:g}); "
                                "will heat up if held.")
            if a.speed_util is not None and a.speed_util > 1:
                warnings.append(f"{pretty(a.name)}: speed demand {a.speed:.2f} exceeds max {a.max_speed:g}.")
    except actuators.TransmissionError as e:
        errors.append(str(e))
    if acts is not None:
        try:
            rods = rods_mod.ankle_rod_loads(cfg, fk, [a.value for a in acts])
        except ValueError as e:
            errors.append(f"Ankle rod loads unavailable ({e}); lower-shank, ankle and foot loads below use the plain cut "
                          "method WITHOUT the rods and are not valid.")
    links = structure.link_loads(cfg, fk, inp.load, rods=rods)
    for lr in links:
        if lr.two_force_violation:
            warnings.append(lr.two_force_violation)
        if lr.min_sf is not None and lr.min_sf < 1.0:
            warnings.append(f"{pretty(lr.name)}: safety factor {lr.min_sf:.2f} < 1 (yield).")
    return Results(inp, fk, J, sing, st, T, acts, shares, rods, links, structure.flange_reactions(cfg, fk, inp.load, rods),
                   warnings, errors)
