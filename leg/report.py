"""Flatten a ``Results`` into a JSON-able dict and a per-joint CSV table (no UI imports).

The JSON is the complete record of one analysis (inputs, hardware choice, kinematics, torques, actuators, every link
station, bearing reactions, warnings).  The CSV is a compact one-row-per-joint summary.  Non-finite numbers are
written as strings ("inf") so the JSON stays strictly valid.
"""
from __future__ import annotations

import csv
import io
import json

import numpy as np

from .analysis import Results
from .config import Config


def _f(x):
    """float(x) if finite, else the string form (inf / nan); None stays None."""
    return None if x is None else (float(x) if np.isfinite(x) else str(x))


def to_dict(cfg: Config, r: Results) -> dict:
    """Everything about one analysis as nested plain Python types."""
    inp = r.inputs
    d: dict = {
        "inputs": {
            "joint_names": cfg.joint_names,
            "q_deg": np.degrees(inp.q).tolist(),
            "pelvis_pitch_deg": float(np.degrees(inp.pelvis_pitch)),
            "pelvis_roll_deg": float(np.degrees(inp.pelvis_roll)),
            "grf_force_N_world": inp.load.force.tolist(),
            "grf_moment_Nm_world": inp.load.moment.tolist(),
            "cop_offset_m_foot_frame": list(inp.cop_offset),
            "payload_mass_kg": inp.payload_mass,
            "joint_velocity_rad_s": None if inp.qd is None else inp.qd.tolist(),
            "robot_total_mass_kg": cfg.robot_total_mass,
            "actuator_choices": dict(cfg.active_choices),
            "config_file": str(cfg.path) if cfg.path else None,
            "placeholder_values_remaining": len(cfg.placeholder_lines),
        },
        "kinematics": {
            "joint_positions_m": r.fk.p.tolist(),
            "contact_point_m": r.fk.p_contact.tolist(),
            "foot_jacobian": r.J.tolist(),
            "reach_ratio": _f(r.singularity["reach_ratio"]),
        },
        "joint_torques_Nm": {
            n: {"total": float(t), "ground_reaction": float(g), "gravity": float(w)}
            for n, t, g, w in zip(cfg.joint_names, r.static.tau, r.static.tau_grf, r.static.tau_gravity)
        },
        "actuators": None if r.actuators is None else [
            {"name": a.name, "unit": a.unit, "value": a.value, "continuous": a.continuous, "peak": a.peak,
             "util_continuous": a.util_cont, "util_peak": a.util_peak, "status": a.status,
             "exceeds_continuous": a.exceeds_continuous, "speed": a.speed, "max_speed": a.max_speed}
            for a in r.actuators],
        "links": [
            {"name": lr.name, "length_m": lr.length, "two_force": lr.two_force,
             "peak_bending_Nm": lr.peak_bending.M_bend, "peak_bending_at_s_m": lr.peak_bending.s,
             "peak_stress_Pa": _f(lr.peak_stress.sigma) if lr.peak_stress else None,
             "peak_stress_at_s_m": lr.peak_stress.s if lr.peak_stress else None,
             "min_safety_factor": _f(lr.min_sf), "two_force_violation": lr.two_force_violation,
             "stations": [{"s_m": s.s, "N": s.N, "Vy": s.Vy, "Vz": s.Vz, "T": s.T, "My": s.My, "Mz": s.Mz,
                           "sigma_Pa": _f(s.sigma), "sf": _f(s.sf)} for s in lr.stations]}
            for lr in r.links],
        "flange_reactions": [
            {"joint": f.joint, "force_joint_frame_N": f.force_local.tolist(),
             "moment_joint_frame_Nm": f.moment_local.tolist(), "axial_N": f.axial_force,
             "radial_N": f.radial_force, "tilt_moment_Nm": f.tilt_moment}
            for f in r.flanges],
        "ankle_rods": None if r.rods is None else [
            {"actuator": rd.name, "force_N": rd.force, "lever_angle_deg": float(np.degrees(rd.lever_angle)),
             "effective_lever_arm_m": rd.arm, "rod_length_m": rd.length} for rd in r.rods],
        "warnings": r.warnings + r.errors,
    }
    return d


def to_json(cfg: Config, r: Results) -> str:
    """The full record as indented JSON (numpy scalars / arrays are converted)."""
    def _default(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not JSON serializable: {type(o).__name__}")

    return json.dumps(to_dict(cfg, r), indent=2, default=_default)


def to_csv(cfg: Config, r: Results) -> str:
    """One row per joint: angle, torque split, the driving actuator, and the distal link's peak load / stress.

    For coupled joints (the ankle pair) the actuator columns show the busier actuator of the group, matching the UI.
    """
    group_of = {j: spec.actuators for spec in cfg.transmissions for j in spec.joints}    # joint -> actuator indices
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["joint", "angle_deg", "tau_total_Nm", "tau_grf_Nm", "tau_gravity_Nm", "actuator", "actuator_value",
                "actuator_unit", "util_continuous", "util_peak", "link", "peak_bending_Nm", "peak_stress_MPa",
                "min_safety_factor", "bearing_axial_N", "bearing_radial_N", "bearing_tilt_Nm"])
    for i, n in enumerate(cfg.joint_names):
        a = None
        if r.actuators:
            a = max((r.actuators[k] for k in group_of.get(i, [])), key=lambda x: x.util_cont, default=None)
        lr, fl = r.links[i], r.flanges[i]
        w.writerow([n, f"{np.degrees(r.inputs.q[i]):.3f}", f"{r.static.tau[i]:.4f}", f"{r.static.tau_grf[i]:.4f}",
                    f"{r.static.tau_gravity[i]:.4f}", a.name if a else "", f"{a.value:.4f}" if a else "",
                    a.unit if a else "", f"{a.util_cont:.4f}" if a else "", f"{a.util_peak:.4f}" if a else "",
                    lr.name, f"{lr.peak_bending.M_bend:.4f}",
                    f"{lr.peak_stress.sigma / 1e6:.4f}" if lr.peak_stress else "",
                    f"{lr.min_sf:.4f}" if lr.min_sf is not None else "",
                    f"{fl.axial_force:.3f}", f"{fl.radial_force:.3f}", f"{fl.tilt_moment:.4f}"])
    return buf.getvalue()
