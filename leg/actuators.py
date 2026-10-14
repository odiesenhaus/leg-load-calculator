"""C/E. Joint space <-> actuator space transmission, and the actuator limit checks.

Transmission matrix ``T(q)`` (n_actuators x n_joints) maps joint velocities to actuator OUTPUT velocities:

    qdot_act = T(q) qdot_joint                         (rad/s, or m/s for a linear actuator)

Power balance  ``tau_joint . qdot_joint = f_act . qdot_act``  then gives

    tau_joint = T^T f_act        <=>        f_act = T^{-T} tau_joint.

Special cases:  single joint with ratio N -> T = [N] -> tau_act = tau_joint / N.  Differential pair -> T = [[N,N],[N,-N]].
Parallel ankle -> T = d(lever angles)/d(ankle angles), pose dependent (``leg/linkages.py``).  The ratings in the config are
OUTPUT-side, so an actuator with an integrated gearbox uses ratio 1 and its rotor inertia is stored already multiplied by
N^2 (see ``reflected_rotor_inertia``).

Transmission types in the YAML: ``ratio`` (constant), ``matrix`` (constant n x n), ``table`` (single joint, ratio
interpolated against the joint angle) and ``function`` (a Python function ``fn(q_group, **params) -> matrix``).
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass

import numpy as np

from .config import Config, TransmissionSpec


class TransmissionError(Exception):
    """A transmission could not be evaluated at this pose (unimplemented hook, unreachable linkage, singular T)."""


def _group_matrix(spec: TransmissionSpec, qg: np.ndarray) -> np.ndarray:
    """T block of one transmission group at the group's joint angles ``qg``."""
    if spec.kind == "ratio":
        return np.array([[spec.ratio]])
    if spec.kind == "matrix":
        return spec.matrix
    if spec.kind == "table":
        # Pose-dependent single-joint ratio d(actuator)/d(joint): linear interpolation, clamped to the end values.
        return np.array([[np.interp(qg[0], spec.table_q, spec.table_ratio)]])
    if spec.kind == "function":
        mod, attr = spec.function.split(":")
        try:
            fn = getattr(importlib.import_module(mod), attr)
            M = np.atleast_2d(np.asarray(fn(qg, **spec.params), dtype=float))
        except NotImplementedError as e:
            raise TransmissionError(f"Transmission hook {spec.function} is not implemented yet: {e}") from e
        except (ImportError, AttributeError) as e:
            raise TransmissionError(f"Cannot import transmission function {spec.function}: {e}") from e
        except (ValueError, TypeError) as e:
            raise TransmissionError(f"{spec.function} failed at this pose: {e}") from e
        if M.shape != (len(qg), len(qg)):
            raise TransmissionError(f"{spec.function} returned shape {M.shape}, expected {(len(qg),) * 2}")
        return M
    raise TransmissionError(f"unknown transmission type {spec.kind}")


def transmission_matrix(cfg: Config, q) -> np.ndarray:
    """Assemble the full T(q); rows follow ``cfg.actuators``, columns ``cfg.joints`` (block structure per group)."""
    q = np.asarray(q, float)
    T = np.zeros((len(cfg.actuators), cfg.n))
    for spec in cfg.transmissions:
        T[np.ix_(spec.actuators, spec.joints)] = _group_matrix(spec, q[spec.joints])
    return T


def reflected_rotor_inertia(cfg: Config, T: np.ndarray) -> np.ndarray:
    """Joint-space inertia of the actuator rotors:  ``T^T diag(I_act) T``.

    ``I_act`` is each actuator's ``rotor_inertia`` as stored (referred to its OUTPUT shaft, i.e. N^2 x the datasheet
    value for a geared motor), and ``T`` converts joint rates to output rates, so any further lever / coupling ratio is
    included.  For a plain ratio-N transmission of a motor-side inertia this is the familiar N^2 I_rotor.
    """
    return T.T @ np.diag([a.rotor_inertia for a in cfg.actuators]) @ T


@dataclass
class ActuatorLoad:
    """One actuator's demand against its limits at one pose."""
    name: str
    kind: str
    unit: str               # "N*m" or "N"
    value: float            # signed torque / force the actuator must supply
    continuous: float
    peak: float
    util_cont: float        # |value| / continuous
    util_peak: float        # |value| / peak
    status: str             # "ok" (< 60 % of continuous) | "warn" (up to peak) | "over_peak"
    exceeds_continuous: bool  # |value| > continuous: would heat up if held
    speed: float | None = None      # actuator output speed from the joint velocities, if given
    max_speed: float = 0.0
    speed_util: float | None = None


@dataclass
class JointShare:
    """How much of its actuator group's capacity ONE joint's torque uses, on its own.

    Coupled joints (the ankle pitch / roll pair drives both ankle motors) cannot be given a utilisation by "the actuator
    that drives them": each motor serves both joints.  Because ``f = T^-T tau`` is linear, the load on each motor is the sum
    of what every joint's torque contributes, so the share of joint j is the motor load that tau_j alone would cause.  A
    joint with zero torque therefore has a zero share.  The real load of each motor (the sum over joints, where
    contributions can also cancel) is in ``ActuatorLoad``.
    """
    joint: int
    actuators: list[str]          # the motors in this joint's transmission group
    util_cont: float              # worst |f| / continuous over those motors, from this joint's torque alone
    util_peak: float
    status: str                   # same thresholds as ActuatorLoad
    exceeds_continuous: bool


def joint_shares(cfg: Config, T: np.ndarray, tau_joint: np.ndarray) -> list[JointShare]:
    """Per-joint share of the actuator capacity (see ``JointShare``)."""
    tau_joint = np.asarray(tau_joint, float)
    out = []
    for spec in cfg.transmissions:
        for j in spec.joints:
            e = np.zeros(cfg.n)
            e[j] = tau_joint[j]
            try:
                f = np.linalg.solve(T.T, e)
            except np.linalg.LinAlgError as err:
                raise TransmissionError("Transmission matrix is singular at this pose; "
                                        "actuator forces are unbounded.") from err
            uc = max(abs(f[a]) / cfg.actuators[a].continuous for a in spec.actuators)
            up = max(abs(f[a]) / cfg.actuators[a].peak for a in spec.actuators)
            rank = {"ok": 0, "warn": 1, "over_peak": 2}
            status = "ok"
            for a in spec.actuators:
                v, ac = abs(f[a]), cfg.actuators[a]
                status = max(status, "ok" if v < 0.6 * ac.continuous else ("warn" if v <= ac.peak else "over_peak"),
                             key=rank.get)
            out.append(JointShare(j, [cfg.actuators[a].name for a in spec.actuators], float(uc), float(up), status, bool(uc > 1.0)))
    out.sort(key=lambda s: s.joint)
    return out


def actuator_loads(cfg: Config, T: np.ndarray, tau_joint: np.ndarray,
                   qd_joint: np.ndarray | None = None) -> list[ActuatorLoad]:
    """E. Map joint torques to actuator loads (``T^-T tau``) and compare with the configured ratings.

    Status thresholds: green below 60 % of continuous, amber up to the peak rating, red above it.  Speeds are only
    evaluated when joint velocities are supplied.
    """
    try:
        f = np.linalg.solve(T.T, tau_joint)
    except np.linalg.LinAlgError as e:
        raise TransmissionError("Transmission matrix is singular at this pose; "
                                "actuator forces are unbounded.") from e
    w = None if qd_joint is None else T @ np.asarray(qd_joint, float)
    out = []
    for k, a in enumerate(cfg.actuators):
        v = abs(f[k])
        status = "ok" if v < 0.6 * a.continuous else ("warn" if v <= a.peak else "over_peak")
        out.append(ActuatorLoad(
            a.name, a.kind, "N*m" if a.kind == "rotary" else "N", float(f[k]), a.continuous, a.peak,
            float(v / a.continuous), float(v / a.peak), status, bool(v > a.continuous),
            None if w is None else float(w[k]), a.max_speed,
            None if w is None else abs(float(w[k])) / a.max_speed))
    return out
