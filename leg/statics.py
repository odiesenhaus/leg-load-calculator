"""B. Quasi-static joint torques.

Sign convention (used everywhere in the tool): ``tau`` is the torque the *actuator must apply to the distal link* about
the joint axis (positive = right-hand about the world-resolved axis) to hold the pose.  Equilibrium of everything distal
to joint i, with the actuator torque ``tau_i`` acting along ``a_i``:

    tau_i  +  a_i . [ sum_k r_k x f_k ]  = 0,        f_k = ground reaction or a weight, r_k from the joint axis

which in Jacobian form is

    tau = -( J^T W_grf  +  sum_m Jv_m^T (m_m g) )

``W_grf`` is the wrench the ground exerts ON the foot (Fz > 0 pushes the foot up), applied at the contact point, and the
second term sums the weight of every link COM and lumped mass.  The minus sign is what makes ``tau`` an *actuator*
(holding) torque rather than the external load's torque.

The tool is quasi-static: accelerations are ignored.  The hook for full inverse dynamics is documented in
``dynamics.inverse_dynamics_hook``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .kinematics import FK, foot_jacobian


@dataclass
class Load:
    """Ground reaction on the foot, world frame, applied at the contact point."""
    force: np.ndarray = field(default_factory=lambda: np.zeros(3))   # N (Fz > 0 is up, into the foot)
    moment: np.ndarray = field(default_factory=lambda: np.zeros(3))  # N*m about the contact point


@dataclass
class StaticResult:
    tau: np.ndarray             # total actuator-side joint torque to hold the pose [N*m]
    tau_grf: np.ndarray         # part due to the ground reaction
    tau_gravity: np.ndarray     # part due to the weights of links and lumped masses
    per_mass: list[tuple[str, np.ndarray]]   # (mass-point name, its contribution to tau): a diagnostic breakdown


def gravity_vec(cfg: Config) -> np.ndarray:
    """Gravity acceleration vector in the world frame (-z)."""
    return np.array([0.0, 0.0, -cfg.gravity])


def joint_torques(cfg: Config, fk: FK, load: Load) -> StaticResult:
    """B.  ``tau = -(J^T W_grf + sum_m Jv_m^T m g)``, returned with its two parts.

    The Jacobian of a mass point only has columns for the joints proximal to the body it sits on, which is exactly the
    "propagate each weight through the chain" recursion of the specification.
    """
    J = foot_jacobian(fk)
    W = np.concatenate([load.force, load.moment])
    tau_grf = -(J.T @ W)
    g = gravity_vec(cfg)
    tau_g = np.zeros(cfg.n)
    per_mass = []
    for m in fk.mass_points:
        Jm = fk.jacobian_point(m.p, m.link)
        t = -(Jm[:3].T @ (m.mass * g))
        per_mass.append((m.name, t))
        tau_g += t
    return StaticResult(tau_grf + tau_g, tau_grf, tau_g, per_mass)


def default_load(cfg: Config, fraction: float, payload_mass: float = 0.0,
                 horizontal=(0.0, 0.0)) -> Load:
    """"fraction x robot weight" vertical ground reaction, plus an optional horizontal force.

    The weight includes ``payload_mass``.  The fraction is the share carried by THIS leg: 0.5 for an even double-support
    split, 1.0 for single support, 3.0 for a landing peak.  No torso model: the load is simply what the foot receives.
    """
    Fz = fraction * (cfg.robot_total_mass + payload_mass) * cfg.gravity
    return Load(np.array([horizontal[0], horizontal[1], Fz]), np.zeros(3))
