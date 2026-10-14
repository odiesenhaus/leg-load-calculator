"""F. Joint-space mass matrix (inertia of the links, lumped masses and reflected rotors).

    H(q) = sum over bodies of ( m Jv^T Jv  +  Jw^T I_world Jw )          link structure (mass + inertia) and lumped masses
           + T^T diag(I_rotor) T                                           reflected rotor inertia (see actuators.py)

Lumped masses (actuator bodies, rods) are point masses with no inertia of their own.  A link's structure inertia is the
config ``inertia`` about its COM or, by default, a thin rod ``m L^2/12 (I - u u^T)`` along its beam.

Nothing in the quasi-static pipeline uses this today.  It is kept for the planned acceleration-headroom and mass-budget
views.  (The foot-impact estimate that used to live here was removed: it depended on an uncertain rotor-inertia reading,
assumed a fixed base, produced no force, and fed nothing else.)
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .kinematics import FK


def link_inertia_local(cfg: Config, i: int) -> np.ndarray:
    """3x3 inertia of link ``i``'s structure about its COM in the link frame: the config value, else a thin rod
    ``m L^2/12`` perpendicular to the beam (L = beam length; zero for a zero-length beam)."""
    lk = cfg.joints[i].link
    if lk.inertia is not None:
        return lk.inertia
    nxt = cfg.joints[i + 1].origin_xyz if i + 1 < cfg.n else cfg.foot.beam_end
    if lk.beam:
        nxt = lk.beam[1] - lk.beam[0]
    L = float(np.linalg.norm(nxt))
    if L < 1e-9 or lk.mass == 0:
        return np.zeros((3, 3))
    u = nxt / L
    return lk.mass * L**2 / 12.0 * (np.eye(3) - np.outer(u, u))


def link_mass_matrix(cfg: Config, fk: FK) -> np.ndarray:
    """Joint-space mass matrix of the links and lumped masses (no rotors): ``sum m Jv^T Jv + Jw^T I_w Jw``."""
    H = np.zeros((cfg.n, cfg.n))
    for m in fk.mass_points:
        J = fk.jacobian_point(m.p, m.link)
        H += m.mass * J[:3].T @ J[:3]
        if m.is_com:                                  # only the structure mass carries the link inertia
            Iw = fk.R[m.link] @ link_inertia_local(cfg, m.link) @ fk.R[m.link].T
            H += J[3:].T @ Iw @ J[3:]
    return H


def inverse_dynamics_hook(cfg: Config, fk: FK, qdd: np.ndarray):
    """Hook for full inverse dynamics (NOT implemented: the scope is quasi-static).

    To add accelerations: ``tau_act = statics.joint_torques(...).tau + H(q) qdd + C(q, qd) qd``, with H from
    ``link_mass_matrix`` + ``reflected_rotor_inertia`` and C from the Christoffel symbols (or RNEA).
    ``structure._wrench`` then needs inertial forces added to each mass point's external force
    (``w = m (g - a)``): it is the single place where weights enter the link loads.
    """
    raise NotImplementedError("quasi-static tool: accelerations are out of scope")
