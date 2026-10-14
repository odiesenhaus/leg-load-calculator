"""Forward kinematics, mass points, beams and the foot Jacobian, all in the WORLD frame.

Conventions (the README has the full statement and the equations):

* World frame: x forward, y left, z up.  Its origin is the pelvis centre line at the height of the hip-pitch axis; the
  first joint's ``origin_xyz`` places the leg relative to it.  Gravity is -z.
* The pelvis (base) orientation is ``R_base = Ry(pitch) @ Rx(roll)`` (positive pitch = forward lean about +y, positive
  roll about +x).  It rotates the whole leg; the ground and the ground reaction stay in the world frame.
* Joint i frame:  ``T_i = T_{i-1} * [fixed origin_i] * Rot(axis_i, q_i)``.  ``axis_i`` is given in the joint frame and is
  unchanged by the rotation about itself.  Link i is the body distal to joint i and is rigidly attached to frame i.
* Positive q is a right-hand rotation about the configured axis.
* Jacobian columns are ``[v; w] = J qdot`` with ``J_v[:, j] = a_j x (p - p_j)`` and ``J_w[:, j] = a_j`` for the joints
  j proximal to (or at) the body the point is attached to.

Every quantity here is a pure function of the configuration and the joint angles: no state, no UI.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config


def rot_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix: rotation by ``angle`` [rad] about the unit vector ``axis`` (right-hand rule)."""
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def base_rotation(pelvis_pitch: float, pelvis_roll: float) -> np.ndarray:
    """Pelvis orientation ``Ry(pitch) @ Rx(roll)`` [rad]: positive pitch = forward lean, positive roll about +x."""
    cp, sp, cr, sr = np.cos(pelvis_pitch), np.sin(pelvis_pitch), np.cos(pelvis_roll), np.sin(pelvis_roll)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Ry @ Rx


@dataclass
class MassPoint:
    """A point mass in the world: a link's structure COM (``is_com``) or a lumped actuator / rod."""
    link: int                  # index of the link (== joint index) it is rigidly attached to
    name: str                  # "<link>:structure" or "<link>:<point mass name>"
    mass: float                # kg
    p: np.ndarray              # world position [m]
    is_com: bool               # True for the link's structure mass (the only one that carries the link inertia)
    kind: str = "actuator"     # "structure" | "actuator" | "linkage": only affects how it is drawn


@dataclass
class FK:
    """Result of ``forward``: everything the statics / structure / dynamics code needs, in world coordinates."""
    q: np.ndarray              # joint angles [rad]
    R_base: np.ndarray         # pelvis orientation
    p: np.ndarray              # (n,3) joint frame origins
    R: np.ndarray              # (n,3,3) joint frame orientations after the joint rotation
    axis: np.ndarray           # (n,3) joint axes
    beam_end: np.ndarray       # (n,3) end of each link's structural beam
    mass_points: list[MassPoint]
    p_contact: np.ndarray      # foot contact point (sole point under the ankle, shifted by the CoP offset)
    cop_offset: np.ndarray     # (dx, dy, 0) shift of the contact point in the foot frame
    beam_start: np.ndarray = None   # (n,3) start of each link's structural beam (== p unless a beam override is set)

    def jacobian_point(self, point: np.ndarray, link: int) -> np.ndarray:
        """6xn Jacobian [Jv; Jw] of a point rigidly attached to ``link`` (columns of joints distal to it are 0)."""
        n = len(self.q)
        J = np.zeros((6, n))
        for j in range(link + 1):
            J[:3, j] = np.cross(self.axis[j], point - self.p[j])
            J[3:, j] = self.axis[j]
        return J


def forward(cfg: Config, q, pelvis_pitch: float = 0.0, pelvis_roll: float = 0.0,
            cop_offset=(0.0, 0.0)) -> FK:
    """Forward kinematics of every joint, link beam, mass point and the foot contact point.

    ``cop_offset`` = (dx, dy): shift of the contact point in the foot (last link) frame, i.e. the centre of pressure
    relative to the point under the ankle.  The contact point stays at the sole height.
    """
    q = np.asarray(q, dtype=float)
    n = cfg.n
    if q.shape != (n,):
        raise ValueError(f"q must have {n} entries, got shape {q.shape}")
    Rb = base_rotation(pelvis_pitch, pelvis_roll)
    p = np.zeros((n, 3))
    R = np.zeros((n, 3, 3))
    axis = np.zeros((n, 3))
    beam_end = np.zeros((n, 3))
    beam_start = np.zeros((n, 3))
    mass_points: list[MassPoint] = []
    Rp, pp = Rb, np.zeros(3)                       # parent frame: starts at the pelvis
    for i, j in enumerate(cfg.joints):
        p[i] = pp + Rp @ j.origin_xyz
        Rj0 = Rp @ j.origin_R                      # joint frame before the rotation about its own axis
        axis[i] = Rj0 @ j.axis
        R[i] = Rj0 @ rot_axis(j.axis, q[i])
        Rp, pp = R[i], p[i]
    for i, j in enumerate(cfg.joints):
        nxt = cfg.joints[i + 1].origin_xyz if i + 1 < cfg.n else cfg.foot.beam_end
        b0, b1 = j.link.beam if j.link.beam else (np.zeros(3), nxt)          # default beam: joint -> next joint
        beam_start[i] = p[i] + R[i] @ b0
        beam_end[i] = p[i] + R[i] @ b1
        lk = j.link
        if lk.mass > 0:
            mass_points.append(MassPoint(i, f"{lk.name}:structure", lk.mass, p[i] + R[i] @ lk.com, True, "structure"))
        for pm in lk.point_masses:
            if pm.mass > 0:
                mass_points.append(MassPoint(i, f"{lk.name}:{pm.name}", pm.mass, p[i] + R[i] @ pm.pos, False, pm.kind))
    off = np.array([cop_offset[0], cop_offset[1], 0.0])
    p_contact = p[-1] + R[-1] @ (cfg.foot.contact_point + off)
    return FK(q, Rb, p, R, axis, beam_end, mass_points, p_contact, off, beam_start)


def foot_jacobian(fk: FK) -> np.ndarray:
    """6xn world-frame Jacobian [v; w] = J qdot at the foot contact point."""
    return fk.jacobian_point(fk.p_contact, len(fk.q) - 1)


def jacobian_numeric(cfg: Config, q, pelvis_pitch=0.0, pelvis_roll=0.0, cop_offset=(0, 0),
                     eps: float = 1e-6) -> np.ndarray:
    """Finite-difference version of ``foot_jacobian`` (central differences; the rotation part from the relative
    rotation of the last frame).  Used by the tests as an independent check of the analytic Jacobian."""
    q = np.asarray(q, float)
    n = len(q)
    J = np.zeros((6, n))
    for i in range(n):
        dq = np.zeros(n)
        dq[i] = eps
        a = forward(cfg, q + dq, pelvis_pitch, pelvis_roll, cop_offset)
        b = forward(cfg, q - dq, pelvis_pitch, pelvis_roll, cop_offset)
        J[:3, i] = (a.p_contact - b.p_contact) / (2 * eps)
        dR = a.R[-1] @ b.R[-1].T  # ~ exp([w] * 2 eps)
        J[3:, i] = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / (4 * eps)
    return J


def singularity_report(cfg: Config, fk: FK, J: np.ndarray) -> dict:
    """How close the leg is to fully straight: ``reach_ratio`` = |p_contact - p_J3| / (sum of the straight segments
    J3 -> knee -> ankle -> contact), measured from the hip-rotation joint so the lateral hip offsets do not count as leg
    length.  1.0 means fully extended (knee torque ~ 0, and the foot cannot move along the leg).  Informational only:
    standing straight is a normal case, so no warning is raised.

    (The foot-Jacobian condition number was dropped: its rows mix metres and radians, so the number had no clear meaning
    and the straight-leg case it detected is already visible in the reach ratio.)
    """
    i0 = min(2, len(fk.p) - 1)
    pts = np.vstack([fk.p[i0:], fk.p_contact])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
    ratio = float(np.linalg.norm(fk.p_contact - fk.p[i0]) / seg) if seg > 1e-9 else 1.0
    return {"reach_ratio": ratio}


def sole_corners(cfg: Config, fk: FK) -> np.ndarray:
    """4x3 world corners of the rectangular sole (heel-right, heel-left, toe-left, toe-right), for drawing and tests."""
    f = cfg.foot
    z = f.contact_point[2]
    local = np.array([[f.heel_x, -f.half_width, z], [f.heel_x, f.half_width, z],
                      [f.toe_x, f.half_width, z], [f.toe_x, -f.half_width, z]])
    return fk.p[-1] + local @ fk.R[-1].T


def foot_box_vertices(cfg: Config, fk: FK) -> np.ndarray:
    """8x3 world vertices of the foot plate: the 4 sole corners (see ``sole_corners``) then the 4 above them."""
    f = cfg.foot
    z = f.contact_point[2]
    xy = [(f.heel_x, -f.half_width), (f.heel_x, f.half_width), (f.toe_x, f.half_width), (f.toe_x, -f.half_width)]
    local = np.array([(x, y, z) for x, y in xy] + [(x, y, z + f.thickness) for x, y in xy])
    return fk.p[-1] + local @ fk.R[-1].T


def foot_mount_top(cfg: Config, fk: FK) -> np.ndarray:
    """World point on the top face of the foot plate directly under the ankle roll axis (end of the mount post)."""
    f = cfg.foot
    return fk.p[-1] + fk.R[-1] @ np.array([f.contact_point[0], f.contact_point[1], f.contact_point[2] + f.thickness])
