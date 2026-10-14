"""Ankle-rod forces: the parallel ankle's load path (G).

The ankle torque does not pass through the lower shank as a bending moment.  Two motor-driven levers on the shank push or
pull two rods that act on the foot, so the rods are the real path from the foot to the shank.  The model must therefore
treat each rod as a two-force member and include its force in every free-body cut that the rod crosses.

Rod force.  A lever of length ``l`` on the motor output shaft (pivot ``A``, angle ``th`` about +y) carries the rod at its
tip ``T = A + l (cos th, 0, -sin th)``.  With ``u`` the unit vector from the foot pickup ``P`` to the tip, a rod in tension
``F > 0`` pulls the foot toward the tip with ``F u``.  Moment balance of the (massless) lever about its shaft gives

    f_act = F [(T - A) x u]_y          ->     F = f_act / arm,     arm = l (-sin th u_x - cos th u_z)

where ``f_act`` is the actuator torque already computed from ``T(q)^-T tau`` (actuators.py).  The foot then receives ``F u``
at the pickup, which by construction reproduces the ankle joint torques (checked in the tests).

What a cut sees.  A rod is internal to a free body that contains both of its ends.  It is an *external* force ``F u`` on a
free body that contains the foot end but not the lever end, i.e. for a cut in the lower shank below a lever pivot, in the
ankle cross, in the foot between the ankle and the pickup, and for the J5 / J6 bearing reactions.  ``structure._wrench``
applies exactly that rule.

Simplifications (also in the README): rods are ideal two-force members (ball joints, no end moments, no rod mass in the load
path), the lever is massless and rigid, the force enters the shank at the pivot axis (the motor mount is not resolved), and
no friction or backlash.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import linkages
from .config import Config
from .kinematics import FK, rot_axis

ANKLE_FUNCTION = "leg.linkages:ankle_parallel_jacobian"


@dataclass
class RodLoad:
    """One lever + rod of the parallel ankle at the current pose, in world coordinates."""
    name: str                 # name of the driving actuator
    actuator: int             # its index in ``cfg.actuators``
    lever_angle: float        # rad, about +y of the shank frame
    force: float              # N along the rod; tension positive (the rod pulls the foot toward the lever)
    arm: float                # m, effective lever arm [(T - A) x u]_y (signed)
    length: float             # m, rod length (pickup to lever tip)
    pivot: np.ndarray         # lever pivot A
    tip: np.ndarray           # lever tip (rod's upper end)
    pickup: np.ndarray        # foot pickup (rod's lower end)
    u: np.ndarray             # unit vector pickup -> tip
    mount_link: int           # link the lever pivot is mounted on (the shank)
    foot_link: int            # link the pickup is on (the foot)

    @property
    def force_on_foot(self) -> np.ndarray:
        """Force the rod exerts on the foot at the pickup [N, world]."""
        return self.force * self.u


def ankle_spec(cfg: Config):
    """The ankle linkage transmission group, or None if the config has no such group."""
    for t in cfg.transmissions:
        if t.kind == "function" and t.function == ANKLE_FUNCTION:
            return t
    return None


def ankle_rod_loads(cfg: Config, fk: FK, f_act) -> list[RodLoad] | None:
    """Rod forces for the actuator loads ``f_act`` (N*m, one per actuator in ``cfg.actuators`` order).

    Returns None when the config has no ankle linkage group.  Raises ``ValueError`` when the linkage cannot be solved at
    this pose (the caller reports that and falls back to the cut method without rods).
    """
    spec = ankle_spec(cfg)
    if spec is None:
        return None
    p = spec.params
    j5, j6 = spec.joints
    mount = j5 - 1                                      # the lever pivots are mounted on the link proximal to J5
    q2 = fk.q[[j5, j6]]
    th = linkages.solve_ankle_actuator_angles(q2, **p)
    # shank frame at J5 before the ankle rotates -> world
    Rj0 = fk.R[mount] @ cfg.joints[j5].origin_R
    o = fk.p[j5]
    R5 = rot_axis(np.asarray(p["ankle_pitch_axis"], float), q2[0])
    R6 = R5 @ rot_axis(np.asarray(p["ankle_roll_axis"], float), q2[1])
    p6 = R5 @ np.asarray(p["j5_to_j6"], float)
    l = p["lever_length"]
    out = []
    for i, a_idx in enumerate(spec.actuators):
        A = np.array([0.0, p["lever_y"][i], p["actuator_axis_z"][i]])
        tip = A + l * np.array([np.cos(th[i]), 0.0, -np.sin(th[i])])
        P = p6 + R6 @ np.asarray(p["pickup_foot"][i], float)
        d = tip - P
        L = float(np.linalg.norm(d))
        u = d / L
        arm = l * (-np.sin(th[i]) * u[0] - np.cos(th[i]) * u[2])
        if abs(arm) < 1e-6:
            raise ValueError(f"ankle rod {i} is at a dead-centre (lever and rod aligned): rod force is unbounded")
        out.append(RodLoad(cfg.actuators[a_idx].name, a_idx, float(th[i]), float(f_act[a_idx] / arm), float(arm), L,
                           o + Rj0 @ A, o + Rj0 @ tip, o + Rj0 @ P, Rj0 @ u, mount, j6))
    return out
