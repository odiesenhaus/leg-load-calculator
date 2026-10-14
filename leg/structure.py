"""D. Per-link internal loads (free-body cuts), stresses and joint-bearing reactions.

Method.  Every link has a straight "beam" (joint origin -> next joint origin, or an explicit override).  At a station
``s`` along it the leg is cut by a plane normal to the beam.  The resultant on the *proximal* face (outward normal
+x_b along the beam) equals the sum of the external wrenches on everything DISTAL to the cut:

    F_cut = F_grf + sum_distal w_m
    M_cut = M_grf + (p_contact - c) x F_grf + sum_distal (p_m - c) x w_m             (c = the cut point)

resolved in the beam frame (x_b along the beam, y_b from the section's ``y_ref``, z_b = x_b x y_b):

    N = F.x_b (tension +),  Vy = F.y_b,  Vz = F.z_b,  T = M.x_b,  My = M.y_b,  Mz = M.z_b.

There are no supports or stiffnesses: it is pure statics, equivalent to a cantilever clamped at the proximal joint and
loaded by everything beyond.  Internal joint torques (the actuators) never appear because they act between two bodies
that are both distal to the cut, so they cancel; they are balanced by the external loads, which is what the cut sums.

Which masses are distal: every mass of a later link; for the link itself, point masses (link COM, lumped actuators, rods)
whose beam coordinate is >= the station's (the joint-interface station s = 0 takes the whole link).  Self-weight is
therefore lumped at the COM and the actuator positions: moments are piecewise linear with kinks there, so stations are
also added at every lumped mass and section change to be sure the true peak is not missed.

Stress.  ``sigma = Kt (|N|/A + |My| c_z / Iy + |Mz| c_y / Iz)`` from the section at the station (the weaker neighbour at
a boundary); safety factor = yield / sigma.  Shear and torsion are reported but NOT in the stress.

Ankle rods.  The ankle torque reaches the shank through the two lever + rod linkages, not as a bending moment in the
lower shank.  Each rod is a two-force member (``rods.RodLoad``); its force ``F u`` is added to a cut whenever the rod crosses
it, i.e. the foot end is distal and the lever end is proximal (lower shank below a lever pivot, ankle cross, foot between
the ankle and the pickups, and the J5 / J6 bearing reactions).  Rods internal to the free body (cuts above both pivots)
add nothing.  Pass ``rods=None`` to get the plain cut method (e.g. when the linkage cannot be solved at a pose).

KNOWN LIMITS (also in the README): rods are ideal two-force members (no end moments), the motor mount is not resolved
(the rod force enters the shank at the lever pivot axis), and stress is not evaluated in the levers, rods or their mounts.
The foot load acts on the last link only and is distal to a cut only if the cut is on the root side of the contact point
along the beam.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .kinematics import FK
from .rods import RodLoad
from .statics import Load, gravity_vec


@dataclass
class Station:
    """Internal loads at one cut, in the beam frame."""
    s: float            # distance from the beam start [m]
    even: bool          # one of the evenly spaced stations (False: an extra station at a mass or section change)
    N: float            # axial force, tension positive [N]
    Vy: float           # shear [N]
    Vz: float
    T: float            # torsion [N*m]
    My: float           # bending moments [N*m]
    Mz: float
    M_bend: float       # sqrt(My^2 + Mz^2)
    sigma: float | None  # Kt (|N|/A + |My| cz/Iy + |Mz| cy/Iz) [Pa]; None if the link has no section
    sf: float | None    # yield / sigma


@dataclass
class LinkResult:
    index: int
    name: str
    length: float                  # beam length [m]
    frame_world: np.ndarray        # 3x3, columns x_b, y_b, z_b in world coordinates
    stations: list[Station]
    peak_bending: Station          # station with the largest bending moment magnitude
    peak_stress: Station | None    # station with the largest stress (None without a section)
    min_sf: float | None
    two_force: bool
    two_force_violation: str | None  # set when a two-force member carries transverse load (design error)


@dataclass
class FlangeReaction:
    """Wrench across a joint at its origin, in the joint frame: the load on the output flange / bearings."""
    joint: str
    force_local: np.ndarray    # [N]
    moment_local: np.ndarray   # [N*m]
    axial_force: float         # along the joint axis (thrust)
    radial_force: float        # perpendicular to the axis
    tilt_moment: float         # moment perpendicular to the axis (what the bearing sees)
    axis_moment: float         # moment about the axis = - the actuator torque


def _rod_active(fk: FK, r: RodLoad, k: int, x_b: np.ndarray, s: float, whole_link: bool) -> bool:
    """True if rod ``r`` crosses a cut on link ``k``: its foot end is distal and its lever end is proximal."""
    if r.foot_link > k:
        foot_distal = True
    elif r.foot_link == k:
        foot_distal = whole_link or (r.pickup - fk.beam_start[k]) @ x_b >= s - 1e-9
    else:
        foot_distal = False
    if r.mount_link < k:
        lever_proximal = True
    elif r.mount_link == k:
        lever_proximal = (not whole_link) and (r.pivot - fk.beam_start[k]) @ x_b < s - 1e-9
    else:
        lever_proximal = False
    return foot_distal and lever_proximal


def _wrench(cfg: Config, fk: FK, load: Load, k: int, c: np.ndarray, x_b: np.ndarray, s: float,
            whole_link: bool, rods: list[RodLoad] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Force and moment (about the cut point ``c``) of everything distal to a cut on link ``k``.

    ``s`` is the cut's beam coordinate, ``x_b`` the beam direction; ``whole_link`` makes every mass of link ``k`` distal
    (joint-interface cut).  Weights enter ONLY here (``w = m g``): this is where inertial forces would be added for
    inverse dynamics (``w = m (g - a)``).  ``rods`` adds the ankle-rod forces that cross the cut (see the module docstring).
    """
    g = gravity_vec(cfg)
    # The foot load acts on the last link only, at the contact point.  It is distal to a cut only if the cut is on the
    # root side of where it acts along the beam (matters when the beam is a plate running past the centre of pressure).
    grf = 1.0
    if k == len(fk.q) - 1 and not whole_link and (fk.p_contact - fk.beam_start[k]) @ x_b < s - 1e-9:
        grf = 0.0
    F = grf * load.force
    M = grf * (load.moment + np.cross(fk.p_contact - c, load.force))
    for m in fk.mass_points:
        if m.link > k or (m.link == k and (whole_link or (m.p - fk.beam_start[k]) @ x_b >= s - 1e-9)):
            w = m.mass * g
            F += w
            M += np.cross(m.p - c, w)
    for r in rods or ():
        if _rod_active(fk, r, k, x_b, s, whole_link):
            f = r.force_on_foot
            F += f
            M += np.cross(r.pickup - c, f)
    return F, M


def beam_frame(cfg: Config, fk: FK, k: int) -> tuple[np.ndarray, float]:
    """Beam frame of link ``k`` (columns x_b, y_b, z_b in world coordinates) and the beam length.

    x_b runs along the beam (the link-frame x axis for a zero-length beam).  y_b is the section's ``y_ref`` (default: the
    link-frame y axis) made perpendicular to x_b, falling back to the link z axis if it is parallel; z_b = x_b x y_b.
    """
    d = fk.beam_end[k] - fk.beam_start[k]
    L = float(np.linalg.norm(d))
    x = d / L if L > 1e-9 else fk.R[k][:, 0]
    sec = cfg.joints[k].link.section
    y_ref = fk.R[k] @ (sec.y_ref if sec is not None else np.array([0.0, 1.0, 0.0]))
    y = y_ref - (y_ref @ x) * x
    if np.linalg.norm(y) < 1e-6:
        y = fk.R[k][:, 2] - (fk.R[k][:, 2] @ x) * x
    y = y / np.linalg.norm(y)
    return np.column_stack([x, y, np.cross(x, y)]), L


def link_loads(cfg: Config, fk: FK, load: Load, n_stations: int | None = None,
               rods: list[RodLoad] | None = None) -> list[LinkResult]:
    """Internal loads, peak bending moment, peak stress and safety factor for every link (see the module docstring)."""
    n_st = n_stations or cfg.n_stations
    results = []
    for k, j in enumerate(cfg.joints):
        lk = j.link
        B, L = beam_frame(cfg, fk, k)
        x_b = B[:, 0]
        even = np.linspace(0.0, L, n_st) if L > 1e-9 else np.array([0.0])
        # extra stations: at every lumped mass and section change (kinks of the moment diagram / jumps of the stress)
        kinks = [t for m in fk.mass_points if m.link == k
                 for t in [(m.p - fk.beam_start[k]) @ x_b] if 1e-9 < t < L - 1e-9]
        kinks += [b_ for (s0_, s1_, _) in lk.segments for b_ in (s0_, s1_) if 1e-9 < b_ < L - 1e-9]
        for r in rods or ():                   # rod force enters at the lever pivot (shank) and leaves at the pickup (foot)
            for link_i, pt in ((r.mount_link, r.pivot), (r.foot_link, r.pickup)):
                t = (pt - fk.beam_start[k]) @ x_b
                if link_i == k and 1e-9 < t < L - 1e-9:
                    kinks.append(float(t))
        kinks = sorted(set(round(t, 9) for t in kinks))
        stations_s = sorted([(float(s), True) for s in even] + [(float(t), False) for t in kinks])
        segs = lk.segments
        sts = []
        for s, is_even in stations_s:
            c = fk.beam_start[k] + s * x_b
            F, M = _wrench(cfg, fk, load, k, c, x_b, s, whole_link=(s == 0.0), rods=rods)
            Fl, Ml = B.T @ F, B.T @ M
            sigma = sf = None
            # section(s) at this station; exactly on a boundary both neighbours apply and the weaker one governs
            cands = [sg for (s0_, s1_, sg) in segs if s0_ - 1e-9 <= s <= s1_ + 1e-9]
            if cands:
                sigmas = [sg.kt * (abs(Fl[0]) / sg.A + abs(Ml[1]) * sg.cz / sg.Iy + abs(Ml[2]) * sg.cy / sg.Iz)
                          for sg in cands]
                kk = int(np.argmax(sigmas))
                sigma = sigmas[kk]
                sf = cands[kk].yield_strength / sigma if sigma > 0 else float("inf")
            sts.append(Station(s, is_even, Fl[0], Fl[1], Fl[2], Ml[0], Ml[1], Ml[2],
                               float(np.hypot(Ml[1], Ml[2])), sigma, sf))
        pk_b = max(sts, key=lambda t: t.M_bend)
        pk_s = max(sts, key=lambda t: t.sigma) if segs else None
        viol = None
        if lk.two_force:                       # axial-only member: transverse load means the load path is wrong
            n_max = max(abs(t.N) for t in sts)
            v_max = max(np.hypot(t.Vy, t.Vz) for t in sts)
            m_max = max(t.M_bend for t in sts)
            if v_max > 0.05 * n_max + 1e-6 or m_max > 0.05 * n_max * L + 1e-6:
                viol = (f"two-force member '{lk.name}' carries transverse load "
                        f"(V={v_max:.1f} N, M={m_max:.2f} N*m vs axial {n_max:.1f} N): design error "
                        "-- check pin locations / load path")
        results.append(LinkResult(k, lk.name, L, B, sts, pk_b, pk_s,
                                  None if pk_s is None else pk_s.sf, lk.two_force, viol))
    return results


def flange_reactions(cfg: Config, fk: FK, load: Load, rods: list[RodLoad] | None = None) -> list[FlangeReaction]:
    """Load on each joint's output flange / bearings: the wrench across joint k at its origin, in the joint frame.

    ``axis_moment`` is the moment about the joint axis (= minus the actuator torque; zero for the two ankle joints, where
    the torque is carried explicitly by the rod forces); ``tilt_moment`` is the part
    perpendicular to it, which the bearing has to carry.  Useful for bearing checks and for the static hip box (the
    J1 reaction is what the pelvis-side structure must carry).
    """
    out = []
    for k, j in enumerate(cfg.joints):
        F, M = _wrench(cfg, fk, load, k, fk.p[k], np.array([1.0, 0, 0]), 0.0, whole_link=True, rods=rods)
        Fl, Ml = fk.R[k].T @ F, fk.R[k].T @ M
        a = j.axis
        Fa = float(Fl @ a)
        out.append(FlangeReaction(j.name, Fl, Ml, Fa, float(np.linalg.norm(Fl - Fa * a)),
                                  float(np.linalg.norm(Ml - (Ml @ a) * a)), float(Ml @ a)))
    return out
