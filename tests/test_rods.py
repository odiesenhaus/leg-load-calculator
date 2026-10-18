"""Ankle rods in the load path (leg/rods.py, structure._rod_active).  All checks use the real template config."""
import numpy as np
import pytest

from leg import Inputs, Load, analyze, kinematics, rods, structure
from leg.rods import ankle_rod_loads

ANKLE = (4, 5)


def _pose(rng, cfg):
    """Random pose with the ankle inside its limits and a random ground wrench."""
    lo, hi = cfg.joints[4].limits, cfg.joints[5].limits
    q = rng.uniform(-0.5, 0.5, 6)
    q[1:3] *= 0.3
    q[3] = rng.uniform(0.0, 1.0)
    q[4] = rng.uniform(lo[0], hi[1] * 0.9)
    q[5] = rng.uniform(-0.3, 0.3)
    load = Load(np.array([rng.uniform(-80, 80), rng.uniform(-60, 60), rng.uniform(100, 700)]), rng.uniform(-2, 2, 3))
    return q, load, (rng.uniform(-0.05, 0.15), rng.uniform(-0.02, 0.02))


def test_rod_forces_reproduce_the_ankle_joint_torques(template_cfg, rng):
    cfg = template_cfg
    for _ in range(12):
        q, load, cop = _pose(rng, cfg)
        r = analyze(cfg, Inputs(q=q, load=load, cop_offset=cop))
        assert r.rods is not None and not r.errors
        fk = r.fk
        for j in ANKLE:                                        # moment of the rod forces about each ankle axis = tau
            M = sum(np.cross(rd.pickup - fk.p[j], rd.force_on_foot) for rd in r.rods)
            assert M @ fk.axis[j] == pytest.approx(r.static.tau[j], abs=1e-6)


def test_lever_balance_and_hand_derived_arm_at_neutral(template_cfg):
    cfg = template_cfg
    fk = kinematics.forward(cfg, np.zeros(6))
    f_act = np.zeros(len(cfg.actuators))
    f_act[cfg.transmissions[-1].actuators] = [3.0, -2.0]
    rl = ankle_rod_loads(cfg, fk, f_act)
    P = cfg.transmissions[-1].params
    for i, rd in enumerate(rl):
        # neutral: lever at its neutral angle phi (4.9 deg up), rod from the pickup (83.7 mm ahead of J6, 17.5 mm below
        # J5) to the lever tip.  The arm is the lever's lever-arm about the rod line: l (-sin phi u_x - cos phi u_z).
        phi = rd.lever_angle
        assert phi == pytest.approx(P["neutral_angle"], abs=np.radians(0.1))
        l, bx = P["lever_length"], P["pickup_foot"][i][0]
        dx = l * np.cos(phi) - bx
        dz = P["actuator_axis_z"][i] - l * np.sin(phi) + 0.0175
        L = np.hypot(dx, dz)
        assert rd.length == pytest.approx(P["rod_length"][i], abs=1e-6) and rd.length == pytest.approx(L, abs=1e-6)
        assert rd.arm == pytest.approx(l * (-np.sin(phi) * dx / L - np.cos(phi) * dz / L), rel=1e-6)
        assert rd.force * rd.arm == pytest.approx(f_act[rd.actuator])         # f = F * arm
        # moment of the rod's pull on the lever about the shaft (+y) is minus the motor torque
        lever_force = -rd.force * rd.u
        mom_y = np.cross(rd.tip - rd.pivot, lever_force)[1]
        assert mom_y == pytest.approx(-f_act[rd.actuator], rel=1e-6)


def test_pickups_match_the_foot_frame(template_cfg, rng):
    cfg = template_cfg
    P = cfg.transmissions[-1].params
    for _ in range(5):
        q, load, cop = _pose(rng, cfg)
        r = analyze(cfg, Inputs(q=q, load=load, cop_offset=cop))
        for i, rd in enumerate(r.rods):
            assert np.allclose(rd.pickup, r.fk.p[5] + r.fk.R[5] @ np.array(P["pickup_foot"][i]), atol=1e-9)


def test_rods_do_not_change_anything_they_do_not_cross(template_cfg, rng):
    """Whole-leg equilibrium is unchanged (rods are internal), and so is every cut above both lever pivots."""
    cfg = template_cfg
    for _ in range(6):
        q, load, cop = _pose(rng, cfg)
        r = analyze(cfg, Inputs(q=q, load=load, cop_offset=cop))
        fk = r.fk
        plain = structure.link_loads(cfg, fk, load, rods=None)
        plain_fl = structure.flange_reactions(cfg, fk, load, None)
        for k in range(4):                                     # hip links and the top of the shank: identical flanges
            assert np.allclose(r.flanges[k].force_local, plain_fl[k].force_local)
            assert np.allclose(r.flanges[k].moment_local, plain_fl[k].moment_local)
        for k in range(3):
            for a, b in zip(plain[k].stations, r.links[k].stations):
                assert (a.N, a.My, a.Mz, a.T) == pytest.approx((b.N, b.My, b.Mz, b.T))
        B, _L = structure.beam_frame(cfg, fk, 3)
        s_top = min((rd.pivot - fk.beam_start[3]) @ B[:, 0] for rd in r.rods)
        for a, b in zip(plain[3].stations, r.links[3].stations):
            if a.s <= s_top - 1e-6:
                assert (a.N, a.Vy, a.Vz, a.My, a.Mz, a.T) == pytest.approx((b.N, b.Vy, b.Vz, b.My, b.Mz, b.T))


def test_ankle_bearings_carry_the_rod_force_and_no_axis_moment(template_cfg, rng):
    cfg = template_cfg
    for _ in range(6):
        q, load, cop = _pose(rng, cfg)
        r = analyze(cfg, Inputs(q=q, load=load, cop_offset=cop))
        fk = r.fk
        for j in ANKLE:                                        # the rods now carry the torque: nothing about the axis
            assert r.flanges[j].axis_moment == pytest.approx(0.0, abs=1e-6)
        # J6 bearing: free body = foot; force = GRF + foot weights + rods
        foot_w = sum(m.mass for m in fk.mass_points if m.link == 5) * np.array([0, 0, -cfg.gravity])
        Fsum = load.force + foot_w + sum(rd.force_on_foot for rd in r.rods)
        assert np.allclose(r.flanges[5].force_local, fk.R[5].T @ Fsum)


def test_lower_shank_cut_matches_a_hand_computed_free_body(template_cfg):
    """Below both pivots the rods add exactly sum (pickup - c) x F u to the plain cut."""
    cfg = template_cfg
    q = np.radians([10, 0, 0, 20, 5, 3])
    load = Load(np.array([30.0, 5.0, 392.0]))
    r = analyze(cfg, Inputs(q=q, load=load, cop_offset=(0.08, 0.0)))
    fk = r.fk
    plain = structure.link_loads(cfg, fk, load, rods=None)[3]
    B, L = structure.beam_frame(cfg, fk, 3)
    s_low = max((rd.pivot - fk.beam_start[3]) @ B[:, 0] for rd in r.rods)
    for a, b in zip(plain.stations, r.links[3].stations):
        if a.s > s_low + 1e-6:
            c = fk.beam_start[3] + a.s * B[:, 0]
            dF = sum(rd.force_on_foot for rd in r.rods)
            dM = sum(np.cross(rd.pickup - c, rd.force_on_foot) for rd in r.rods)
            assert b.N - a.N == pytest.approx(B[:, 0] @ dF, abs=1e-6)
            assert b.My - a.My == pytest.approx((B.T @ dM)[1], abs=1e-6)
            assert b.Mz - a.Mz == pytest.approx((B.T @ dM)[2], abs=1e-6)


def test_walking_push_off_rods_and_loads(template_cfg):
    """Headline case (walking push-off, 1 BW, CoP 110 mm ahead): the rods push the foot down with about
    ankle torque / pickup distance, and the J5 bearing carries what is left after the ground force."""
    from leg import gait
    cfg = template_cfg
    g = cfg.gaits["walking"]
    q, f, cop = gait.sample(g, 0.75)
    W = cfg.robot_total_mass * cfg.gravity
    r = analyze(cfg, Inputs(q=q, load=Load(f * W), cop_offset=(cop, 0.0)))
    bx = cfg.transmissions[-1].params["pickup_foot"][0][0]
    total = sum(rd.force for rd in r.rods)
    assert all(rd.force < 0 for rd in r.rods)                               # compression: pushing the toe down
    assert -total == pytest.approx(abs(r.static.tau[4]) / bx, rel=0.05)    # ~ tau / pickup distance (rods nearly vertical)
    assert -total == pytest.approx(530, rel=0.05)                           # 2 x ~265 N with the 83.7 mm pickups
    # foot free body: rods push down ~530 N at the pickups, ground pushes up ~390 N, the J5 bearing takes the difference
    assert 100 < np.linalg.norm(r.flanges[4].force_local) < 250
    assert r.links[4].peak_bending.M_bend < 10                              # ankle cross: was 45 N*m without the rods
