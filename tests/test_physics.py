"""Hand-checkable validation cases 1-7 from the task statement."""
import numpy as np
import pytest

from leg import Inputs, Load, analyze, load_config
from leg import actuators, dynamics, kinematics, statics, structure
from conftest import make_dict

G = 9.81


def _fk(cfg, q, **kw):
    return kinematics.forward(cfg, np.radians(q), **kw)


# ------------------------------------------------------------------ 1. straight leg
def test_1_straight_leg_vertical_force_only_gravity_torques(mk):
    # masses on the leg axis: gravity gives no torque either when the leg is vertical
    cfg = mk(masses=[0, 0, 1.0, 1.0, 0, 0.5], coms=[[0, 0, 0], [0, 0, 0], [0, 0, -0.15], [0, 0, -0.15], [0, 0, 0], [0, 0, -0.03]])
    fk = _fk(cfg, [0] * 6)
    r = statics.joint_torques(cfg, fk, Load(np.array([0, 0, 200.0])))
    assert np.allclose(r.tau_grf, 0, atol=1e-9)
    assert np.allclose(r.tau, 0, atol=1e-9)


def test_1b_inclined_straight_leg_hip_torque_by_hand(mk):
    l1, l2, foot = 0.3, 0.3, 0.07
    m1, c1, m2, c2 = 1.5, 0.12, 0.8, 0.14
    cfg = mk(thigh=l1, shank=l2, contact=(0, 0, -foot), masses=[0, 0, m1, m2, 0, 0],
             coms=[[0, 0, 0]] * 2 + [[0, 0, -c1], [0, 0, -c2], [0, 0, 0], [0, 0, 0]])
    th, Fz = np.radians(20.0), 150.0
    r = statics.joint_torques(cfg, _fk(cfg, [0, 20, 0, 0, 0, 0]), Load(np.array([0, 0, Fz])))
    L = l1 + l2 + foot
    # hand: foot is L sin(th) ahead of the hip -> GRF torque -Fz L sin(th) about the flexion axis;
    # weights at distances d_k along the leg -> +g sin(th) sum m_k d_k
    expected = -Fz * L * np.sin(th) + G * np.sin(th) * (m1 * c1 + m2 * (l1 + c2))
    assert r.tau[1] == pytest.approx(expected, rel=1e-9)
    assert r.tau[0] == pytest.approx(0, abs=1e-9)  # hip roll, planar problem


# ------------------------------------------------------------------ 2. planar crouch
def test_2_planar_two_link_matches_hand_jacobian_transpose(mk):
    l1, l2, ft = 0.30, 0.28, 0.07
    cfg = mk(thigh=l1, shank=l2, contact=(0, 0, -ft))
    h, k = np.radians(50.0), np.radians(80.0)
    Fx, Fz = 40.0, 220.0
    r = statics.joint_torques(cfg, kinematics.forward(cfg, [0, h, 0, k, 0, 0]), Load(np.array([Fx, 0, Fz])))
    # foot (ankle=0) is aligned with the shank: reach beyond the knee is L2 = l2 + ft
    L2 = l2 + ft
    dP_dh = np.array([l1 * np.cos(h) + L2 * np.cos(h - k), l1 * np.sin(h) + L2 * np.sin(h - k)])  # (x, z)
    dP_dk = np.array([-L2 * np.cos(h - k), -L2 * np.sin(h - k)])
    dP_da = np.array([ft * np.cos(h - k), ft * np.sin(h - k)])
    F = np.array([Fx, Fz])
    assert r.tau[1] == pytest.approx(-F @ dP_dh, rel=1e-9)
    assert r.tau[3] == pytest.approx(-F @ dP_dk, rel=1e-9)
    assert r.tau[4] == pytest.approx(-F @ dP_da, rel=1e-9)


# ------------------------------------------------------------------ 3. contact offset
def test_3_contact_offset_gives_F_times_d_at_ankle(mk):
    cfg = mk()
    Fz, d = 300.0, 0.025
    q = np.zeros(6)
    base = statics.joint_torques(cfg, kinematics.forward(cfg, q), Load(np.array([0, 0, Fz])))
    fx = statics.joint_torques(cfg, kinematics.forward(cfg, q, cop_offset=(d, 0)), Load(np.array([0, 0, Fz])))
    fy = statics.joint_torques(cfg, kinematics.forward(cfg, q, cop_offset=(0, d)), Load(np.array([0, 0, Fz])))
    assert fx.tau[4] - base.tau[4] == pytest.approx(-Fz * d)   # pressure ahead of ankle: dorsiflexion-axis torque
    assert fy.tau[5] - base.tau[5] == pytest.approx(-Fz * d)
    assert fx.tau[4] == pytest.approx(-Fz * d) and abs(fx.tau[5]) < 1e-9


# ------------------------------------------------------------------ 4. cantilever
def test_4_cantilever_bending_moment_is_F_times_L_and_linear(mk):
    L, F = 0.25, 120.0
    sec = {"kind": "generic", "area": 1e-4, "Iy": 1e-9, "Iz": 1e-9, "J": 2e-9, "c_y": 0.01, "c_z": 0.01,
           "yield_strength": 2e8, "modulus": 7e10}
    # foot link: horizontal beam of length L, tip load vertical
    cfg = mk(contact=(L, 0, 0), beam_end=(L, 0, 0), section={5: sec})
    fk = kinematics.forward(cfg, np.zeros(6))
    lr = structure.link_loads(cfg, fk, Load(np.array([0, 0, F])), n_stations=5)[5]
    assert lr.length == pytest.approx(L)
    for st in lr.stations:
        assert st.M_bend == pytest.approx(F * (L - st.s), rel=1e-9)
        assert abs(st.N) < 1e-9 and abs(st.Vz) == pytest.approx(F)
    assert lr.peak_bending.s == pytest.approx(0.0) and lr.peak_bending.M_bend == pytest.approx(F * L)
    # stress at the root: Kt * M c / I
    assert lr.peak_stress.sigma == pytest.approx(F * L * 0.01 / 1e-9)
    # axial tip load (pulling away from the root) is tension, no bending
    lr2 = structure.link_loads(cfg, fk, Load(np.array([F, 0, 0])), n_stations=3)[5]
    assert all(st.N == pytest.approx(F) and st.M_bend < 1e-9 for st in lr2.stations)


# ------------------------------------------------------------------ 5. free-body consistency
def test_5_free_body_balance_random_poses(rng):
    d = make_dict(thigh=0.31, shank=0.27, masses=[0.5, 0.7, 1.2, 0.9, 0.2, 0.6],
                  coms=[[0.01, 0.0, -0.02], [0, 0.01, -0.03], [0.0, 0.02, -0.14], [0.01, 0, -0.12],
                        [0, 0, 0.01], [0.03, 0, -0.04]],
                  point_masses={2: [{"name": "m", "mass": 0.9, "pos": [0, 0.03, -0.05]}]})
    d["foot"] = {"contact_point": [0.02, 0.01, -0.07]}
    cfg = load_config(d)
    g = np.array([0, 0, -G])
    for _ in range(25):
        q = rng.uniform(-1.0, 1.0, 6)
        F = rng.uniform(-100, 300, 3)
        M = rng.uniform(-5, 5, 3)
        load = Load(F, M)
        pp, pr, cop = rng.uniform(-.3, .3), rng.uniform(-.3, .3), tuple(rng.uniform(-.02, .02, 2))
        fk = kinematics.forward(cfg, q, pp, pr, cop)
        st = statics.joint_torques(cfg, fk, load)
        wr = [structure._wrench(cfg, fk, load, k, fk.p[k], np.array([1.0, 0, 0]), 0.0, True) for k in range(6)]
        # (a) pelvis reaction balances foot load + all weights (force and moment about the hip root)
        Fsum = F + sum(m.mass * g for m in fk.mass_points)
        Msum = M + np.cross(fk.p_contact - fk.p[0], F) + sum(np.cross(m.p - fk.p[0], m.mass * g) for m in fk.mass_points)
        assert np.allclose(wr[0][0], Fsum) and np.allclose(wr[0][1], Msum)
        # (b) link-by-link: wrench entering link k = wrench leaving it + its own weights
        for k in range(5):
            Fk, Mk = wr[k]
            Fn, Mn = wr[k + 1]
            own = [m for m in fk.mass_points if m.link == k]
            F_exp = Fn + sum(m.mass * g for m in own)
            M_exp = Mn + np.cross(fk.p[k + 1] - fk.p[k], Fn) + sum(np.cross(m.p - fk.p[k], m.mass * g) for m in own)
            assert np.allclose(Fk, F_exp) and np.allclose(Mk, M_exp)
        # (c) the actuator torque is minus the axis component of the transmitted moment
        for k in range(6):
            assert st.tau[k] == pytest.approx(-wr[k][1] @ fk.axis[k], abs=1e-9)
        # (d) pelvis balance via torques: force/moment on the pelvis are exactly the reaction
        assert np.allclose(analyze(cfg, Inputs(q=q, pelvis_pitch=pp, pelvis_roll=pr, load=load, cop_offset=cop)).flanges[0].force_local,
                           fk.R[0].T @ Fsum)


# ------------------------------------------------------------------ 6. joint-space inertia (the foot-impact tests were removed with the impact estimate)
def test_6c_mass_matrix_symmetric_positive_definite(template_cfg, rng):
    cfg = template_cfg
    for _ in range(10):
        q = rng.uniform(-0.8, 0.8, 6)
        fk = kinematics.forward(cfg, q)
        H = dynamics.link_mass_matrix(cfg, fk) + actuators.reflected_rotor_inertia(cfg, actuators.transmission_matrix(cfg, q))
        assert np.allclose(H, H.T) and np.linalg.eigvalsh(H).min() > 0


# ------------------------------------------------------------------ 7. Jacobian
def test_7_analytic_vs_numeric_jacobian(template_cfg, rng):
    d = make_dict()
    # skewed joint frames so the check is not just axis-aligned
    for j, rpy in zip(d["joints"], rng.uniform(-30, 30, (6, 3))):
        j["origin"]["rpy_deg"] = rpy.tolist()
        j["origin"]["xyz"] = (np.array(j["origin"]["xyz"]) + rng.uniform(-0.03, 0.03, 3)).tolist()
    for cfg in (template_cfg, load_config(d)):
        for _ in range(20):
            q = rng.uniform(-1.2, 1.2, 6)
            pp, pr, cop = rng.uniform(-.4, .4), rng.uniform(-.4, .4), rng.uniform(-.03, .03, 2)
            Ja = kinematics.foot_jacobian(kinematics.forward(cfg, q, pp, pr, cop))
            Jn = kinematics.jacobian_numeric(cfg, q, pp, pr, cop)
            assert np.allclose(Ja, Jn, atol=1e-6)
