import shutil

import numpy as np
import pytest

from leg import Inputs, Load, analyze, default_load, load_config, ConfigError
from leg import actuators, config as cfgmod
from conftest import ROOT, make_dict


def test_template_loads_and_flags_placeholders(template_cfg):
    assert template_cfg.n == 6 and len(template_cfg.placeholder_lines) > 0
    assert set(template_cfg.presets) >= {"stand", "deep_squat", "stair_step_up", "sit_to_stand_static", "walking"}


def test_validation_collects_readable_errors():
    d = make_dict()
    d["joints"][1]["axis"] = [0, 0, 0]
    d["joints"][2]["limits_deg"] = [10, -10]
    d["actuators"][0]["peak"] = 1.0
    d["transmissions"][3]["actuators"] = ["nope"]
    with pytest.raises(ConfigError) as e:
        load_config(d)
    msg = str(e.value)
    assert "joints[1].axis" in msg and "joints[2].limits_deg" in msg
    assert "actuators[0].peak" in msg and "transmissions[3].actuators" in msg


def test_missing_key_and_bad_type_messages():
    d = make_dict()
    del d["robot"]["total_mass"]
    d["joints"][0]["link"]["mass"] = "heavy"
    with pytest.raises(ConfigError) as e:
        load_config(d)
    assert "robot.total_mass: missing required key" in str(e.value)
    assert "joints[0].link.mass" in str(e.value)


def test_save_preset_roundtrip_keeps_comments(tmp_path):
    p = tmp_path / "c.yaml"
    shutil.copy(ROOT / "leg_config.yaml", p)
    before = p.read_text(encoding="utf-8")
    cfgmod.save_preset(p, "my_pose", [0, 10, 0, 20, 5, 0], 1.5, "single", 'Landing: "test", 2 parts, ok',
                       {"type": "gait", "gait": "walking", "phase": 0.5})
    c0 = load_config(p)
    assert c0.presets["my_pose"].description == 'Landing: "test", 2 parts, ok'
    assert c0.presets["my_pose"].motion == {"type": "gait", "gait": "walking", "phase": 0.5}
    cfgmod.save_preset(p, "my_pose", [0, 11, 0, 20, 5, 0], 2.0, "double")  # overwrite in place
    c = load_config(p)
    assert np.degrees(c.presets["my_pose"].q[1]) == pytest.approx(11) and c.presets["my_pose"].load_fraction == 2.0
    after = p.read_text(encoding="utf-8")
    assert after.count("my_pose") == 1 and after.startswith(before.split("robot:")[0])
    assert len(c.placeholder_lines) == len(load_config(ROOT / "leg_config.yaml").placeholder_lines)


# ------------------------------------------------------------- transmissions
def _cfg_with(kind, **kw):
    d = make_dict()
    d["transmissions"][3] = {"joints": ["knee_pitch"], "actuators": ["a3"], "type": kind, **kw}
    return load_config(d)


def test_ratio_and_table_and_differential():
    tau = np.array([6.0, 6.0, 6.0, 12.0, 3.0, 1.0])
    cfg = _cfg_with("ratio", ratio=6.0)
    out = actuators.actuator_loads(cfg, actuators.transmission_matrix(cfg, np.zeros(6)), tau)
    assert out[3].value == pytest.approx(2.0)                     # tau_joint / N
    cfg = _cfg_with("table", table={"q_deg": [0, 100], "ratio": [4.0, 8.0]})
    T = actuators.transmission_matrix(cfg, np.radians([0, 0, 0, 50, 0, 0]))
    assert T[3, 3] == pytest.approx(6.0)                          # interpolated pose-dependent ratio
    # differential ankle pair [[N,N],[N,-N]]: tau_joint = T^T f
    d = make_dict()
    N = 5.0
    d["transmissions"][4:] = [{"joints": ["ankle_pitch", "ankle_roll"], "actuators": ["a4", "a5"],
                               "type": "matrix", "matrix": [[N, N], [N, -N]]}]
    cfg = load_config(d)
    T = actuators.transmission_matrix(cfg, np.zeros(6))
    f = actuators.actuator_loads(cfg, T, np.array([0, 0, 0, 0, 10.0, 2.0]))
    fa = np.array([f[4].value, f[5].value])
    assert np.allclose(T[4:, 4:].T @ fa, [10.0, 2.0])


def test_function_hook_stub_reports_not_implemented():
    cfg = _cfg_with("function", function="leg.linkages:knee_linkage_ratio")
    with pytest.raises(actuators.TransmissionError, match="not implemented"):
        actuators.transmission_matrix(cfg, np.zeros(6))
    r = analyze(cfg, Inputs(q=np.zeros(6), load=default_load(cfg, 0.5)))   # surfaced as an error, no crash
    assert r.errors and r.actuators is None


def test_utilization_status_thresholds():
    cfg = load_config(make_dict())     # continuous 20, peak 60
    T = np.eye(6)
    st = [a.status for a in actuators.actuator_loads(cfg, T, np.array([5, 15, 30, -59, -61, 0.0]))]
    assert st == ["ok", "warn", "warn", "warn", "over_peak", "ok"]
    assert [a.exceeds_continuous for a in actuators.actuator_loads(cfg, T, np.array([5, 15, 30, -59, -61, 0.0]))] == \
        [False, False, True, True, True, False]


def test_two_force_member_flags_bending():
    sec = {"kind": "tube", "outer_diameter": 0.02, "wall": 0.002, "yield_strength": 2e8, "modulus": 7e10}
    d = make_dict(section={3: sec})
    d["joints"][3]["link"]["two_force"] = True
    cfg = load_config(d)
    r = analyze(cfg, Inputs(q=np.radians([0, 0, 0, 30, 0, 0]), load=Load(np.array([30.0, 0, 200.0]))))
    assert r.links[3].two_force_violation and any("two-force" in w for w in r.warnings)


def test_presets_deep_squat_is_heavier_than_stand(template_cfg):
    cfg = template_cfg
    res = {}
    for n in ("stand", "deep_squat"):
        p = cfg.presets[n]
        res[n] = analyze(cfg, Inputs(q=p.q, load=default_load(cfg, p.load_fraction)))
    assert abs(res["deep_squat"].static.tau[0]) > 20 * abs(res["stand"].static.tau[0])   # hip pitch (J1)
    assert abs(res["deep_squat"].static.tau[3]) > 20 * abs(res["stand"].static.tau[3])   # knee (J4)
    r0 = analyze(cfg, Inputs(q=np.zeros(6), load=default_load(cfg, 0.5)))
    assert r0.singularity["reach_ratio"] > 0.999 and not r0.warnings        # straight leg is reported, but is a normal case


def test_report_json_and_csv_roundtrip(template_cfg):
    import json
    cfg = template_cfg
    p = cfg.presets["stair_step_up"]
    r = analyze(cfg, Inputs(q=p.q, load=default_load(cfg, p.load_fraction)))
    from leg import report
    d = json.loads(report.to_json(cfg, r))            # must be strict-JSON serialisable
    assert set(d) >= {"inputs", "joint_torques_Nm", "actuators", "links", "warnings"}
    assert len(report.to_csv(cfg, r).strip().splitlines()) == 1 + cfg.n


def test_cop_outside_sole_warns_and_sole_matches_config(template_cfg):
    import numpy as np
    from leg import kinematics
    cfg = template_cfg
    f = cfg.foot
    fk = kinematics.forward(cfg, np.zeros(6))
    c = kinematics.sole_corners(cfg, fk)
    assert np.ptp(c[:, 0]) == pytest.approx(f.toe_x - f.heel_x) and np.ptp(c[:, 1]) == pytest.approx(2 * f.half_width)
    assert fk.p[-1][0] - c[:, 0].min() == pytest.approx(0.055)           # ankle 55 mm from the heel
    q, ld = np.zeros(6), default_load(cfg, 0.5)
    inside = analyze(cfg, Inputs(q=q, load=ld, cop_offset=(f.toe_x - 0.01, 0.0)))
    outside = analyze(cfg, Inputs(q=q, load=ld, cop_offset=(f.toe_x + 0.01, 0.0)))
    assert not any("sole" in w for w in inside.warnings) and any("sole" in w for w in outside.warnings)


def test_com_center_and_template_geometry(template_cfg):
    cfg = template_cfg
    assert cfg.joint_names == ["hip_flexion", "hip_abduction", "hip_rotation", "knee", "ankle_flexion", "ankle_inversion"]
    thigh = cfg.joints[2].link
    assert np.allclose(thigh.com, [0, 0, -0.100])                       # centre of the 200 mm J3 -> knee link
    # J1 -> J5 = 672.5 mm, J1 -> J4 = 320 mm (drawings)
    z = np.cumsum([j.origin_xyz[2] for j in cfg.joints])
    assert z[3] == pytest.approx(-0.320) and z[4] == pytest.approx(-0.6725) and z[5] == pytest.approx(-0.690)


# ------------------------------------------------------------- ankle parallel linkage
def test_ankle_linkage_neutral_jacobian_and_rod_lengths(template_cfg, rng):
    import numpy as np
    from leg import linkages
    spec = next(t for t in template_cfg.transmissions if t.kind == "function")
    P = spec.params
    T0 = linkages.ankle_parallel_jacobian(np.zeros(2), **P)
    # hand estimate at neutral: the rods are nearly vertical, so a pitch rotation lifts the pickup by (pickup x) per radian
    # and the lever tip must follow it vertically, which it does at (lever length x cos(neutral angle)) per radian;
    # roll likewise with the pickup y.  With 83.7 mm / (75 mm cos 4.9 deg) = 1.12 and 21 / 74.7 = 0.28.
    phi = P["neutral_angle"]
    reach = P["lever_length"] * np.cos(phi)
    assert np.allclose(np.abs(T0[:, 0]), P["pickup_foot"][0][0] / reach, rtol=0.03)
    assert np.allclose(T0[:, 1], [abs(P["pickup_foot"][0][1]) / reach, -abs(P["pickup_foot"][1][1]) / reach], rtol=0.03)
    assert np.all(T0[:, 0] < 0)
    # at neutral the levers sit at the configured neutral angle (4.9 deg up)
    assert np.allclose(linkages.solve_ankle_actuator_angles(np.zeros(2), **P), phi, atol=np.radians(0.1))
    for _ in range(20):
        q = np.array([rng.uniform(-0.78, 0.52), rng.uniform(-0.52, 0.52)])
        th = linkages.solve_ankle_actuator_angles(q, **P)
        R5 = kin_rot(P["ankle_pitch_axis"], q[0])
        R6 = R5 @ kin_rot(P["ankle_roll_axis"], q[1])
        for i in range(2):
            pick = R5 @ np.array(P["j5_to_j6"]) + R6 @ np.array(P["pickup_foot"][i])
            tip = np.array([0, P["lever_y"][i], P["actuator_axis_z"][i]]) + P["lever_length"] * np.array([np.cos(th[i]), 0, -np.sin(th[i])])
            assert np.linalg.norm(pick - tip) == pytest.approx(P["rod_length"][i], abs=1e-9)
        # pose-dependent: ankle load maps through T(q)^-T and the power balance holds
        Tq = linkages.ankle_parallel_jacobian(q, **P)
        tau = np.array([5.0, 2.0])
        f = np.linalg.solve(Tq.T, tau)
        assert Tq.T @ f == pytest.approx(tau)


def kin_rot(axis, a):
    import numpy as np
    from leg.kinematics import rot_axis
    return rot_axis(np.asarray(axis, float), a)


def test_ankle_linkage_is_feasible_over_the_whole_ankle_range(template_cfg):
    from leg import linkages
    cfg = template_cfg
    P = next(t for t in cfg.transmissions if t.kind == "function").params
    lo5, hi5 = cfg.joints[4].limits
    lo6, hi6 = cfg.joints[5].limits
    ang = np.array([linkages.solve_ankle_actuator_angles(np.array([qp, qr]), **P)
                    for qp in np.linspace(lo5, hi5, 13) for qr in np.linspace(lo6, hi6, 9)])      # raises if a rod cannot reach
    # given lever swing over the full range of motion: -51 .. +47 deg
    assert np.degrees(ang.min()) == pytest.approx(-51.0, abs=0.5) and np.degrees(ang.max()) == pytest.approx(47.0, abs=0.5)
    # the given rod lengths close the loop with the lever at its neutral angle and the foot flat (pickups at roll-axis
    # height, 17.5 mm below J5): |tip - pickup| with tip = pivot + l (cos phi, 0, -sin phi)
    pivot_z, lever, rods, phi = P["actuator_axis_z"], P["lever_length"], P["rod_length"], P["neutral_angle"]
    for pz, rl in zip(pivot_z, rods):
        assert rl == pytest.approx(np.hypot(lever * np.cos(phi) - P["pickup_foot"][0][0],
                                            pz - lever * np.sin(phi) + 0.0175), abs=1e-4)
    # transmission quality: the lever-rod angle never drops below 45 deg anywhere in the range (dead-centre = 0)
    from leg.kinematics import rot_axis
    worst = 90.0
    for qp in np.linspace(lo5, hi5, 13):
        for qr in np.linspace(lo6, hi6, 9):
            q = np.array([qp, qr])
            th = linkages.solve_ankle_actuator_angles(q, **P)
            R5 = rot_axis(np.asarray(P["ankle_pitch_axis"], float), qp)
            R6 = R5 @ rot_axis(np.asarray(P["ankle_roll_axis"], float), qr)
            for i in range(2):
                pick = R5 @ np.array(P["j5_to_j6"]) + R6 @ np.array(P["pickup_foot"][i])
                lev = np.array([np.cos(th[i]), 0.0, -np.sin(th[i])])
                tip = np.array([0, P["lever_y"][i], pivot_z[i]]) + lever * lev
                u = (tip - pick) / np.linalg.norm(tip - pick)
                worst = min(worst, np.degrees(np.arccos(abs(lev @ u))))
    assert worst > 44.5
