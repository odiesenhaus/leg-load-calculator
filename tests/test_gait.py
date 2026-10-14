import copy

import numpy as np
import pytest

from leg import ConfigError, Inputs, Load, analyze, gait, load_config
from conftest import ROOT


def test_sample_hits_keyframes_and_interpolates(template_cfg):
    g = template_cfg.gaits["walking"]
    for k in g.keys:
        q, f, cop = gait.sample(g, k.phase)
        assert np.allclose(q, k.q) and np.allclose(f, k.force_bw) and cop == pytest.approx(k.cop_x)
    a, b = g.keys[2], g.keys[3]
    mid = 0.5 * (a.phase + b.phase)
    q, f, cop = gait.sample(g, mid)
    assert np.allclose(q, 0.5 * (a.q + b.q)) and f[2] == pytest.approx(0.5 * (a.force_bw[2] + b.force_bw[2]))
    assert np.allclose(gait.sample(g, -1)[0], g.keys[0].q) and np.allclose(gait.sample(g, 2)[0], g.keys[-1].q)  # clamped
    assert gait.nearest_name(g, 0.02) == "Heel strike"


def test_force_reference_walking_1bw_by_150ms(template_cfg):
    w = template_cfg.gaits["walking"]
    assert gait.sample(w, 0.150 / w.stance_time)[1][2] == pytest.approx(1.0)       # ~1 BW within 150 ms of heel strike
    assert max(k.force_bw[2] for k in w.keys) == pytest.approx(1.0)
    fz = [k.force_bw[2] for k in w.keys]
    assert w.keys[0].force_bw[0] < 0 < w.keys[-2].force_bw[0]                        # braking then propulsive
    assert w.keys[0].cop_x < w.keys[-1].cop_x and fz[0] < 0.5 and fz[-1] < 0.1       # CoP rolls heel to toe, loads start/end low
    assert 0.7 < w.keys[3].force_bw[2] < 0.9                                         # walking mid-stance valley


def test_presets_and_validation():
    cfg = load_config(ROOT / "leg_config.yaml")
    assert cfg.presets["walking"].motion == {"type": "gait", "gait": "walking", "phase": 0.25}
    import yaml
    raw = yaml.safe_load((ROOT / "leg_config.yaml").read_text(encoding="utf-8"))
    bad = copy.deepcopy(raw)
    bad["gaits"]["walking"]["keyframes"][-1]["phase"] = 0.9                          # must end at 1
    with pytest.raises(ConfigError, match="strictly increasing"):
        load_config(bad)
    bad = copy.deepcopy(raw)
    bad["presets"]["walking"]["motion"]["gait"] = "no_such_gait"
    with pytest.raises(ConfigError, match="unknown gait"):
        load_config(bad)


def test_sweep_matches_a_direct_analysis_and_joint_rates_integrate(template_cfg):
    cfg = template_cfg
    g = cfg.gaits["walking"]
    sw = gait.stance_sweep(cfg, g, n=11, stance_time=0.6)
    i = 5
    q, f, cop = gait.sample(g, sw["phase"][i])
    r = analyze(cfg, Inputs(q=q, load=Load(f * sw["weight"]), cop_offset=(cop, 0.0)))
    assert np.allclose(sw["tau"][i], r.static.tau) and np.allclose(sw["act"][i], [a.value for a in r.actuators])
    t = sw["phase"] * sw["stance_time"]
    dq = np.trapezoid(sw["qd_deg"], t, axis=0)                                        # integral of the rates = angle change
    assert np.allclose(dq, sw["q"][-1] - sw["q"][0], atol=3.0)


def test_summary_rms_and_ratings(template_cfg):
    cfg = template_cfg
    g = cfg.gaits["walking"]
    n = 9
    sw = {"act": np.full((n, len(cfg.actuators)), 10.0), "speed": np.full((n, len(cfg.actuators)), 2.0)}
    rows = gait.summarize(cfg, g, sw)
    a = cfg.actuators[4]
    r = rows[4]
    assert r["rms_stance"] == pytest.approx(10.0) and r["peak_power"] == pytest.approx(20.0)
    assert r["rms_cycle_pct"] == pytest.approx(10.0 * np.sqrt(g.stance_fraction) / a.continuous * 100)
    assert r["peak_pct"] == pytest.approx(10.0 / a.peak * 100)
    assert not r["over_envelope"]                                                      # 10 N*m at 2 rad/s is inside the envelope
    sw["act"][:] = 29.0
    assert gait.summarize(cfg, g, sw)[4]["over_envelope"]                              # 29 N*m at 2 rad/s is beyond 30*(1-2/11.62)
