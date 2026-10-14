import numpy as np
import pytest

from leg import ConfigError, Load, kinematics, load_config, structure
from conftest import make_dict

MAT = {"yield_strength": 2.4e8, "modulus": 6.9e10}


def _sec(cfg_section):
    d = make_dict(section={5: cfg_section}, contact=(0.25, 0, 0), beam_end=(0.25, 0, 0))
    return load_config(d).joints[5].link


def test_parallel_plates_properties_by_hand():
    t, w, g = 0.003, 0.040, 0.020
    sec = _sec({"kind": "parallel_plates", "thickness": t, "depth": w, "gap": g, **MAT}).section
    d = g / 2 + t / 2
    assert sec.A == pytest.approx(2 * t * w)
    assert sec.Iy == pytest.approx(2 * t * w**3 / 12)                          # each plate bending in its own plane
    assert sec.Iz == pytest.approx(2 * (w * t**3 / 12 + t * w * d**2))         # across the gap: parallel-axis term dominates
    assert sec.cy == pytest.approx(g / 2 + t) and sec.cz == pytest.approx(w / 2)
    assert sec.Iz < sec.Iy                                                     # plates are weak across the gap


def test_i_beam_and_solid_rect_properties_by_hand():
    b, tf, tw, h = 0.030, 0.004, 0.003, 0.050
    sec = _sec({"kind": "i_beam", "flange_width": b, "flange_thickness": tf, "web_thickness": tw, "depth": h, **MAT}).section
    web = h - 2 * tf
    Iy_sum = 2 * (b * tf**3 / 12 + b * tf * ((h - tf) / 2) ** 2) + tw * web**3 / 12   # flanges (parallel axis) + web
    assert sec.Iy == pytest.approx(Iy_sum)
    assert sec.A == pytest.approx(2 * b * tf + web * tw)
    r = _sec({"kind": "rect", "width": 0.05, "height": 0.03, **MAT}).section
    assert r.A == pytest.approx(0.05 * 0.03) and r.Iy == pytest.approx(0.05 * 0.03**3 / 12) and r.Iz == pytest.approx(0.03 * 0.05**3 / 12)


def test_piecewise_sections_use_the_local_section_and_the_weaker_at_a_boundary():
    L, F = 0.25, 120.0
    weak = {"kind": "generic", "area": 1e-4, "Iy": 1e-9, "Iz": 1e-9, "J": 2e-9, "c_y": 0.01, "c_z": 0.01, **MAT}
    strong = {**weak, "Iy": 4e-9, "Iz": 4e-9}
    cfg = load_config(make_dict(contact=(L, 0, 0), beam_end=(L, 0, 0),
                                section={5: [{"from": 0.0, "to": 0.10, **weak}, {"from": 0.10, "to": "end", **strong}]}))
    fk = kinematics.forward(cfg, np.zeros(6))
    lr = structure.link_loads(cfg, fk, Load(np.array([0, 0, F])), n_stations=5)[5]
    by_s = {round(st.s, 4): st for st in lr.stations}
    assert 0.1 in by_s                                                    # a station is added at the section change
    sig = lambda s, I: F * (L - s) * 0.01 / I + F / 1e-4 * 0 * 0         # noqa: E731  (pure bending; N = 0)
    assert by_s[0.0].sigma == pytest.approx(sig(0.0, 1e-9))              # root: weak section
    assert by_s[0.1].sigma == pytest.approx(sig(0.1, 1e-9))              # boundary: weaker neighbour governs
    assert by_s[0.1875].sigma == pytest.approx(sig(0.1875, 4e-9))        # beyond the change: strong section
    assert lr.peak_stress.s == pytest.approx(0.0)


def test_section_segments_must_cover_the_link():
    sec = {"kind": "rect", "width": 0.05, "height": 0.03, **MAT}
    d = make_dict(contact=(0.25, 0, 0), beam_end=(0.25, 0, 0),
                  section={5: [{"from": 0.0, "to": 0.10, **sec}, {"from": 0.15, "to": "end", **sec}]})   # gap 0.10-0.15
    with pytest.raises(ConfigError, match="contiguous"):
        load_config(d)


def test_rect_tube_with_unequal_walls_by_hand():
    b, h, ty, tz = 0.068, 0.055, 0.013, 0.003
    sec = _sec({"kind": "rect_tube", "width": b, "height": h, "wall_y": ty, "wall_z": tz, **MAT}).section
    bi, hi = b - 2 * ty, h - 2 * tz
    assert sec.A == pytest.approx(b * h - bi * hi)
    assert sec.Iy == pytest.approx((b * h**3 - bi * hi**3) / 12) and sec.Iz == pytest.approx((h * b**3 - hi * bi**3) / 12)
    eq = _sec({"kind": "rect_tube", "width": b, "height": h, "wall": 0.003, **MAT}).section      # old keyword still works
    assert eq.A == pytest.approx(b * h - (b - 0.006) * (h - 0.006))


def test_beam_override_makes_a_horizontal_cantilever_out_of_the_foot_link():
    # foot link whose beam runs along x at the plate mid-plane while the joint-to-sole stub is vertical
    L, F, d = 0.175, 100.0, 0.12
    sec = {"kind": "rect", "width": 0.12, "height": 0.010, **MAT}
    dd = make_dict(contact=(d, 0, -0.04), section={5: sec})
    dd["joints"][5]["link"]["beam"] = {"from": [0.0, 0.0, -0.035], "to": [L, 0.0, -0.035]}
    cfg = load_config(dd)
    fk = kinematics.forward(cfg, np.zeros(6))
    lr = structure.link_loads(cfg, fk, Load(np.array([0, 0, F])), n_stations=5)[5]
    assert lr.length == pytest.approx(L)
    root = lr.stations[0]
    assert root.M_bend == pytest.approx(F * d, rel=0.05)           # F * CoP distance (tiny vertical offset ignored)
    assert lr.stations[-1].M_bend == pytest.approx(0.0, abs=1e-9)  # no load beyond the CoP
    assert root.sigma == pytest.approx(F * d * 0.005 / (0.12 * 0.010**3 / 12), rel=0.06)


def test_foot_geometry_from_template(template_cfg):
    from leg.kinematics import foot_mount_top
    cfg = template_cfg
    fk = kinematics.forward(cfg, np.zeros(6))
    assert fk.p_contact[2] - fk.p[-1][2] == pytest.approx(-0.034)             # ankle roll axis is 24 mm above a 10 mm plate
    assert foot_mount_top(cfg, fk)[2] - fk.p[-1][2] == pytest.approx(-0.024)  # the mount post ends on the plate's top face
    assert cfg.joints[0].link.section is not None                              # moving hip bracket: solid block
