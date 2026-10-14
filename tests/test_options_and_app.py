"""Hardware options (actuator swap) and an end-to-end smoke test of the Streamlit app.

The app test drives the real script headlessly with ``streamlit.testing`` (no browser): it applies presets, swaps the
ankle motor and selects gait keyframes, and asserts that nothing raises and the displayed numbers follow the inputs.
"""
import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from leg import ConfigError, Inputs, analyze, default_load, gait, load_config, with_actuator_choices
from conftest import ROOT


# ------------------------------------------------------------------ actuator options (pure physics layer)
def test_template_declares_the_ankle_motor_option(template_cfg):
    opt = template_cfg.actuator_options["ankle_motor"]
    assert set(opt.choices) == {"X4-36-L", "X6-50-L"} and opt.default == "X4-36-L"
    assert template_cfg.active_choices == {"ankle_motor": "X4-36-L"}


def test_swap_changes_only_the_ankle_ratings_and_weight(template_cfg):
    cfg = template_cfg
    new = with_actuator_choices(cfg, {"ankle_motor": "X6-50-L"})
    assert new is not cfg and cfg.actuators[4].peak == 30.0                    # the original is untouched
    for k in (4, 5):
        a = new.actuators[k]
        assert (a.model, a.continuous, a.peak, a.max_speed, a.rated_speed) == ("X6-50-L", 20.0, 45.0, 18.43, 16.02)
        assert a.rotor_inertia == pytest.approx(19.612**2 * 0.0019, rel=1e-3)   # datasheet 0.0019 kg m^2 x N^2
    for k in (0, 1, 2, 3):
        assert new.actuators[k] == cfg.actuators[k]                            # hip / knee actuators unchanged
    masses = {pm.name: pm.mass for j in new.joints for pm in j.link.point_masses}
    assert masses["ankle_upper_actuator"] == masses["ankle_lower_actuator"] == 0.78
    assert new.active_choices == {"ankle_motor": "X6-50-L"}
    assert with_actuator_choices(cfg).active_choices == {"ankle_motor": "X4-36-L"}   # default when nothing is selected


def test_swap_changes_the_results_in_the_expected_direction(template_cfg):
    base, new = template_cfg, with_actuator_choices(template_cfg, {"ankle_motor": "X6-50-L"})
    p = base.presets["walking"]
    args = dict(q=p.q, load=default_load(base, 1.0))
    r0, r1 = analyze(base, Inputs(**args)), analyze(new, Inputs(**args))
    assert np.allclose(r0.static.tau[:3], r1.static.tau[:3], rtol=0.05)         # torques barely move (0.84 kg heavier)
    assert r1.actuators[4].util_cont < r0.actuators[4].util_cont                # but the stronger motor is less utilised
    assert r1.actuators[4].util_cont == pytest.approx(r0.actuators[4].util_cont * 10.5 / 20.0, rel=1e-6)
    # the walking ankle that was over its peak rating with the X4-36-L fits with the X6-50-L
    g = base.gaits["walking"]
    over = lambda cfg: gait.summarize(cfg, g, gait.stance_sweep(cfg, g, n=21))[4]["peak_pct"]   # noqa: E731
    assert over(base) > 100 > over(new)


def test_unknown_choice_and_inconsistent_default_are_rejected(template_cfg):
    with pytest.raises(ValueError, match="unknown choice"):
        with_actuator_choices(template_cfg, {"ankle_motor": "nope"})
    raw = yaml.safe_load((ROOT / "leg_config.yaml").read_text(encoding="utf-8"))
    bad = copy.deepcopy(raw)
    bad["actuator_options"]["ankle_motor"]["choices"]["X4-36-L"]["peak"] = 31.0   # default no longer matches actuators:
    with pytest.raises(ConfigError, match="must match"):
        load_config(bad)
    bad = copy.deepcopy(raw)
    bad["actuator_options"]["ankle_motor"]["applies_to"] = ["not_an_actuator"]
    with pytest.raises(ConfigError, match="unknown actuator"):
        load_config(bad)


# ------------------------------------------------------------------ the whole app, headless
APP = str(Path(ROOT / "app.py"))


@pytest.fixture(scope="module")
def app():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _apply(at, preset):
    at.sidebar.selectbox(key="preset_pick").select(preset).run()
    next(b for b in at.sidebar.button if b.label == "Apply Preset").click().run()
    assert not at.exception, [e.value for e in at.exception]


def _caption(at, startswith):
    return [c.value for c in at.sidebar.caption if c.value.startswith(startswith)]


def test_app_loads_with_all_tabs_and_the_hardware_selector(app):
    assert [t.label for t in app.tabs] == ["Pose", "3D View", "Actuators", "Link Loads", "Export", "Gait Cycle"]
    assert app.sidebar.selectbox(key="opt_ankle_motor").options == ["X4-36-L", "X6-50-L"]


def test_app_gait_keyframe_buttons_drive_pose_and_load(app):
    _apply(app, "walking")
    g = load_config(ROOT / "leg_config.yaml").gaits["walking"]
    names = [o.content for o in app.sidebar.get("pills")[0].proto.options]
    assert names == [k.name for k in g.keys]                                             # one button per keyframe
    for idx in (0, 4, len(names) - 1):
        app.session_state["gait_key"] = idx
        app.run()
        assert not app.exception
        q = [app.sidebar.number_input(key=f"qn{i}").value for i in range(6)]
        assert np.allclose(np.radians(q), g.keys[idx].q, atol=1e-6)                      # joints = the keyframe exactly
        assert _caption(app, "Vertical")[0].startswith(f"Vertical {g.keys[idx].force_bw[2]:.2f} BW")
        assert app.sidebar.number_input(key="cop_x_mm").value == pytest.approx(g.keys[idx].cop_x * 1000)
        assert app.sidebar.number_input(key="fx").disabled                               # driven, not editable


def test_app_hardware_swap(app):
    _apply(app, "walking")
    assert app.session_state["gait_pick"] == "walking"                                   # the Gait Cycle tab follows
    app.sidebar.selectbox(key="opt_ankle_motor").select("X6-50-L").run()
    assert not app.exception
    assert any("does NOT fit" in c.value for c in app.sidebar.caption)                   # the geometry caveat is shown
    app.sidebar.selectbox(key="opt_ankle_motor").select("X4-36-L").run()
    assert not app.exception


def test_app_ankle_inversion_row_is_not_blamed_for_the_flexion_load(app):
    _apply(app, "stair_step_up")
    app.session_state["cop_x_mm"] = 60.0                                                 # forward CoP: ankle pitch overloads the motors
    app.session_state["frac_choice"] = "1.0"
    app.run()
    assert not app.exception
    df = next(d.value for d in app.dataframe if "Own Share [% Cont]" in getattr(d.value, "columns", []))
    flex = df[df["Joint"] == "Ankle Flexion"].iloc[0]
    inv = df[df["Joint"] == "Ankle Inversion"].iloc[0]
    assert flex["Own Share [% Cont]"] > 100 and inv["Own Share [% Cont]"] < 5          # inversion carries only its own (tiny) load
    assert inv["_status"] == "ok"


def test_app_warns_when_a_selected_preset_has_not_been_applied(app):
    _apply(app, "stand")
    app.sidebar.selectbox(key="preset_pick").select("stair_step_up").run()
    assert any(c.value.startswith("⚠") for c in app.sidebar.caption)                    # selected but not applied
    next(b for b in app.sidebar.button if b.label == "Apply Preset").click().run()
    assert not any(c.value.startswith("⚠") for c in app.sidebar.caption)


def test_app_applying_a_static_preset_clears_the_leftover_walking_load(app):
    """Regression: Fx / Fy / CoP from the last walking keyframe used to survive 'Apply Preset' on stand / stair."""
    _apply(app, "walking")
    app.session_state["gait_key"] = 4                                                    # Peak 2: Fx ~ 78 N, CoP 110 mm
    app.run()
    assert app.sidebar.number_input(key="cop_x_mm").value == pytest.approx(110.0)
    for name in ("stand", "stair_step_up", "deep_squat"):
        _apply(app, name)
        for key in ("fx", "fy", "cop_x_mm", "cop_y_mm", "mx", "my", "mz"):
            assert app.sidebar.number_input(key=key).value == 0.0, (name, key)
        assert any("only, at the point under the ankle" in c.value for c in app.sidebar.caption)
        df = next(d.value for d in app.dataframe if "% Cont" in getattr(d.value, "columns", []))
        assert df["% Cont"].iloc[4:].max() < 10, name                                    # ankle motors barely loaded


def test_app_read_only_mode_for_the_hosted_copy(monkeypatch, tmp_path):
    """A hosted copy is shared by every visitor: saving presets is disabled and Export writes nothing to disk."""
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("LEG_CALC_READ_ONLY", "1")
    before = (ROOT / "leg_config.yaml").read_text(encoding="utf-8")
    exports = ROOT / "exports"
    n_before = len(list(exports.glob("*"))) if exports.exists() else 0
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception
    save = next(b for b in at.sidebar.button if b.label == "Save Current Pose as Preset")
    assert save.disabled
    exp = next(b for b in at.button if b.label.startswith("Export Report"))
    exp.click().run()
    assert not at.exception
    assert [d.label for d in at.get("download_button")][:2] == ["Download JSON", "Download CSV"]
    assert (ROOT / "leg_config.yaml").read_text(encoding="utf-8") == before
    assert (len(list(exports.glob("*"))) if exports.exists() else 0) == n_before
