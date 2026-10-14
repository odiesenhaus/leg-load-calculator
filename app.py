"""Streamlit UI for the humanoid leg load calculator.   Run:  streamlit run app.py

All physics lives in the ``leg/`` package (no UI imports there); this file only wires widgets, tables and figures.
Streamlit was chosen over Dash because every input change simply re-runs one pure function (``leg.analyze`` takes
about 10-30 ms), so there is no callback graph to maintain.

How the script is organised (Streamlit re-runs it top to bottom on every interaction):

  1. Config            load + validate ``leg_config.yaml`` (cached), apply the selected hardware options
  2. Session state     defaults and callbacks.  Widgets that must be driven from code (joint angles, preset
                       motions, the walking keyframe) keep their value in ``st.session_state`` under a fixed key
  3. Sidebar           inputs: hardware options, preset + pose, pelvis, foot load, optional extras, save-preset
  4. Analysis          one ``leg.analyze`` call on the inputs
  5. Main area         banners and tabs (Pose, 3D View, Actuators, Link Loads, Export, Gait Cycle)

The walking preset overrides the manual inputs ("Preset Motion" on the vertical load; "Follow Walking Cycle"): it writes
the driven values into session state *before* the widgets are created and disables those widgets, so the displayed
numbers are always the ones used.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import viz
from leg import ConfigError, Inputs, Load, analyze, load_config, with_actuator_choices
from leg import config as cfgmod
from leg import gait as gait_mod, report
from leg.names import pretty

CONFIG_PATH = Path(__file__).with_name("leg_config.yaml")
EXPORT_DIR = Path(__file__).with_name("exports")
FRACTIONS = ["0.5", "1.0", "2.0", "3.0"]                       # quick "x body weight" load choices
LABELS = {"custom": "Custom", "manual": "Manual", "motion": "Preset Motion"}


def _lbl(x: str) -> str:
    """Display label for the vertical-load radio options."""
    return LABELS.get(x, pretty(x))


st.set_page_config(page_title="Leg load calculator", layout="wide")
ss = st.session_state


def _running_locally() -> bool:
    """True when the app is opened on this machine.  A hosted copy is shared by every visitor, so it must not write
    to ``leg_config.yaml`` (one visitor's saved preset would change the app for everyone) or fill ``exports/``.
    Set the environment variable LEG_CALC_READ_ONLY=1 to force read-only mode."""
    if os.environ.get("LEG_CALC_READ_ONLY"):
        return False
    try:
        host = (st.context.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
    except Exception:  # no request context (e.g. the headless tests): treat as local
        return True
    return host in ("", "localhost", "127.0.0.1", "::1") or host.endswith(".localhost")


LOCAL = _running_locally()


# =========================================================================== 1. config
@st.cache_resource
def get_config(path: str, mtime: float):
    """Parsed config, cached until the file changes (the mtime is part of the cache key)."""
    return load_config(path)


try:
    base_cfg = get_config(str(CONFIG_PATH), CONFIG_PATH.stat().st_mtime)
except ConfigError as e:
    st.error("leg_config.yaml is invalid:")
    for msg in e.errors:
        st.markdown(f"- `{msg}`")
    st.stop()
except FileNotFoundError:
    st.error(f"Config not found: {CONFIG_PATH}")
    st.stop()

# Hardware options (e.g. the ankle motor model).  The selection is read from session state *before* the widget is
# drawn (Streamlit has already applied any interaction), then every downstream use sees the swapped hardware.
cfg = with_actuator_choices(base_cfg, {k: ss.get(f"opt_{k}", o.default) for k, o in base_cfg.actuator_options.items()})
N = cfg.n


# =========================================================================== 2. session state + callbacks
def _init() -> None:
    """Defaults for every state key the script reads before its widget exists (runs once per session)."""
    if "ready" in ss:
        return
    ss.ready = True
    for i in range(N):
        ss[f"qs{i}"] = ss[f"qn{i}"] = 0.0                  # joint angle slider / number box [deg]
    ss.support, ss.frac_choice, ss.frac_custom = "double", "0.5", 0.5
    ss.fx = ss.fy = ss.fz = 0.0
    ss.mx = ss.my = ss.mz = 0.0                              # foot moment [N*m]
    ss.cop_x_mm = ss.cop_y_mm = 0.0
    ss.active_preset = None                                  # the preset last applied with the button
    ss.gait_follow = True
    ss.gait_key = 0                                          # index of the selected walking keyframe
    ss.gait_pick = next(iter(cfg.gaits)) if cfg.gaits else None


_init()

def _slider_changed(i: int) -> None:
    ss[f"qn{i}"] = ss[f"qs{i}"]


def _number_changed(i: int) -> None:
    lo, hi = np.degrees(cfg.joints[i].limits)
    ss[f"qs{i}"] = float(min(max(ss[f"qn{i}"], lo), hi))


def _set_fraction(f: float) -> None:
    """Select the quick-fraction radio if ``f`` is one of them, else switch to 'custom' with that value."""
    key = f"{f:.1f}"
    if key in FRACTIONS:
        ss.frac_choice = key
    else:
        ss.frac_choice, ss.frac_custom = "custom", float(f)


def _support_changed() -> None:
    _set_fraction(cfg.single_support_fraction if ss.support == "single" else cfg.double_support_fraction)


def _key_label(g, i: int) -> str:
    """Label of gait keyframe ``i`` for the keyframe buttons."""
    k = g.keys[i]
    return k.name or f"{k.phase * 100:.0f}%"


def _apply_preset() -> None:
    """'Apply Preset' button: load the preset's pose, support, load fraction and (if any) motion settings."""
    p = cfg.presets.get(ss.get("preset_pick"))
    if p is None:
        return
    for i, q in enumerate(np.degrees(p.q)):
        ss[f"qn{i}"] = float(q)
        _number_changed(i)
    ss.support = p.support
    # A preset defines pose + vertical load only.  Reset the horizontal force, centre of pressure and foot moment too:
    # they may still hold the last walking keyframe (the CoP boxes sit in a collapsed expander and are easy to miss).
    ss.fx = ss.fy = 0.0
    ss.cop_x_mm = ss.cop_y_mm = 0.0
    ss.mx = ss.my = ss.mz = 0.0
    _set_fraction(p.load_fraction)
    ss.active_preset = ss.get("preset_pick")
    if p.motion:                                             # walking: pick the keyframe nearest the preset
        m = p.motion
        g = cfg.gaits[m["gait"]]
        ss.gait_key = int(np.argmin([abs(k.phase - m["phase"]) for k in g.keys]))
        ss.gait_follow = True
        ss.gait_pick = m["gait"]                             # the Gait Cycle tab follows the preset
        ss.frac_choice = "motion"


# =========================================================================== 3. sidebar (inputs)
sb = st.sidebar

# ---- hardware options -------------------------------------------------------------------------------------------
if base_cfg.actuator_options:
    sb.header("Hardware")
    for _key, _opt in base_cfg.actuator_options.items():
        _names = list(_opt.choices)
        sb.selectbox(_opt.label, _names, index=_names.index(_opt.default), key=f"opt_{_key}",
                     help="Replaces the ratings, rotor inertia and weight of these actuators; geometry is not changed.")
        _note = _opt.choices[cfg.active_choices[_key]]["note"]
        if _note:
            sb.caption(_note)

# ---- preset + pose ----------------------------------------------------------------------------------------------
sb.header("Pose")
sb.selectbox("Preset", list(cfg.presets), key="preset_pick", format_func=pretty)
_picked = cfg.presets.get(ss.get("preset_pick", next(iter(cfg.presets))))
if _picked is not None and _picked.description:
    sb.caption(_picked.description)
sb.button("Apply Preset", on_click=_apply_preset)
if ss.get("preset_pick", next(iter(cfg.presets))) != (ss.active_preset or next(iter(cfg.presets))):
    sb.caption("⚠ **Not applied yet**: press Apply Preset. The pose and load below still belong to "
               + (f"'{pretty(ss.active_preset)}'" if ss.active_preset else "the default / your manual inputs") + ".")

# Gait presets: pick a keyframe with buttons; angles, ground force and centre of pressure then come from that keyframe.
active = cfg.presets.get(ss.active_preset) if ss.active_preset else None
motion_cfg = active.motion if active else None
G = cfg.gaits[motion_cfg["gait"]] if (motion_cfg and motion_cfg["type"] == "gait") else None
gait_on = False
f_g, cop_g, key_idx = None, 0.0, 0
if G is not None:
    sb.checkbox("Follow Walking Cycle", key="gait_follow",
                help="Drive the joint angles, ground force and center of pressure from the gait keyframes. "
                     "Untick to edit them by hand.")
    gait_on = bool(ss.gait_follow)
    if gait_on:
        _sel = sb.pills("Stance Keyframe", list(range(len(G.keys))), format_func=lambda i: _key_label(G, i),
                        selection_mode="single", key="gait_key",
                        help="Each button is one keyframe of the stance phase (heel strike -> toe-off).")
        key_idx = int(_sel) if _sel is not None else int(ss.get("gait_key_last", 0))   # a deselected button keeps the last one
        ss.gait_key_last = key_idx
        q_g, f_g, cop_g = gait_mod.sample(G, G.keys[key_idx].phase)
        for _i, _qd in enumerate(np.degrees(q_g)):
            ss[f"qn{_i}"] = float(_qd)
            _number_changed(_i)
        sb.caption(f"Stance phase **{G.keys[key_idx].phase * 100:.0f}%** of {G.stance_time:g} s")

q_deg = np.zeros(N)
for i, j in enumerate(cfg.joints):
    lo, hi = np.degrees(j.limits)
    a, b = sb.columns([3, 2])
    a.slider(pretty(j.name), float(lo), float(hi), key=f"qs{i}", step=0.5, on_change=_slider_changed, args=(i,),
             disabled=gait_on)
    b.number_input("deg", -360.0, 360.0, key=f"qn{i}", step=1.0, on_change=_number_changed, args=(i,),
                   label_visibility="visible", disabled=gait_on)
    q_deg[i] = ss[f"qn{i}"]

# ---- pelvis + payload -------------------------------------------------------------------------------------------
sb.subheader("Pelvis")
c1, c2 = sb.columns(2)
pelvis_pitch = c1.number_input("Pitch [deg] (+ = Forward Lean)", -45.0, 45.0, 0.0, 1.0)
pelvis_roll = c2.number_input("Roll [deg]", -45.0, 45.0, 0.0, 1.0)
payload = sb.number_input("External Payload at Pelvis [kg]", 0.0, 500.0, 0.0, 0.5,
                          help="Adds to the weight used by the 'fraction x weight' presets. It acts on the pelvis, "
                               "so it reaches the leg only through the foot load.")

# ---- foot contact load ------------------------------------------------------------------------------------------
sb.header("Foot Contact Load")
sb.radio("Support", ["double", "single"], key="support", horizontal=True, on_change=_support_changed,
         format_func=pretty,
         help="Only changes the default load fraction (config: double_support_fraction / single_support_fraction).")
_opts = FRACTIONS + ["custom", "manual"] + (["motion"] if motion_cfg and gait_on else [])
if gait_on:
    ss.frac_choice = "motion"
if ss.frac_choice not in _opts:
    ss.frac_choice = "0.5"
sb.radio("Vertical Load", _opts, key="frac_choice", horizontal=True, format_func=_lbl, disabled=gait_on,
         help="fraction x robot weight; 'manual' lets you type Fz; 'Preset Motion' comes from the walking keyframe.")
if ss.frac_choice == "custom":
    sb.number_input("Fraction × Robot Weight", 0.0, 10.0, key="frac_custom", step=0.1)
weight = (cfg.robot_total_mass + payload) * cfg.gravity          # one body weight [N], payload included

if ss.frac_choice == "manual":
    fz = sb.number_input("Fz [N]", -5000.0, 20000.0, key="fz", step=10.0)
    frac_eff = fz / weight
elif ss.frac_choice == "motion" and motion_cfg:                    # gait keyframe: ground force from the keyframe
    frac_eff = float(f_g[2])
    sb.caption(f"Vertical {f_g[2]:.2f} BW · fore-aft {f_g[0]:+.2f} BW · lateral {f_g[1]:+.2f} BW · "
               f"CoP {cop_g * 1000:+.0f} mm from the ankle")
    fz = frac_eff * weight
    sb.caption(f"Fz = {frac_eff:.2f} × {weight:.0f} N = **{fz:.0f} N**")
else:
    frac_eff = ss.frac_custom if ss.frac_choice == "custom" else float(ss.frac_choice)
    fz = frac_eff * weight
    sb.caption(f"Fz = {frac_eff:g} × {weight:.0f} N = **{fz:.0f} N**")
if gait_on:                                                        # horizontal force and CoP also come from the keyframe
    ss.fx, ss.fy = float(f_g[0] * weight), float(f_g[1] * weight)
    ss.cop_x_mm, ss.cop_y_mm = float(cop_g * 1000.0), 0.0
c1, c2 = sb.columns(2)
fx = c1.number_input("Fx [N]", -5000.0, 5000.0, key="fx", step=10.0, disabled=gait_on)
fy = c2.number_input("Fy [N]", -5000.0, 5000.0, key="fy", step=10.0, disabled=gait_on)
with sb.expander("Foot Moment / Center of Pressure"):
    m1, m2, m3 = st.columns(3)
    Mx, My_, Mz_ = (m1.number_input("Mx [N·m]", key="mx"), m2.number_input("My [N·m]", key="my"),
                    m3.number_input("Mz [N·m]", key="mz"))
    k1, k2 = st.columns(2)
    cop_x = k1.number_input("CoP X [mm]", -200.0, 200.0, key="cop_x_mm", step=1.0, disabled=gait_on,
                            help="Shift of the contact point in the foot frame (x forward)")
    cop_y = k2.number_input("CoP Y [mm]", -200.0, 200.0, key="cop_y_mm", step=1.0, disabled=gait_on)

_extra = [f"Fx {fx:+.0f} N" if abs(fx) > 1e-9 else "", f"Fy {fy:+.0f} N" if abs(fy) > 1e-9 else "",
          f"CoP ({cop_x:+.0f}, {cop_y:+.0f}) mm" if (abs(cop_x) > 1e-9 or abs(cop_y) > 1e-9) else "",
          "foot moment" if any(abs(m) > 1e-9 for m in (Mx, My_, Mz_)) else ""]
_extra = [e for e in _extra if e]
sb.caption("Load in use: **vertical " + f"{fz:.0f} N**" + (" + " + ", ".join(_extra) if _extra else
           " only, at the point under the ankle (no horizontal force, CoP offset or moment)"))

# ---- save the current pose as a preset --------------------------------------------------------------------------
sb.header("Save Pose")
if not LOCAL:
    sb.caption("Saving presets is turned off in the shared web version (it would change the app for every visitor). "
               "Run the app on your own computer to save presets to leg_config.yaml.")
pname = sb.text_input("Preset Name", "my_pose", disabled=not LOCAL)
pdesc = sb.text_area("Description (Optional)", key="save_desc", height=90, disabled=not LOCAL,
                     placeholder="What is this pose / load case? Shown under the preset selector.")
if sb.button("Save Current Pose as Preset", disabled=not LOCAL):
    try:
        f_now = frac_eff
        mo = None
        if ss.frac_choice == "motion" and motion_cfg:          # keep the gait keyframe with the saved preset
            mo = {"type": "gait", "gait": G.name, "phase": round(G.keys[key_idx].phase, 3)}
        cfgmod.save_preset(CONFIG_PATH, pname, [round(float(x), 1) for x in q_deg], round(f_now, 3), ss.support,
                           pdesc, mo)
        sb.success(f"Saved '{pname}' to {CONFIG_PATH.name}")
        st.rerun()
    except ValueError as e:
        sb.error(str(e))

# =========================================================================== 4. analysis
inp = Inputs(q=np.radians(q_deg), pelvis_pitch=np.radians(pelvis_pitch), pelvis_roll=np.radians(pelvis_roll),
             load=Load(np.array([fx, fy, fz]), np.array([Mx, My_, Mz_])), cop_offset=(cop_x / 1000, cop_y / 1000),
             payload_mass=payload)
t0 = time.perf_counter()
res = analyze(cfg, inp)
t_calc = (time.perf_counter() - t0) * 1000

# =========================================================================== 5. main area
st.title("Humanoid Leg Load Calculator")
st.caption("First-pass quasi-static sizing tool (not a simulator).  Torque sign: torque the actuator applies to the "
           "distal link about the joint axis to hold the pose.")
if cfg.placeholder_lines:
    with st.container(border=True):
        st.error(f"**{len(cfg.placeholder_lines)} PLACEHOLDER values remain in leg_config.yaml** — results are "
                 "illustrative only until geometry, masses, sections and actuator limits are replaced.")
        with st.expander("Show Placeholder Lines"):
            st.code("\n".join(f"{n:4d}: {t}" for n, t in cfg.placeholder_lines), language="yaml")
for w in res.warnings:
    st.warning(w)
for e in res.errors:
    st.error(e)

# ---- per-joint summary table ------------------------------------------------------------------------------------
act_by_joint = {}                                    # joint index -> names of the actuators driving it
for t in cfg.transmissions:
    for jj in t.joints:
        act_by_joint[jj] = [cfg.actuators[a].name for a in t.actuators]


def joint_table() -> pd.DataFrame:
    """One row per joint: angle, torque split, the motors driving it (with this joint's own share of them) and the
    distal link's peak load / stress."""
    rows = []
    for i, j in enumerate(cfg.joints):
        lr, fl = res.links[i], res.flanges[i]
        names = act_by_joint[i]
        sh = res.shares[i] if res.shares else None
        shared = len(names) > 1
        rows.append({
            "Joint": pretty(j.name), "Angle [°]": q_deg[i], "Torque [N·m]": res.static.tau[i],
            "From GRF": res.static.tau_grf[i], "From Gravity": res.static.tau_gravity[i],
            "Motors": ", ".join(pretty(n) for n in names) + (" (shared)" if shared else ""),
            "Own Share [% Cont]": None if sh is None else sh.util_cont * 100,
            "Own Share [% Peak]": None if sh is None else sh.util_peak * 100,
            "Link": pretty(lr.name), "Peak Bending [N·m]": lr.peak_bending.M_bend,
            "Peak Stress [MPa]": None if lr.peak_stress is None else lr.peak_stress.sigma / 1e6,
            "Safety Factor": lr.min_sf, "Bearing F [N]": float(np.linalg.norm(fl.force_local)),
            "_status": sh.status if sh else "ok", "_cont": bool(sh and sh.exceeds_continuous)})
    return pd.DataFrame(rows)


def style_rows(df: pd.DataFrame):
    """Colour only the cell that has the problem (same colours as the diagram labels): the motor-share cells by the
    joint's motor status, the safety-factor cells by the link's safety factor; hide the helper columns."""
    def fill(kind: str) -> str:
        return f"background-color: {viz.LIGHT[kind]}; color: {viz.DARK[kind]}"

    def cell_styles(d: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame("", index=d.index, columns=d.columns)
        for i, r in d.iterrows():
            motor = fill("over_peak") if r["_status"] == "over_peak" else fill("warn") if r["_cont"] else ""
            out.loc[i, ["Own Share [% Cont]", "Own Share [% Peak]"]] = motor
            sf = r["Safety Factor"]
            if sf is not None and pd.notna(sf):
                out.loc[i, ["Safety Factor", "Peak Stress [MPa]"]] = (
                    fill("over_peak") if sf < 1.0 else fill("warn") if sf < 2.0 else "")
        return out
    return (df.style.apply(cell_styles, axis=None).hide(axis="columns", subset=["_status", "_cont"])
            .format({"Angle [°]": "{:.1f}", "Torque [N·m]": "{:.2f}", "From GRF": "{:.2f}", "From Gravity": "{:.2f}",
                     "Own Share [% Cont]": "{:.0f}", "Own Share [% Peak]": "{:.0f}", "Peak Bending [N·m]": "{:.2f}",
                     "Peak Stress [MPa]": "{:.1f}", "Safety Factor": "{:.1f}", "Bearing F [N]": "{:.0f}"}, na_rep="–"))


df = joint_table()
styled = style_rows(df)

# ---- tabs -------------------------------------------------------------------------------------------------------
_tabs = st.tabs(["Pose", "3D View", "Actuators", "Link Loads", "Export"] + (["Gait Cycle"] if cfg.gaits else []))
tab_pose, tab_3d, tab_act, tab_links, tab_export = _tabs[:5]
tab_gait = _tabs[5] if cfg.gaits else None

with tab_pose:
    left, right = st.columns([3, 2], gap="medium")
    with left:
        st.plotly_chart(viz.stick_figures(cfg, res), width="stretch", theme=None, key="stick")
    with right:
        st.markdown("**Per-Joint Summary**  \nmotor cells: amber over continuous, red over peak · safety-factor cells: amber < 2, red < 1")
        st.dataframe(styled, width="stretch", hide_index=True, height=300)
        s = res.singularity
        st.caption(f"Leg reach = {s['reach_ratio'] * 100:.1f}% of full extension (100% = straight leg) · computed in {t_calc:.1f} ms")
        st.caption("*Own Share* is the load on the joint's motors from that joint's torque alone. The two ankle joints share the two ankle motors, so each ankle row shows only its own demand (a joint with zero torque shows 0 %); each motor's true combined load is on the Actuators tab.")
        st.caption("Link labels: peak bending moment [N·m], coloured by safety factor "
                   "(green ≥ 2, amber 1–2, red < 1). Zero-length links are not drawn.")

with tab_3d:
    st.plotly_chart(viz.fig_3d(cfg, res), width="stretch", theme=None, key="fig3d")
    st.caption("Frame triads (x red, y green, z blue): pelvis and each joint frame after its rotation. "
               "World: x forward, y left, z up.")

with tab_act:
    if res.actuators:
        st.plotly_chart(viz.util_bars(cfg, res), width="stretch", theme=None, key="util")
        st.dataframe(pd.DataFrame([{
            "Actuator": pretty(a.name), "Load": f"{a.value:+.2f} {a.unit}", "Continuous": a.continuous, "Peak": a.peak,
            "% Cont": round(a.util_cont * 100, 1), "% Peak": round(a.util_peak * 100, 1), "Status": pretty(a.status)}
            for a in res.actuators]),
            hide_index=True, width="stretch")
        st.caption("green < 60 % continuous · amber up to peak · red over peak. "
                   "Any actuator above 100 % continuous will heat up if the pose is held.")
    if res.rods:
        st.subheader("Ankle Rods")
        st.dataframe(pd.DataFrame([{
            "Rod": pretty(rd.name), "Force [N]": rd.force, "State": "tension" if rd.force > 0 else "compression",
            "Lever Angle [°]": float(np.degrees(rd.lever_angle)), "Effective Lever Arm [mm]": abs(rd.arm) * 1000,
            "Rod Length [mm]": rd.length * 1000} for rd in res.rods]).style.format(
            {"Force [N]": "{:+.0f}", "Lever Angle [°]": "{:+.1f}", "Effective Lever Arm [mm]": "{:.1f}",
             "Rod Length [mm]": "{:.1f}"}), hide_index=True, width="stretch")
        st.caption("Rod force = actuator torque / effective lever arm, with the arm the perpendicular distance from the lever "
                   "shaft to the rod line. Rods are ideal two-force members (no end moments). Check the rod end bearings and "
                   "buckling (compression) against these forces.")
    with st.expander("Hand check: where do these numbers come from?"):
        _fk = res.fk
        _F = inp.load.force
        _r = _fk.p_contact - _fk.p                          # joint -> contact point, world frame
        _rows = []
        for _i, _n in ((0, "hip_flexion"), (3, "knee"), (4, "ankle_flexion")):
            _lever = float(_r[_i][0])                       # forward distance from the joint to the contact point
            _est = abs(_r[_i][2] * _F[0] - _lever * _F[2])
            _rows.append({"Joint": pretty(_n), "Forward Lever [mm]": _lever * 1000, "Vertical Force [N]": float(_F[2]),
                          "|Fz×lever − Fx×height| [N·m]": _est, "Tool: Ground-Reaction Part [N·m]": abs(res.static.tau_grf[_i])})
        st.dataframe(pd.DataFrame(_rows).style.format(
            {"Forward Lever [mm]": "{:+.0f}", "Vertical Force [N]": "{:.0f}", "|Fz×lever − Fx×height| [N·m]": "{:.1f}",
             "Tool: Ground-Reaction Part [N·m]": "{:.1f}"}), hide_index=True, width="stretch")
        st.caption("Sagittal-plane check: a joint torque from the ground force is about the vertical force times the forward "
                   "distance from the joint to the point where the force acts (plus the fore-aft force times the height). "
                   "The two right-hand columns should agree; any difference is the pelvis tilt or hip abduction / rotation. "
                   "The tool then adds the leg's own weight.")
        if res.T is not None and res.T.shape == (6, 6):
            _T = res.T[4:, 4:]
            st.caption(f"Ankle: each motor's torque is set by the lever ratio T (motor angle per ankle angle). Here pitch "
                       f"{_T[0, 0]:+.2f} / {_T[1, 0]:+.2f} and roll {_T[0, 1]:+.2f} / {_T[1, 1]:+.2f} for the upper / lower "
                       f"motor. A ratio below 1 means the motors must supply MORE torque than the ankle sees; the two "
                       f"motors share the pitch torque.")
    st.subheader("Joint-Torque Split")
    st.dataframe(pd.DataFrame({"Joint": [pretty(n) for n in cfg.joint_names], "Total [N·m]": res.static.tau,
                               "Ground Reaction [N·m]": res.static.tau_grf, "Gravity of Links [N·m]": res.static.tau_gravity}),
                 hide_index=True, width="stretch")

with tab_links:
    names = [lr.name for lr in res.links]
    if res.rods is None:
        st.warning("**Ankle rods not included.** The rod forces could not be computed at this pose, so the lower shank, "
                   "ankle cross, foot and J5 / J6 bearing loads below use the plain cut method and are not valid.")
    else:
        st.info("The two ankle rods are in the load path: each is a two-force member whose force is added to every cut it "
                "crosses (lower shank below a lever pivot, ankle cross, foot root, J5 / J6 bearings). Not modelled: "
                "rod-end moments, the motor mounts (the rod force enters the shank at the lever pivot) and stress in the "
                "levers, rods and pickups. Loads inside the motor-braced shank zone are lumped at the pivots.")
    pick = st.selectbox("Link", names, index=min(2, len(names) - 1), format_func=pretty)
    lr = res.links[names.index(pick)]
    if lr.two_force:
        st.info("Two-force member: only tension/compression is meaningful. "
                + (lr.two_force_violation or "Transverse loads are negligible."))
    st.plotly_chart(viz.link_diagram(lr), width="stretch", theme=None, key="linkdiag")
    st.dataframe(pd.DataFrame([{
        "Distance s [mm]": s_.s * 1000, "N [N]": s_.N, "Vy [N]": s_.Vy, "Vz [N]": s_.Vz, "T [N·m]": s_.T, "My [N·m]": s_.My,
        "Mz [N·m]": s_.Mz, "σ [MPa]": None if s_.sigma is None else s_.sigma / 1e6, "SF": s_.sf, "Even Station": s_.even}
        for s_ in lr.stations]), hide_index=True, width="stretch")
    st.caption("Beam frame: x along the link, y from the section's y_ref, z = x × y. N>0 is tension. Stress = "
               "Kt(|N|/A + |My|c_z/I_y + |Mz|c_y/I_z); shear and torsion are reported but not in the stress. "
               "'Even Station' = one of the evenly spaced stations; unticked rows are extra stations added at lumped "
               "masses and section changes so the true peak is not missed.")
    st.subheader("Joint Bearing / Output-Flange Reactions (Joint Frame)")
    st.dataframe(pd.DataFrame([{
        "Joint": pretty(f.joint), "Fx": f.force_local[0], "Fy": f.force_local[1], "Fz": f.force_local[2],
        "Axial Thrust [N]": f.axial_force, "Radial [N]": f.radial_force, "Tilt Moment [N·m]": f.tilt_moment,
        "Axis Moment [N·m]": f.axis_moment} for f in res.flanges]).round(2), hide_index=True, width="stretch")

with tab_export:
    st.write("Packages the current pose, inputs and all results for download"
             + (" and also writes them to `exports/`." if LOCAL else "."))
    if st.button("Export Report (JSON + CSV" + (" + PNG)" if LOCAL else ")")):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        js, cs = report.to_json(cfg, res), report.to_csv(cfg, res)
        ss.export = {"stamp": stamp, "json": js, "csv": cs, "png": None, "png_err": None}
        if LOCAL:
            EXPORT_DIR.mkdir(exist_ok=True)
            (EXPORT_DIR / f"leg_report_{stamp}.json").write_text(js, encoding="utf-8")
            (EXPORT_DIR / f"leg_report_{stamp}.csv").write_text(cs, encoding="utf-8")
            try:
                png = viz.stick_figures(cfg, res).to_image(format="png", width=1500, height=700, scale=2)
                (EXPORT_DIR / f"leg_figure_{stamp}.png").write_bytes(png)
                ss.export["png"] = png
            except Exception as e:  # kaleido needs a Chrome install; degrade gracefully
                ss.export["png_err"] = str(e)
    ex = ss.get("export")
    if ex:
        if LOCAL:
            st.success(f"Saved to {EXPORT_DIR} (stamp {ex['stamp']}).")
        c1, c2, c3 = st.columns(3)
        c1.download_button("Download JSON", ex["json"], f"leg_report_{ex['stamp']}.json", "application/json")
        c2.download_button("Download CSV", ex["csv"], f"leg_report_{ex['stamp']}.csv", "text/csv")
        if ex["png"]:
            c3.download_button("Download PNG", ex["png"], f"leg_figure_{ex['stamp']}.png", "image/png")
        elif LOCAL:
            st.warning("PNG export failed (Plotly's image exporter, kaleido, needs Chrome — run "
                       "`plotly_get_chrome` once). Details: " + str(ex["png_err"])[:300])
    st.caption("To save a picture of any figure, use the camera icon in the figure's top-right toolbar.")


# ---- gait cycle tab ---------------------------------------------------------------------------------------------
@st.cache_data(show_spinner="Sweeping the stance phase...")
def _sweep(_cfg, gait_name: str, hardware: tuple, mtime: float, payload_kg: float):
    """Cached stance sweep.  ``_cfg`` is not hashed, so everything that changes the result is an explicit argument:
    the gait, the selected hardware options, the config file's mtime and the payload."""
    return gait_mod.stance_sweep(_cfg, _cfg.gaits[gait_name], 41, payload_kg)


if tab_gait is not None:
    with tab_gait:
        _gnames = list(cfg.gaits)
        _gsel = st.selectbox("Gait", _gnames, format_func=pretty, key="gait_pick")
        GS = cfg.gaits[_gsel]
        st.caption(GS.description)
        sw = _sweep(cfg, _gsel, tuple(sorted(cfg.active_choices.items())), CONFIG_PATH.stat().st_mtime, payload)
        rows = gait_mod.summarize(cfg, GS, sw)
        st.subheader("Actuator Load Over the Stance Phase")
        st.plotly_chart(viz.walk_load_plot(cfg, GS, sw), width="stretch", theme=None, key="walk_load")
        tbl = pd.DataFrame([{
            "Actuator": pretty(r_["name"]), "Model": r_["model"], "Peak Load": r_["peak"], "% of Peak Rating": r_["peak_pct"],
            "RMS (Stance)": r_["rms_stance"], "RMS % of Continuous (Stance)": r_["rms_stance_pct"],
            "_bad": r_["peak_pct"] > 100, "_warn": r_["rms_stance_pct"] > 100} for r_ in rows])

        def _gait_style(r_):
            col = (f"background-color: {viz.LIGHT['over_peak']}; color: {viz.DARK['over_peak']}" if r_["_bad"]
                   else f"background-color: {viz.LIGHT['warn']}; color: {viz.DARK['warn']}" if r_["_warn"] else "")
            return [col] * len(r_)

        st.dataframe(tbl.style.apply(_gait_style, axis=1).hide(axis="columns", subset=["_bad", "_warn"]).format(
            {"Peak Load": "{:.1f}", "% of Peak Rating": "{:.0f}", "RMS (Stance)": "{:.1f}",
             "RMS % of Continuous (Stance)": "{:.0f}"}),
            hide_index=True, width="stretch")
        st.caption("Red: peak above the actuator's peak rating. Amber: RMS over the stance above the continuous rating. "
                   "Quasi-static: ground force and the leg's own weight only, no leg inertia and no swing phase, so the "
                   "hip and knee are under-stated. The joint angles, ground-force shares and centre-of-pressure path are "
                   "typical human values scaled to this robot, not robot data; the peak torque is mostly "
                   "(vertical force) x (centre-of-pressure distance from the joint).")
