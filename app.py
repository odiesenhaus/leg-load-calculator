"""Streamlit UI for the humanoid leg load calculator.   Run:  streamlit run app.py

All physics lives in the ``leg/`` package (no UI imports there); this file only wires widgets, tables and figures.
Streamlit was chosen over Dash because every input change simply re-runs one pure function (``leg.analyze`` takes
about 10-30 ms), so there is no callback graph to maintain.

How the script is organised (Streamlit re-runs it top to bottom on every interaction):

  1. Config            load + validate ``leg_config.yaml`` (cached), apply the selected hardware options
  2. Session state     defaults and callbacks.  Widgets that must be driven from code (joint angles, preset
                       motions, the walking keyframe) keep their value in ``st.session_state`` under a fixed key
  3. Sidebar (rail)    inputs: hardware options, load case + pose, pelvis, foot load, save-preset
  4. Analysis          one ``leg.analyze`` call on the inputs
  5. Main area         header + status pill, the stage (leg drawing, motor readouts) beside the spec sheet, then the
                       analysis cards (actuators, per-joint summary, link loads, gait cycle, hand check, export,
                       conventions).  Look and feel: .streamlit/config.toml + the small CSS block in section 5.

The walking preset overrides the manual inputs ("Preset Motion" on the vertical load; "Follow Walking Cycle"): it writes
the driven values into session state *before* the widgets are created and disables those widgets, so the displayed
numbers are always the ones used.
"""
from __future__ import annotations

import html
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
LABELS = {"custom": "Custom", "manual": "Manual", "motion": "Preset motion"}


def _lbl(x: str) -> str:
    """Display label for the vertical-load radio options."""
    return LABELS.get(x, pretty(x))


st.set_page_config(page_title="Leg load calculator", page_icon=":material/precision_manufacturing:", layout="wide")
ss = st.session_state
viz.use_theme("auto")                  # figures: transparent, text / grid colours from Streamlit's light or dark theme


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
    sb.html('<p class="eyebrow">Hardware</p>')
    for _key, _opt in base_cfg.actuator_options.items():
        _names = list(_opt.choices)
        sb.selectbox(_opt.label, _names, index=_names.index(_opt.default), key=f"opt_{_key}",
                     help="Replaces the ratings, rotor inertia and weight of these actuators; geometry is not changed.")
        _note = _opt.choices[cfg.active_choices[_key]]["note"]
        if _note:
            sb.caption(_note)

# ---- preset + pose ----------------------------------------------------------------------------------------------
sb.html('<p class="eyebrow">Load case</p>')
sb.selectbox("Preset", list(cfg.presets), key="preset_pick", format_func=pretty)
_picked = cfg.presets.get(ss.get("preset_pick", next(iter(cfg.presets))))
if _picked is not None and _picked.description:
    sb.caption(_picked.description)
sb.button("Apply preset", on_click=_apply_preset, type="primary", icon=":material/check:")
if ss.get("preset_pick", next(iter(cfg.presets))) != (ss.active_preset or next(iter(cfg.presets))):
    sb.caption("⚠ **Not applied yet**: press Apply preset. The pose and load below still belong to "
               + (f"'{pretty(ss.active_preset)}'" if ss.active_preset else "the default / your manual inputs") + ".")

# Gait presets: pick a keyframe with buttons; angles, ground force and centre of pressure then come from that keyframe.
active = cfg.presets.get(ss.active_preset) if ss.active_preset else None
motion_cfg = active.motion if active else None
G = cfg.gaits[motion_cfg["gait"]] if (motion_cfg and motion_cfg["type"] == "gait") else None
gait_on = False
f_g, cop_g, key_idx = None, 0.0, 0
if G is not None:
    sb.checkbox("Follow walking cycle", key="gait_follow",
                help="Drive the joint angles, ground force and center of pressure from the gait keyframes. "
                     "Untick to edit them by hand.")
    gait_on = bool(ss.gait_follow)
    if gait_on:
        _sel = sb.pills("Stance keyframe", list(range(len(G.keys))), format_func=lambda i: _key_label(G, i),
                        selection_mode="single", key="gait_key",
                        help="Each button is one keyframe of the stance phase (heel strike -> toe-off).")
        key_idx = int(_sel) if _sel is not None else int(ss.get("gait_key_last", 0))   # a deselected button keeps the last one
        ss.gait_key_last = key_idx
        q_g, f_g, cop_g = gait_mod.sample(G, G.keys[key_idx].phase)
        for _i, _qd in enumerate(np.degrees(q_g)):
            ss[f"qn{_i}"] = float(_qd)
            _number_changed(_i)
        sb.caption(f"Stance phase **{G.keys[key_idx].phase * 100:.0f}%** of {G.stance_time:g} s")

sb.html('<p class="eyebrow">Joint angles</p>')
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
sb.html('<p class="eyebrow">Pelvis</p>')
c1, c2 = sb.columns(2)
pelvis_pitch = c1.number_input("Pitch [deg] (+ = forward lean)", -45.0, 45.0, 0.0, 1.0)
pelvis_roll = c2.number_input("Roll [deg]", -45.0, 45.0, 0.0, 1.0)
payload = sb.number_input("External payload at pelvis [kg]", 0.0, 500.0, 0.0, 0.5,
                          help="Adds to the weight used by the 'fraction x weight' presets. It acts on the pelvis, "
                               "so it reaches the leg only through the foot load.")

# ---- foot contact load ------------------------------------------------------------------------------------------
sb.html('<p class="eyebrow">Foot contact load</p>')
sb.radio("Support", ["double", "single"], key="support", horizontal=True, on_change=_support_changed,
         format_func=pretty,
         help="Only changes the default load fraction (config: double_support_fraction / single_support_fraction).")
_opts = FRACTIONS + ["custom", "manual"] + (["motion"] if motion_cfg and gait_on else [])
if gait_on:
    ss.frac_choice = "motion"
if ss.frac_choice not in _opts:
    ss.frac_choice = "0.5"
sb.radio("Vertical load", _opts, key="frac_choice", horizontal=True, format_func=_lbl, disabled=gait_on,
         help="fraction x robot weight; 'manual' lets you type Fz; 'Preset Motion' comes from the walking keyframe.")
if ss.frac_choice == "custom":
    sb.number_input("Fraction × robot weight", 0.0, 10.0, key="frac_custom", step=0.1)
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
with sb.expander("Foot moment / centre of pressure"):
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
sb.html('<p class="eyebrow">Save pose</p>')
if not LOCAL:
    sb.caption("Saving presets is turned off in the shared web version (it would change the app for every visitor). "
               "Run the app on your own computer to save presets to leg_config.yaml.")
pname = sb.text_input("Preset name", "my_pose", disabled=not LOCAL)
pdesc = sb.text_area("Description (optional)", key="save_desc", height=90, disabled=not LOCAL,
                     placeholder="What is this pose / load case? Shown under the preset selector.")
if sb.button("Save current pose as preset", disabled=not LOCAL, icon=":material/save:"):
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
# Layout ("drafting desk"): header with a status pill; a stage card (the leg drawing, Side / Front / 3D, plus one
# readout per motor) beside a spec-sheet card of the headline numbers; then a band of analysis cards.  Colours come
# from .streamlit/config.toml; the few custom pieces (eyebrow labels, pill, readouts, spec sheet, card fill) are
# styled by the CSS below.  Its colours use CSS light-dark(), so they follow the viewer's theme in the browser
# (Streamlit sets color-scheme on the app container); the figures use viz's "auto" palette for the same reason.
st.html("""<style>
[data-testid="stApp"]{
  --panel:light-dark(#ffffff,#202326); --sunken:light-dark(#f4f6f7,#1b1d20);
  --ink:light-dark(#13171b,#eceef0); --ink-2:light-dark(#444d56,#b2b8be); --ink-3:light-dark(#717a84,#868d95);
  --rule:light-dark(#d2d8dd,#373b40); --rule-2:light-dark(#e4e8eb,#2c3034);
  --rod1:light-dark(#2a78d6,#3987e5); --rod2:light-dark(#eb6834,#d95926);
  --good:#0ca30c; --warn:#fab219; --crit:#d03b3b;
  --good-ink:light-dark(#0a6c0a,#62d062); --warn-ink:light-dark(#7a5200,#f7c04f); --crit-ink:light-dark(#a92626,#ff8d8d);
}
[class*="st-key-card_"]{background:var(--panel);border-color:var(--rule) !important}
.eyebrow{font-family:'Barlow Condensed',sans-serif;font-weight:600;font-size:13px;letter-spacing:.09em;
  text-transform:uppercase;color:var(--ink-3);margin:10px 0 -4px}
.sub{margin:2px 0 0;color:var(--ink-2);max-width:70ch;font-size:15px}
.mono{font-family:'JetBrains Mono',ui-monospace,monospace;font-variant-numeric:tabular-nums}
.pills{display:flex;flex-wrap:wrap;gap:8px;justify-content:flex-end}
.pill{display:inline-flex;align-items:center;gap:8px;padding:6px 12px 6px 10px;border-radius:999px;
  border:1px solid var(--rule);background:var(--panel);font-weight:500;font-size:14px;color:var(--ink)}
.pill svg{width:18px;height:18px;flex:none}
.pill.good svg{color:var(--good)} .pill.good .t{color:var(--good-ink)}
.pill.warn svg{color:var(--warn)} .pill.warn .t{color:var(--warn-ink)}
.pill.crit svg{color:var(--crit)} .pill.crit .t{color:var(--crit-ink)}
.readout{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:8px}
.ro{display:flex;align-items:baseline;gap:10px;padding:8px 10px;border-radius:6px;background:var(--sunken);font-size:13.5px}
.ro .k{display:flex;align-items:center;gap:6px;color:var(--ink-2);min-width:92px}
.ro .v{font-family:'JetBrains Mono',monospace;font-size:17px;font-weight:500;color:var(--ink)}
.ro .x{font-family:'JetBrains Mono',monospace;color:var(--ink-3);font-size:12px;margin-left:auto;white-space:nowrap}
.dot{width:9px;height:9px;border-radius:50%;flex:none;display:inline-block}
.dot.good{background:var(--good)} .dot.warn{background:var(--warn)} .dot.crit{background:var(--crit)}
.legend{display:flex;gap:14px;font-size:13px;color:var(--ink-2);flex-wrap:wrap}
.sw{display:inline-block;width:14px;height:4px;border-radius:2px;vertical-align:middle;margin-right:6px}
.sw.rod1{background:var(--rod1)} .sw.rod2{background:var(--rod2)}
.sw.good{background:var(--good)} .sw.warn{background:var(--warn)} .sw.crit{background:var(--crit)}
.boxes{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:2px 0 14px}
.box{padding:10px 12px;border-radius:6px;background:var(--sunken)}
.box .k{display:flex;align-items:center;gap:6px;font-size:13px;color:var(--ink-2)}
.box .v{font-family:'JetBrains Mono',monospace;font-size:26px;font-weight:500;line-height:1.15;margin-top:2px}
.box .v.good{color:var(--good-ink)} .box .v.warn{color:var(--warn-ink)} .box .v.crit{color:var(--crit-ink)}
.box .v small{font-size:13px;color:var(--ink-3);font-weight:400;margin-left:3px}
.box .n{font-size:12.5px;color:var(--ink-3);margin-top:2px}
.spech{font-family:'Barlow Condensed',sans-serif;font-size:14px;letter-spacing:.08em;text-transform:uppercase;
  color:var(--ink-3);margin:4px 0 8px;font-weight:600}
.spec{margin:0 0 14px;display:grid;grid-template-columns:minmax(0,1fr) auto;gap:5px 12px;font-size:14px}
.spec dt{color:var(--ink-2);margin:0;padding:0}
.spec,.conv dl,.flags,.spec dt,.spec dd,.conv dt,.conv dd{margin-left:0 !important;padding-left:0 !important}
.spec dd{margin:0;font-family:'JetBrains Mono',monospace;font-size:13px;text-align:right;white-space:nowrap;color:var(--ink)}
.spec dd.warn{color:var(--warn-ink)} .spec dd.crit{color:var(--crit-ink)} .spec dd.good{color:var(--good-ink)}
.flags{list-style:none;margin:0 0 12px;padding:0;font-size:13.5px}
.flags li{display:flex;gap:8px;align-items:baseline;padding:3px 0;color:var(--ink-2)}
.flags .dot{transform:translateY(1px)}
.fine{font-size:12.5px;color:var(--ink-3);margin:0}
.conv dl{display:grid;grid-template-columns:auto minmax(0,1fr);gap:6px 14px;margin:0;font-size:14px}
.conv dt{font-weight:600;color:var(--ink)} .conv dd{margin:0;color:var(--ink-2)}
</style>""")

_ICON = {"good": '<path fill="currentColor" d="M10 1.5a8.5 8.5 0 1 0 0 17 8.5 8.5 0 0 0 0-17Zm-1.2 12.3L4.9 9.9l1.4-1.4 2.5 2.5 4.9-4.9 1.4 1.4-6.3 6.3Z"/>',
         "warn": '<path fill="currentColor" d="M10 1.8 1.2 17.5h17.6L10 1.8Zm-.9 5.7h1.8v5h-1.8v-5Zm0 6.4h1.8v1.8h-1.8v-1.8Z"/>',
         "crit": '<path fill="currentColor" d="M10 1.5a8.5 8.5 0 1 0 0 17 8.5 8.5 0 0 0 0-17Zm3.6 10.7-1.4 1.4L10 11.4l-2.2 2.2-1.4-1.4L8.6 10 6.4 7.8l1.4-1.4L10 8.6l2.2-2.2 1.4 1.4L11.4 10l2.2 2.2Z"/>'}


def pill(kind: str, text: str) -> str:
    return (f'<span class="pill {kind}"><svg viewBox="0 0 20 20" aria-hidden="true">{_ICON[kind]}</svg>'
            f'<span class="t">{html.escape(text)}</span></span>')


CLS = {"over_peak": "crit", "warn": "warn", "ok": "good"}        # actuator status -> CSS class


def short(name: str) -> str:
    """'ankle_upper_act' -> 'Ankle Upper' (readout labels)."""
    return pretty(name).replace(" Actuator", "")


# ---- headline status -------------------------------------------------------------------------------------------
acts = res.actuators or []
worst = max(acts, key=lambda a: a.util_peak) if acts else None
_sfs = [(lr.min_sf, lr.name) for lr in res.links if lr.min_sf is not None]
min_sf, min_sf_link = min(_sfs) if _sfs else (None, None)
over_peak = [a for a in acts if a.status == "over_peak"]
over_cont = [a for a in acts if a.exceeds_continuous]
if res.errors:
    status = ("crit", "Analysis incomplete: see the error above")
elif over_peak:
    a_ = max(over_peak, key=lambda a: a.util_peak)
    status = ("crit", f"{short(a_.name)} motor at {a_.util_peak * 100:.0f} % of peak")
elif min_sf is not None and min_sf < 1:
    status = ("crit", f"{pretty(min_sf_link)} safety factor {min_sf:.2f}")
elif over_cont:
    a_ = max(over_cont, key=lambda a: a.util_cont)
    status = ("warn", f"{short(a_.name)} motor at {a_.util_cont * 100:.0f} % of continuous")
elif min_sf is not None and min_sf < 2:
    status = ("warn", f"{pretty(min_sf_link)} safety factor {min_sf:.2f}")
else:
    status = ("good", "All motors within continuous rating"
              + (f" · lowest safety factor {min_sf:.1f}" if min_sf is not None else ""))

with st.container(horizontal=True, vertical_alignment="bottom", gap="medium"):
    with st.container():
        st.title("Leg load calculator")
        st.html('<p class="sub">One humanoid leg, held still. Set a pose and a foot load; the calculator gives each motor\'s '
                'torque against its rating, the loads in every link and bearing, and the ankle rod forces.</p>')
    _pills = pill(*status)
    if cfg.placeholder_lines:
        _pills += pill("warn", f"{len(cfg.placeholder_lines)} placeholder values in the config")
    st.html(f'<div class="pills">{_pills}</div>', width="content")
for e in res.errors:
    st.error(e, icon=":material/error:")

# ---- per-joint summary table (used in the band) -----------------------------------------------------------------
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


def _fill(kind: str) -> str:
    return f"background-color: {viz.status_fill(kind)}; color: {viz.status_ink(kind)}"


def style_rows(df: pd.DataFrame):
    """Colour only the cell that has the problem (same colours as the diagram labels): the motor-share cells by the
    joint's motor status, the safety-factor cells by the link's safety factor; hide the helper columns."""
    def cell_styles(d: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame("", index=d.index, columns=d.columns)
        for i, r in d.iterrows():
            motor = _fill("over_peak") if r["_status"] == "over_peak" else _fill("warn") if r["_cont"] else ""
            out.loc[i, ["Own Share [% Cont]", "Own Share [% Peak]"]] = motor
            sf = r["Safety Factor"]
            if sf is not None and pd.notna(sf):
                out.loc[i, ["Safety Factor", "Peak Stress [MPa]"]] = (
                    _fill("over_peak") if sf < 1.0 else _fill("warn") if sf < 2.0 else "")
        return out
    return (df.style.apply(cell_styles, axis=None).hide(axis="columns", subset=["_status", "_cont"])
            .format({"Angle [°]": "{:.1f}", "Torque [N·m]": "{:.2f}", "From GRF": "{:.2f}", "From Gravity": "{:.2f}",
                     "Own Share [% Cont]": "{:.0f}", "Own Share [% Peak]": "{:.0f}", "Peak Bending [N·m]": "{:.2f}",
                     "Peak Stress [MPa]": "{:.1f}", "Safety Factor": "{:.1f}", "Bearing F [N]": "{:.0f}"}, na_rep="–"))


# ---- stage + spec sheet ------------------------------------------------------------------------------------------
stage_col, sheet_col = st.columns([1.9, 1], gap="medium")

with stage_col, st.container(border=True, key="card_stage"):
    with st.container(horizontal=True, vertical_alignment="center"):
        view = st.segmented_control("View", ["Side", "Front", "3D"], default="Side", required=True, key="view",
                                    label_visibility="collapsed")
        st.html('<div class="legend"><span><i class="sw rod1"></i>Upper ankle rod</span>'
                '<span><i class="sw rod2"></i>Lower ankle rod</span><span><i class="sw good"></i>SF ≥ 2</span>'
                '<span><i class="sw warn"></i>SF 1–2</span><span><i class="sw crit"></i>SF &lt; 1</span></div>',
                width="content")
    if view == "3D":
        st.plotly_chart(viz.fig_3d(cfg, res), width="stretch", theme="streamlit", key="fig3d")
        st.caption("Drag to orbit, scroll to zoom. Frame triads: x red, y green, z blue, for the pelvis and each joint.")
    else:
        st.plotly_chart(viz.single_view(cfg, res, "sagittal" if view == "Side" else "coronal"), width="stretch",
                        theme="streamlit", key="stick")
        st.caption("Joint labels: angle, torque and the motor load that joint causes (border colour = motor status). "
                   "Link labels: peak bending moment [N·m], coloured by safety factor.")
    if acts:
        _ro = "".join(
            f'<div class="ro"><span class="k"><i class="dot {CLS[a.status]}"></i>'
            f'{html.escape(short(a.name))}</span><span class="v">{a.value:+.1f}</span>'
            f'<span class="x">N·m · {a.util_peak * 100:.0f} % pk · {a.util_cont * 100:.0f} % cont</span></div>'
            for a in acts)
        st.html(f'<div class="readout">{_ro}</div>')


def _dd(text: str, kind: str = "") -> str:
    return f'<dd class="{kind}">{html.escape(text)}</dd>'


with sheet_col, st.container(border=True, key="card_sheet"):
    st.subheader("Load summary")
    boxes = ""
    if worst is not None:
        k_ = CLS[worst.status]
        boxes += (f'<div class="box"><div class="k"><i class="dot {k_}">'
                  f'</i>Busiest motor</div><div class="v {k_}">{worst.util_peak * 100:.0f}'
                  f'<small>% of peak</small></div><div class="n">{html.escape(short(worst.name))} · '
                  f'{abs(worst.value):.1f} N·m</div></div>')
    if min_sf is not None:
        k_ = "crit" if min_sf < 1 else "warn" if min_sf < 2 else "good"
        boxes += (f'<div class="box"><div class="k"><i class="dot {k_}"></i>Lowest safety factor'
                  f'</div><div class="v {k_}">{min_sf:.1f}<small>yield</small></div>'
                  f'<div class="n">{html.escape(pretty(min_sf_link))}</div></div>')
    parts = [f'<div class="boxes">{boxes}</div>'] if boxes else []
    rows_ = ""
    for i, j in enumerate(cfg.joints):
        sh = res.shares[i] if res.shares else None
        kind = "" if sh is None else ("crit" if sh.status == "over_peak" else "warn" if sh.exceeds_continuous else "")
        rows_ += (f"<dt>{html.escape(pretty(j.name))}</dt>"
                  + _dd(f"{res.static.tau[i]:+.1f} N·m" + ("" if sh is None else f" · {sh.util_cont * 100:.0f} %"), kind))
    parts.append(f'<p class="spech">Joint torques · own share of continuous</p><dl class="spec">{rows_}</dl>')
    F_ = inp.load.force
    load_rows = (f"<dt>Vertical</dt>{_dd(f'{F_[2]:.0f} N · {F_[2] / weight:.2f} BW')}"
                 f"<dt>Fore-aft / lateral</dt>{_dd(f'{F_[0]:+.0f} / {F_[1]:+.0f} N')}"
                 f"<dt>Centre of pressure</dt>{_dd(f'{cop_x:+.0f}, {cop_y:+.0f} mm')}")
    parts.append(f'<p class="spech">Foot load</p><dl class="spec">{load_rows}</dl>')
    if res.rods:
        rod_rows = "".join(f"<dt>{html.escape(short(rd.name))} rod</dt>"
                           + _dd(f"{rd.force:+.0f} N {'tension' if rd.force > 0 else 'comp.'}") for rd in res.rods)
        rod_rows += f"<dt>J5 bearing</dt>{_dd(f'{np.linalg.norm(res.flanges[4].force_local):.0f} N')}"
        parts.append(f'<p class="spech">Ankle linkage</p><dl class="spec">{rod_rows}</dl>')
    if res.warnings:
        items = ""
        for w in res.warnings:
            kind = "crit" if ("PEAK" in w or "< 1" in w) else "warn"
            items += f'<li><i class="dot {kind}"></i><span>{html.escape(w)}</span></li>'
        parts.append(f'<p class="spech">Flags</p><ul class="flags">{items}</ul>')
    s = res.singularity
    parts.append(f'<p class="fine">Leg reach {s["reach_ratio"] * 100:.1f} % of full extension · computed in '
                 f'{t_calc:.1f} ms</p>')
    st.html("".join(parts))

# ---- analysis band -----------------------------------------------------------------------------------------------
with st.container(border=True, key="card_act"):
    st.subheader("Actuators")
    st.caption("Each motor's load against its ratings. Green under 60 % of continuous, amber up to peak, red over peak; "
               "anything over 100 % of continuous heats up if the pose is held.")
    if acts:
        st.plotly_chart(viz.util_bars(cfg, res), width="stretch", theme="streamlit", key="util")
        st.dataframe(pd.DataFrame([{
            "Actuator": pretty(a.name), "Load": f"{a.value:+.2f} {a.unit}", "Continuous": a.continuous, "Peak": a.peak,
            "% Cont": round(a.util_cont * 100, 1), "% Peak": round(a.util_peak * 100, 1), "Status": pretty(a.status)}
            for a in acts]), hide_index=True, width="stretch")
    if res.rods:
        st.dataframe(pd.DataFrame([{
            "Rod": pretty(rd.name), "Force [N]": rd.force, "State": "tension" if rd.force > 0 else "compression",
            "Lever Angle [°]": float(np.degrees(rd.lever_angle)), "Effective Lever Arm [mm]": abs(rd.arm) * 1000,
            "Rod Length [mm]": rd.length * 1000} for rd in res.rods]).style.format(
            {"Force [N]": "{:+.0f}", "Lever Angle [°]": "{:+.1f}", "Effective Lever Arm [mm]": "{:.1f}",
             "Rod Length [mm]": "{:.1f}"}), hide_index=True, width="stretch")
        st.caption("Rod force = motor torque / effective lever arm (the perpendicular distance from the motor shaft to the "
                   "rod line). Rods are ideal two-force members: check the rod-end bearings and buckling against these.")

with st.container(border=True, key="card_joints"):
    st.subheader("Per-joint summary")
    st.caption("Motor cells: amber over continuous, red over peak. Safety-factor cells: amber under 2, red under 1. "
               "*Own share* is the motor load from that joint's torque alone; the two ankle joints share the two ankle "
               "motors, so each motor's combined load is in the Actuators card.")
    st.dataframe(style_rows(joint_table()), width="stretch", hide_index=True, height=260,
                 column_config={"_status": None, "_cont": None})          # helper columns (drive the colours)
    st.dataframe(pd.DataFrame({"Joint": [pretty(n) for n in cfg.joint_names], "Total [N·m]": res.static.tau,
                               "Ground Reaction [N·m]": res.static.tau_grf,
                               "Gravity of Links [N·m]": res.static.tau_gravity}).round(2),
                 hide_index=True, width="stretch")

with st.container(border=True, key="card_links"):
    st.subheader("Link loads")
    if res.rods is None:
        st.warning("**Ankle rods not included.** The rod forces could not be computed at this pose, so the lower shank, "
                   "ankle cross, foot and J5 / J6 bearing loads below use the plain cut method and are not valid.")
    else:
        st.caption("Free-body cuts along each link, with the two ankle rods in the load path (each rod's force is added to "
                   "every cut it crosses). Not modelled: rod-end moments, the motor mounts, and stress in the levers, rods "
                   "and pickups.")
    names = [lr.name for lr in res.links]
    pick = st.selectbox("Link", names, index=min(2, len(names) - 1), format_func=pretty)
    lr = res.links[names.index(pick)]
    if lr.two_force:
        st.caption("Two-force member: only tension / compression is meaningful. "
                   + (lr.two_force_violation or "Transverse loads are negligible."))
    st.plotly_chart(viz.link_diagram(lr), width="stretch", theme="streamlit", key="linkdiag")
    st.dataframe(pd.DataFrame([{
        "Distance s [mm]": s_.s * 1000, "N [N]": s_.N, "Vy [N]": s_.Vy, "Vz [N]": s_.Vz, "T [N·m]": s_.T, "My [N·m]": s_.My,
        "Mz [N·m]": s_.Mz, "σ [MPa]": None if s_.sigma is None else s_.sigma / 1e6, "SF": s_.sf, "Even Station": s_.even}
        for s_ in lr.stations]), hide_index=True, width="stretch")
    st.caption("Beam frame: x along the link, y from the section's y_ref, z = x × y. N > 0 is tension. Stress = "
               "Kt(|N|/A + |My|c_z/I_y + |Mz|c_y/I_z); shear and torsion are reported but not in the stress. Unticked "
               "rows are extra stations at lumped masses and section changes, so the true peak is not missed.")
    st.markdown("**Joint bearing / output-flange reactions** (joint frame)")
    st.dataframe(pd.DataFrame([{
        "Joint": pretty(f.joint), "Fx": f.force_local[0], "Fy": f.force_local[1], "Fz": f.force_local[2],
        "Axial Thrust [N]": f.axial_force, "Radial [N]": f.radial_force, "Tilt Moment [N·m]": f.tilt_moment,
        "Axis Moment [N·m]": f.axis_moment} for f in res.flanges]).round(2), hide_index=True, width="stretch")


@st.cache_data(show_spinner="Sweeping the stance phase...", max_entries=32)
def _sweep(_cfg, gait_name: str, hardware: tuple, mtime: float, payload_kg: float):
    """Cached stance sweep.  ``_cfg`` is not hashed, so everything that changes the result is an explicit argument:
    the gait, the selected hardware options, the config file's mtime and the payload."""
    return gait_mod.stance_sweep(_cfg, _cfg.gaits[gait_name], 41, payload_kg)


if cfg.gaits:
    with st.container(border=True, key="card_gait"):
        st.subheader("Gait cycle")
        _gnames = list(cfg.gaits)
        _gsel = st.selectbox("Gait", _gnames, format_func=pretty, key="gait_pick")
        GS = cfg.gaits[_gsel]
        st.caption(GS.description)
        sw = _sweep(cfg, _gsel, tuple(sorted(cfg.active_choices.items())), CONFIG_PATH.stat().st_mtime, payload)
        rows = gait_mod.summarize(cfg, GS, sw)
        st.plotly_chart(viz.walk_load_plot(cfg, GS, sw), width="stretch", theme="streamlit", key="walk_load")
        tbl = pd.DataFrame([{
            "Actuator": pretty(r_["name"]), "Model": r_["model"], "Peak Load": r_["peak"], "% of Peak Rating": r_["peak_pct"],
            "RMS (Stance)": r_["rms_stance"], "RMS % of Continuous (Stance)": r_["rms_stance_pct"],
            "_bad": r_["peak_pct"] > 100, "_warn": r_["rms_stance_pct"] > 100} for r_ in rows])

        def _gait_style(r_):
            col = _fill("over_peak") if r_["_bad"] else _fill("warn") if r_["_warn"] else ""
            return [col] * len(r_)

        st.dataframe(tbl.style.apply(_gait_style, axis=1).hide(axis="columns", subset=["_bad", "_warn"]).format(
            {"Peak Load": "{:.1f}", "% of Peak Rating": "{:.0f}", "RMS (Stance)": "{:.1f}",
             "RMS % of Continuous (Stance)": "{:.0f}"}),
            hide_index=True, width="stretch", column_config={"_bad": None, "_warn": None})
        st.caption("Red: peak above the motor's peak rating. Amber: RMS over the stance above the continuous rating. "
                   "Held-still (quasi-static) loads: ground force and the leg's own weight only, no leg inertia and no "
                   "swing phase, so the hip and knee are under-stated. The joint angles, force shares and centre-of-"
                   "pressure path are typical human values scaled to this robot, not robot data.")

band_l, band_r = st.columns(2, gap="medium")
with band_l, st.container(border=True, key="card_check"):
    st.subheader("Hand check")
    st.caption("A joint torque from the ground force is about the vertical force times the forward distance from the "
               "joint to where the force acts (plus the fore-aft force times the height). The two right-hand columns "
               "should agree; any difference is pelvis tilt or hip abduction / rotation. The tool then adds the leg's "
               "own weight.")
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
    if res.T is not None and res.T.shape == (6, 6):
        _T = res.T[4:, 4:]
        st.caption(f"Ankle: each motor's torque is set by the lever ratio T (motor angle per ankle angle). Here pitch "
                   f"{_T[0, 0]:+.2f} / {_T[1, 0]:+.2f} and roll {_T[0, 1]:+.2f} / {_T[1, 1]:+.2f} for the upper / lower "
                   f"motor. A ratio below 1 means the motors must supply MORE torque than the ankle sees; the two "
                   f"motors share the pitch torque.")

with band_r, st.container(border=True, key="card_export"):
    st.subheader("Export")
    st.caption("Packages the current pose, inputs and all results for download"
               + (" and also writes them to `exports/`." if LOCAL else ".")
               + " To save a picture of any figure, use the camera icon in the figure's toolbar.")
    if st.button("Export report (JSON + CSV" + (" + PNG)" if LOCAL else ")"), icon=":material/download:"):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        js, cs = report.to_json(cfg, res), report.to_csv(cfg, res)
        ss.export = {"stamp": stamp, "json": js, "csv": cs, "png": None, "png_err": None}
        if LOCAL:
            EXPORT_DIR.mkdir(exist_ok=True)
            (EXPORT_DIR / f"leg_report_{stamp}.json").write_text(js, encoding="utf-8")
            (EXPORT_DIR / f"leg_report_{stamp}.csv").write_text(cs, encoding="utf-8")
            try:
                with viz.theme("light"):                # a file has no viewer theme: draw it in the light palette
                    png = viz.stick_figures(cfg, res).to_image(format="png", width=1500, height=700, scale=2)
                (EXPORT_DIR / f"leg_figure_{stamp}.png").write_bytes(png)
                ss.export["png"] = png
            except Exception as e:  # kaleido needs a Chrome install; degrade gracefully
                ss.export["png_err"] = str(e)
    ex = ss.get("export")
    if ex:
        if LOCAL:
            st.caption(f"Saved to {EXPORT_DIR} (stamp {ex['stamp']}).")
        with st.container(horizontal=True):
            st.download_button("Download JSON", ex["json"], f"leg_report_{ex['stamp']}.json", "application/json")
            st.download_button("Download CSV", ex["csv"], f"leg_report_{ex['stamp']}.csv", "text/csv")
            if ex["png"]:
                st.download_button("Download PNG", ex["png"], f"leg_figure_{ex['stamp']}.png", "image/png")
        if not ex["png"] and LOCAL:
            st.caption("PNG export failed (Plotly's image exporter, kaleido, needs Chrome — run `plotly_get_chrome` "
                       "once). Details: " + str(ex["png_err"])[:300])

with st.container(border=True, key="card_conv"):
    st.subheader("Conventions")
    st.html('<div class="conv"><dl>'
            '<dt>Frame</dt><dd>Origin on the pelvis centre line at the hip-pitch (J1) height. +X forward, +Y left, +Z up. '
            'The left leg sits at y = +137.5 mm.</dd>'
            '<dt>Joints</dt><dd>Hip flexion, knee flexion and ankle dorsiflexion are positive, so a squat has three '
            'positive angles. Hip abduction and ankle inversion are positive toward +Y.</dd>'
            '<dt>Torque</dt><dd>The torque the motor applies to the outer link about the joint axis to hold the pose. '
            'Compare magnitudes; the sign gives the direction.</dd>'
            '<dt>Foot load</dt><dd>The ground reaction on the foot, world frame, at the contact point under the ankle '
            'shifted by the centre of pressure. BW = one robot body weight (payload included).</dd>'
            '<dt>Ankle</dt><dd>Two motors on the shank drive two levers and two ball-joint rods to the foot. Pitch turns '
            'both motors together, roll turns them against each other.</dd>'
            '<dt>Limits</dt><dd>Held-still (quasi-static) loads: no leg inertia, no swing phase, no torso model. Yield-'
            'based safety factors with a guessed stress concentration; fatigue and buckling are not checked.</dd>'
            '</dl></div>')
    if cfg.placeholder_lines:
        with st.expander(f"{len(cfg.placeholder_lines)} values in leg_config.yaml are still placeholders",
                         icon=":material/warning:"):
            st.caption("Results are only as good as these guessed values (geometry, masses, sections, ratings).")
            st.code("\n".join(f"{n:4d}: {t}" for n, t in cfg.placeholder_lines), language="yaml")
