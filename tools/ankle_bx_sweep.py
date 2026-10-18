"""Sweep the fore-aft position of the ankle rod pickups on the foot and plot what it does to the RSU ankle.

Follows the RSU model of Cervettini et al., "A Framework for Optimal Ankle Design of Humanoid Robots" (Humanoids 2025):
``b_i = U_i - U_0`` is the rod pickup on the foot in the foot frame (Eq. 5), here ``pickup_foot[i]`` of the
``leg.linkages:ankle_parallel_jacobian`` transmission in leg_config.yaml.  Only its x component is swept; everything else
(lever length, lever pivots, pickup y / z, motors) stays as configured.

For every candidate ``b_x`` the ankle range of motion (the J5 / J6 joint limits) is gridded and at each pose we compute

* the lever (crank) angles -> required crank travel, and whether the rods can close the loop at all (Eq. 9);
* the lever-rod transmission quality ``sin(mu)`` (1 = rod perpendicular to the lever, 0 = dead-centre);
* ``T = d(lever angles) / d(pitch, roll)`` and from it, with both motors at their limits, the largest PURE pitch
  (resp. roll) torque and speed the ankle can deliver, and their ratio (torque-to-speed, N*m*s/rad);
* the motor torque / speed the task set needs (walking stance sweep + static presets), as % of the motor ratings.

Rod length: by default it is re-derived for each ``b_x`` so the lever stays perpendicular to the shin at the neutral
pose (how the current design is defined, see the config note).  ``--keep-rod`` keeps the configured rod lengths instead
(the lever then starts tilted).

    python tools/ankle_bx_sweep.py                       # writes exports/ankle_bx_sweep.html
    python tools/ankle_bx_sweep.py --bx -40 130 5        # sweep range / step in mm
    python tools/ankle_bx_sweep.py --keep-rod --step-deg 5

Run from the repository root (``leg_calc``).
"""
import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from leg import Inputs, analyze, default_load, gait, load_config  # noqa: E402
from leg.kinematics import rot_axis  # noqa: E402
from leg.linkages import solve_ankle_actuator_angles  # noqa: E402

MIN_SIN_MU = np.sin(np.radians(45))   # usable if the lever-rod angle stays within 45..135 deg everywhere in the ROM
                                       # (rod force = motor torque / (lever * sin mu): 1.41x its minimum at 45 deg)


def ankle_group(cfg):
    for spec in cfg.transmissions:
        if spec.function == "leg.linkages:ankle_parallel_jacobian":
            return spec
    raise SystemExit("no ankle_parallel_jacobian transmission in the config")


def params_for(base: dict, bx: float, keep_rod: bool) -> dict:
    p = copy.deepcopy(base)
    p["pickup_foot"] = [[bx, b[1], b[2]] for b in base["pickup_foot"]]
    if not keep_rod:
        # rod length that closes the loop with the lever at neutral_angle and the ankle at q = 0
        th = p.get("neutral_angle", 0.0)
        rods = []
        for i, b in enumerate(p["pickup_foot"]):
            P = np.asarray(p["j5_to_j6"], float) + np.asarray(b, float)
            tip = np.array([0.0, p["lever_y"][i], p["actuator_axis_z"][i]]) + p["lever_length"] * np.array(
                [np.cos(th), 0.0, -np.sin(th)])
            rods.append(float(np.linalg.norm(P - tip)))
        p["rod_length"] = rods
    return p


def pose_metrics(q, p):
    """Lever angles, signed transmission sine per rod, and T at ankle pose ``q`` (raises ValueError if unreachable)."""
    th = solve_ankle_actuator_angles(q, **p)
    R5 = rot_axis(np.asarray(p["ankle_pitch_axis"], float), q[0])
    R6 = R5 @ rot_axis(np.asarray(p["ankle_roll_axis"], float), q[1])
    p6 = R5 @ np.asarray(p["j5_to_j6"], float)
    s = np.zeros(2)
    for i in range(2):
        P = p6 + R6 @ np.asarray(p["pickup_foot"][i], float)
        u = np.array([np.cos(th[i]), 0.0, -np.sin(th[i])])
        tip = np.array([0.0, p["lever_y"][i], p["actuator_axis_z"][i]]) + p["lever_length"] * u
        d = (P - tip) / np.linalg.norm(P - tip)
        s[i] = np.cross(u, d)[1]                       # sin of the lever-rod angle, about the lever axis (+y)
    eps = 1e-6
    T = np.zeros((2, 2))
    for j in range(2):
        dq = np.zeros(2)
        dq[j] = eps
        T[:, j] = (solve_ankle_actuator_angles(q + dq, **p) - solve_ankle_actuator_angles(q - dq, **p)) / (2 * eps)
    return th, s, T


def capability(T, tau_pk, w_max):
    """Largest pure-pitch / pure-roll joint torque and speed with every motor inside its box limits."""
    Tit = np.linalg.inv(T).T                           # tau_act = T^-T tau_joint
    tau = np.array([1.0 / np.max(np.abs(Tit[:, j]) / tau_pk) for j in range(2)])
    w = np.array([1.0 / np.max(np.abs(T[:, j]) / w_max) for j in range(2)])
    return tau, w


def task_demands(cfg, spec):
    """(q_ankle [rad], tau_ankle [N*m], qd_ankle [rad/s] or None, label) for the walking sweep and the presets."""
    j = spec.joints
    out = []
    for g in cfg.gaits.values():
        sw = gait.stance_sweep(cfg, g)
        for k in range(len(sw["phase"])):
            out.append((np.radians(sw["q"][k][j]), sw["tau"][k][j], np.radians(sw["qd_deg"][k][j]),
                        f"{g.name or 'gait'} {sw['phase'][k]:.2f}"))
    for name, pr in cfg.presets.items():
        if pr.motion:
            continue                                     # gait-driven presets are covered by the sweep above
        r = analyze(cfg, Inputs(q=pr.q, load=default_load(cfg, pr.load_fraction)))
        out.append((pr.q[j], r.static.tau[j], None, name))
    return out


def sweep(cfg, bxs, keep_rod, step_deg):
    spec = ankle_group(cfg)
    acts = [cfg.actuators[i] for i in spec.actuators]
    tau_pk = np.array([a.peak for a in acts])
    w_max = np.array([a.max_speed for a in acts])
    (p_lo, p_hi), (r_lo, r_hi) = (cfg.joints[spec.joints[0]].limits, cfg.joints[spec.joints[1]].limits)
    pitch = np.radians(np.arange(np.degrees(p_lo), np.degrees(p_hi) + 1e-9, step_deg))
    roll = np.radians(np.arange(np.degrees(r_lo), np.degrees(r_hi) + 1e-9, step_deg))
    tasks = task_demands(cfg, spec)

    rows, maps = [], {}
    for bx in bxs:
        p = params_for(spec.params, bx, keep_rod)
        shape = (len(pitch), len(roll))
        TH = np.full(shape + (2,), np.nan)
        S = np.full(shape + (2,), np.nan)
        TAU = np.full(shape + (2,), np.nan)
        W = np.full(shape + (2,), np.nan)
        ok = True
        try:
            _, s0, _ = pose_metrics(np.zeros(2), p)
        except ValueError:
            ok, s0 = False, np.ones(2)
        for a, qp in enumerate(pitch):
            for b, qr in enumerate(roll):
                try:
                    th, s, T = pose_metrics(np.array([qp, qr]), p)
                except (ValueError, np.linalg.LinAlgError):
                    ok = False
                    continue
                if np.any(np.sign(s) != np.sign(s0)):    # crossed a dead-centre: the other assembly branch
                    ok = False
                    continue
                TH[a, b], S[a, b] = th, np.abs(s)
                TAU[a, b], W[a, b] = capability(T, tau_pk, w_max)
        # task demands at this geometry
        u_tau, u_w = 0.0, 0.0
        worst_tau = worst_w = ""
        for q, tau, qd, label in tasks:
            try:
                _, _, T = pose_metrics(q, p)
                ta = np.linalg.solve(T.T, tau)
            except (ValueError, np.linalg.LinAlgError):
                u_tau, worst_tau = np.inf, label
                continue
            ut = np.max(np.abs(ta) / tau_pk)
            if ut > u_tau:
                u_tau, worst_tau = ut, label
            if qd is not None:
                uw = np.max(np.abs(T @ qd) / w_max)
                if uw > u_w:
                    u_w, worst_w = uw, label
        ratio = TAU / W
        rows.append(dict(
            bx=bx, rod=p["rod_length"], feasible=ok,
            travel=np.nanmax(TH, axis=(0, 1)) - np.nanmin(TH, axis=(0, 1)),
            th_min=np.nanmin(TH, axis=(0, 1)), th_max=np.nanmax(TH, axis=(0, 1)),
            sin_mu_min=np.nanmin(S),
            tau_min=np.nanmin(TAU, axis=(0, 1)), tau_max=np.nanmax(TAU, axis=(0, 1)),
            w_min=np.nanmin(W, axis=(0, 1)), w_max=np.nanmax(W, axis=(0, 1)),
            ratio_min=np.nanmin(ratio, axis=(0, 1)), ratio_max=np.nanmax(ratio, axis=(0, 1)),
            task_tau_pct=100 * u_tau, task_tau_case=worst_tau, task_w_pct=100 * u_w, task_w_case=worst_w,
        ))
        maps[bx] = dict(TAU=TAU, W=W, S=S, TH=TH)
    return rows, maps, np.degrees(pitch), np.degrees(roll), tasks, spec


def usable(r):
    return r["feasible"] and r["sin_mu_min"] >= MIN_SIN_MU


def figure(rows, maps, pitch_deg, roll_deg, tasks, bx_now, bx_pick, keep_rod, spec):
    bx = np.array([r["bx"] for r in rows]) * 1000
    col = lambda k, i=None: np.array([r[k] if i is None else r[k][i] for r in rows], float)  # noqa: E731
    req_tau = np.max([np.abs(t[1]) for t in tasks], axis=0)
    req_w = np.max([np.abs(t[2]) for t in tasks if t[2] is not None], axis=0)
    blue, orange, grey = "#2a6fdb", "#e07b1a", "#888"

    fig = make_subplots(
        rows=4, cols=2, vertical_spacing=0.07, horizontal_spacing=0.09,
        specs=[[{}, {}], [{}, {}], [{}, {}], [{"type": "heatmap"}, {"type": "heatmap"}]],
        subplot_titles=(
            "Crank (lever) travel over the ROM", "Lever-rod transmission quality, worst pose",
            "Worst-case available ankle torque (both motors at peak)", "Worst-case available ankle speed (motors at no-load speed)",
            "Torque-to-speed ratio over the ROM (band = min..max)", "Task demand on the motors (walking + presets)",
            f"Available pitch torque over the ROM, current b_x = {bx_now * 1000:.1f} mm",
            f"Available pitch torque over the ROM, b_x = {bx_pick * 1000:.1f} mm"))

    def vline(r, c):
        fig.add_vline(x=bx_now * 1000, line=dict(color=grey, dash="dash"), row=r, col=c)
        fig.add_vline(x=bx_pick * 1000, line=dict(color="#2a9d4b", dash="dot"), row=r, col=c)

    bad = [r["bx"] * 1000 for r in rows if not usable(r)]
    for r_ in range(1, 4):
        for c_ in (1, 2):
            for x in bad:
                fig.add_vrect(x0=x - 2.5, x1=x + 2.5, fillcolor="#d33", opacity=0.08, line_width=0, row=r_, col=c_)
            vline(r_, c_)

    names = ["upper motor", "lower motor"]
    for i, c in enumerate((blue, orange)):
        fig.add_trace(go.Scatter(x=bx, y=np.degrees(col("travel", i)), name=f"travel, {names[i]}",
                                 line=dict(color=c)), 1, 1)
        fig.add_trace(go.Scatter(x=bx, y=np.degrees(col("th_max", i)), name=f"max lever angle, {names[i]}",
                                 line=dict(color=c, dash="dot")), 1, 1)
        fig.add_trace(go.Scatter(x=bx, y=np.degrees(col("th_min", i)), name=f"min lever angle, {names[i]}",
                                 line=dict(color=c, dash="dashdot")), 1, 1)
    fig.add_trace(go.Scatter(x=bx, y=np.degrees(np.arcsin(np.clip(col("sin_mu_min"), 0, 1))),
                             name="min lever-rod angle", line=dict(color="#555")), 1, 2)
    fig.add_hline(y=np.degrees(np.arcsin(MIN_SIN_MU)), line=dict(color="#d33", dash="dash"), row=1, col=2)

    for j, (c, ax) in enumerate(((blue, "pitch"), (orange, "roll"))):
        fig.add_trace(go.Scatter(x=bx, y=col("tau_min", j), name=f"{ax} torque, worst pose", line=dict(color=c)), 2, 1)
        fig.add_hline(y=req_tau[j], line=dict(color=c, dash="dot"), row=2, col=1,
                      annotation_text=f"task peak {ax} {req_tau[j]:.1f} N*m", annotation_font_color=c)
        fig.add_trace(go.Scatter(x=bx, y=col("w_min", j), name=f"{ax} speed, worst pose", line=dict(color=c)), 2, 2)
        fig.add_hline(y=req_w[j], line=dict(color=c, dash="dot"), row=2, col=2,
                      annotation_text=f"walking peak {ax} {req_w[j]:.1f} rad/s", annotation_font_color=c)
        fig.add_trace(go.Scatter(x=np.r_[bx, bx[::-1]], y=np.r_[col("ratio_max", j), col("ratio_min", j)[::-1]],
                                 fill="toself", fillcolor=c, opacity=0.25, line=dict(width=0),
                                 name=f"{ax} torque/speed band"), 3, 1)
        fig.add_trace(go.Scatter(x=bx, y=col("ratio_max", j), name=f"{ax} peak torque/speed", line=dict(color=c)), 3, 1)
    fig.add_trace(go.Scatter(x=bx, y=col("task_tau_pct"), name="peak motor torque, % of peak rating",
                             line=dict(color="#7b3fb5")), 3, 2)
    fig.add_trace(go.Scatter(x=bx, y=col("task_w_pct"), name="peak motor speed, % of no-load (walking)",
                             line=dict(color="#2a9d4b")), 3, 2)
    fig.add_hline(y=100, line=dict(color="#d33", dash="dash"), row=3, col=2)

    zmax = max(np.nanmax(maps[bx_now]["TAU"][..., 0]), np.nanmax(maps[bx_pick]["TAU"][..., 0]))
    for c_, b in ((1, bx_now), (2, bx_pick)):
        fig.add_trace(go.Heatmap(x=roll_deg, y=pitch_deg, z=maps[b]["TAU"][..., 0], zmin=0, zmax=zmax,
                                 colorscale="Viridis", colorbar=dict(title="N*m", len=0.2, y=0.1),
                                 showscale=c_ == 2,
                                 hovertemplate="roll %{x} deg<br>pitch %{y} deg<br>%{z:.1f} N*m<extra></extra>"), 4, c_)
        q = np.degrees([t[0] for t in tasks])
        fig.add_trace(go.Scatter(x=q[:, 1], y=q[:, 0], mode="markers", marker=dict(color="red", size=5),
                                 name="task poses", showlegend=c_ == 1), 4, c_)

    for c_ in (1, 2):
        for r_ in (1, 2, 3):
            fig.update_xaxes(title_text="pickup b_x ahead of the ankle [mm]", row=r_, col=c_)
        fig.update_xaxes(title_text="roll [deg]", row=4, col=c_)
        fig.update_yaxes(title_text="pitch (+ dorsiflexion) [deg]", row=4, col=c_)
    fig.update_yaxes(title_text="deg", row=1, col=1)
    fig.update_yaxes(title_text="deg (90 = perpendicular)", row=1, col=2)
    fig.update_yaxes(title_text="N*m", row=2, col=1)
    fig.update_yaxes(title_text="rad/s", row=2, col=2)
    fig.update_yaxes(title_text="N*m per rad/s", row=3, col=1)
    fig.update_yaxes(title_text="% of rating", row=3, col=2)
    rod_note = "configured rod lengths" if keep_rod else "rod length re-derived so the lever is perpendicular at neutral"
    fig.update_layout(
        height=1900, width=1300, template="plotly_white",
        title=dict(text=("RSU ankle: sweep of the rod pickup x position b<sub>i,x</sub> (Cervettini et al. 2025, Eq. 5)"
                         f"<br><sup>grey dash = current design, green dot = suggested; red bands = unusable "
                         f"(rod cannot close the loop, or lever-rod angle < {np.degrees(np.arcsin(MIN_SIN_MU)):.0f} deg "
                         f"somewhere in the ROM); {rod_note}; lever {spec.params['lever_length'] * 1000:.0f} mm</sup>")),
        legend=dict(orientation="h", y=-0.03))
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bx", nargs=3, type=float, default=[-40, 130, 5], metavar=("FROM", "TO", "STEP"),
                    help="b_x sweep in mm (default -40 130 5)")
    ap.add_argument("--keep-rod", action="store_true", help="keep the configured rod lengths")
    ap.add_argument("--step-deg", type=float, default=2.5, help="ROM grid step in degrees (default 2.5)")
    ap.add_argument("--out", default=str(ROOT / "exports" / "ankle_bx_sweep.html"))
    a = ap.parse_args()

    cfg = load_config(ROOT / "leg_config.yaml")
    spec = ankle_group(cfg)
    bx_now = float(spec.params["pickup_foot"][0][0])
    bxs = np.round(np.arange(a.bx[0], a.bx[1] + 1e-9, a.bx[2]) / 1000, 6)
    bxs = np.unique(np.r_[bxs, bx_now])
    rows, maps, pitch_deg, roll_deg, tasks, spec = sweep(cfg, bxs, a.keep_rod, a.step_deg)

    print(f"{'b_x mm':>7} {'rods mm':>13} {'usable':>6} {'travel deg':>12} {'min mu':>6} "
          f"{'tau_p min':>9} {'tau_r min':>9} {'w_p min':>7} {'w_r min':>7} {'T/S p max':>9} {'task tau%':>9} {'task w%':>7}")
    for r in rows:
        print(f"{r['bx'] * 1000:7.1f} {r['rod'][0] * 1000:6.1f}/{r['rod'][1] * 1000:5.1f} {('yes' if usable(r) else 'NO'):>6} "
              f"{np.degrees(r['travel'][0]):5.1f}/{np.degrees(r['travel'][1]):5.1f} "
              f"{np.degrees(np.arcsin(min(r['sin_mu_min'], 1))):6.1f} "
              f"{r['tau_min'][0]:9.1f} {r['tau_min'][1]:9.1f} {r['w_min'][0]:7.1f} {r['w_min'][1]:7.1f} "
              f"{r['ratio_max'][0]:9.2f} {r['task_tau_pct']:9.0f} {r['task_w_pct']:7.0f}")

    # suggestion: the usable b_x with the highest worst-case pitch torque, inside the foot plate
    ok = [r for r in rows if usable(r) and cfg.foot.heel_x <= r["bx"] <= cfg.foot.toe_x]
    pick = max(ok, key=lambda r: r["tau_min"][0]) if ok else min(rows, key=lambda r: abs(r["bx"] - bx_now))
    print(f"\ncurrent b_x {bx_now * 1000:.1f} mm; strongest usable b_x {pick['bx'] * 1000:.1f} mm "
          f"(worst-case pitch torque {pick['tau_min'][0]:.1f} N*m, rods {pick['rod'][0] * 1000:.1f} / "
          f"{pick['rod'][1] * 1000:.1f} mm)")

    fig = figure(rows, maps, pitch_deg, roll_deg, tasks, bx_now, pick["bx"], a.keep_rod, spec)
    Path(a.out).parent.mkdir(exist_ok=True)
    fig.write_html(a.out, include_plotlyjs=True)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
