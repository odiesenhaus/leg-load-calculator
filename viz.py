"""Plotly figures for the leg (UI side; the physics lives in ``leg/``).

Figures
    stick_figures   sagittal + coronal views of the pose from the real forward kinematics, with joint labels, peak bending
                    labels, centre-of-mass markers and the ground-reaction arrow
    fig_3d          orbitable 3D view with the frame triads, a solid foot plate and a solid GRF arrow
    util_bars       actuator utilisation against continuous / peak ratings
    link_diagram    N, V, M, T along one link
    walk_load_plot  the Gait Cycle tab

Colour language (shared by the figures and the summary tables)
    links           dark grey; once a link has a section it is coloured by its safety factor (green >= 2, amber 1-2, red < 1)
    actuator status ``DARK`` (border / text) and ``LIGHT`` (fill) for ok / warn / over_peak
    mass markers    structure COM = cyan square, actuator mass = magenta diamond, ankle rod = yellow circle (no outline)

Label placement.  Labels never sit on the model: joint labels go in a column beside the leg ordered by height, peak-bending
labels in a second column on the other side, both with leader lines and a minimum spacing (``_spread``), so they cannot
overlap however the joints bunch up.  In 3D the joint labels sit in a column to the screen-right of the default camera.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from leg import Config, Results
from leg.kinematics import foot_box_vertices, foot_mount_top, sole_corners
from leg.names import pretty

GREEN, AMBER, RED, GREY = "#2e9e5b", "#d9a400", "#d64545", "#8a8f98"
LINK_GREY = "#4a4f58"      # link / foot bodies (darker than the ground and grid greys)
INK, GRF_RED = "#2b2f36", "#c0392b"
# cyan / magenta / yellow, no outline: structure COMs, actuator masses, ankle-linkage rods
MASS_STYLE = {"structure": ("square", "#00b8d9", "Structure COM"),
              "actuator": ("diamond", "#e6008a", "Actuator Mass"),
              "linkage": ("circle", "#f5c400", "Linkage Rod")}
CAMERA_EYE = (1.6, 1.6, 0.7)   # default 3D camera (label column is placed screen-right of this view)
STATUS_COLOR = {"ok": GREEN, "warn": AMBER, "over_peak": RED}
# actuator-status colours shared by the diagram labels and the summary table:
# DARK = label border and text, LIGHT = label fill (and the table row fill for warn / over-peak)
DARK = {"ok": "#1f7a45", "warn": "#9a6b00", "over_peak": "#b3261e"}
LIGHT = {"ok": "#dcf2e4", "warn": "#fdf0c9", "over_peak": "#f8d7d7"}


def sf_color(sf: float | None) -> str:
    """Link colour from its safety factor (grey when the link has no section)."""
    if sf is None:
        return LINK_GREY
    return RED if sf < 1.0 else AMBER if sf < 2.0 else GREEN


def _joint_label(cfg: Config, r: Results, i: int) -> str:
    """"<b>Name</b> angle | tau | actuator utilisation %" for joint ``i`` (plotly HTML)."""
    s = f"<b>{pretty(cfg.joints[i].name)}</b> {np.degrees(r.inputs.q[i]):.0f}° | τ {r.static.tau[i]:.1f} N·m"
    util = joint_util(cfg, r, i)
    return s + (f" | {util * 100:.0f}%" if util is not None else "")


def joint_util(cfg: Config, r: Results, i: int) -> float | None:
    """Share of the continuous rating of the actuators driving joint ``i`` that this joint's own torque uses.
    (Coupled joints share motors, so this is the joint's own contribution, not the motors' combined load.)"""
    if r.shares is None:
        return None
    return r.shares[i].util_cont


def joint_status(cfg: Config, r: Results, i: int) -> str | None:
    """Status ('ok' | 'warn' | 'over_peak') from the joint's own share of its actuators (see ``joint_util``)."""
    if r.shares is None:
        return None
    return r.shares[i].status


def joint_color(cfg: Config, r: Results, i: int) -> str:
    """Dark status colour (label border / text, joint ring)."""
    st_ = joint_status(cfg, r, i)
    return INK if st_ is None else DARK[st_]


def joint_fill(cfg: Config, r: Results, i: int) -> str:
    """Light status colour used as the joint label fill."""
    st_ = joint_status(cfg, r, i)
    return "rgba(255,255,255,0.9)" if st_ is None else LIGHT[st_]


def _grf_style(cfg: Config, F) -> tuple[float, float, float]:
    """(length [m], 2D width [px], 3D shaft radius [m]) of the GRF arrow.

    Magnitude shows mostly as thickness: the length only grows from 0.14 m (zero load) to a 0.22 m cap.
    """
    rel = float(np.linalg.norm(F)) / (cfg.robot_total_mass * cfg.gravity)   # multiples of body weight
    length = 0.14 + min(0.08, 0.03 * rel)
    return length, min(2.0 + 3.0 * rel, 11.0), 0.003 + 0.0022 * min(rel, 4.0)


def _arrow_mesh(tail, tip, r_shaft: float, color: str, name: str, n: int = 24) -> go.Mesh3d:
    """Solid arrow in world units (so it zooms with the model): a cylinder shaft that stops at the flat
    face of a cone head whose apex is exactly ``tip``."""
    tail, tip = np.asarray(tail, float), np.asarray(tip, float)
    d = tip - tail
    L = float(np.linalg.norm(d))
    d = d / L
    r_head = 3.0 * r_shaft
    head_len = min(0.045 + 2.0 * r_shaft, 0.6 * L)
    base = tip - d * head_len
    a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0.0, 1, 0])
    u = np.cross(d, a)
    u /= np.linalg.norm(u)
    v = np.cross(d, u)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = lambda c, r: c + r * (np.outer(np.cos(ang), u) + np.outer(np.sin(ang), v))  # noqa: E731
    V = np.vstack([ring(tail, r_shaft), ring(base, r_shaft), ring(base, r_head), tail, tip])
    t0, t1, t2, ctr, apex = 0, n, 2 * n, 3 * n, 3 * n + 1
    tris = []
    for k in range(n):
        k1 = (k + 1) % n
        tris += [(t0 + k, t1 + k, t1 + k1), (t0 + k, t1 + k1, t0 + k1),       # shaft side
                 (ctr, t0 + k1, t0 + k),                                      # tail cap
                 (t2 + k, t2 + k1, apex),                                     # cone side
                 (t1 + k, t2 + k, t2 + k1), (t1 + k, t2 + k1, t1 + k1)]       # cone's flat face (annulus)
    T = np.array(tris)
    return go.Mesh3d(x=V[:, 0], y=V[:, 1], z=V[:, 2], i=T[:, 0], j=T[:, 1], k=T[:, 2], color=color, name=name,
                     flatshading=True, showlegend=True, hoverinfo="name", lighting=dict(ambient=0.7, diffuse=0.6))


def _mass_name(name: str) -> str:
    """'hip_flexion_bracket:structure' -> 'Hip Flexion Bracket: Structure'."""
    return ": ".join(pretty(part) for part in name.split(":"))


def _spread(values, gap: float) -> list[float]:
    """Positions >= ``gap`` apart that stay as close as possible to ``values`` (returned in input order)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    pos = []
    for k, i in enumerate(order):
        pos.append(values[i] if k == 0 else max(values[i], pos[-1] + gap))
    if pos:
        shift = float(np.mean([values[i] for i in order]) - np.mean(pos))
        pos = [p_ + shift for p_ in pos]
    out = [0.0] * len(values)
    for k, i in enumerate(order):
        out[i] = pos[k]
    return out


def _view_axes(view: str):
    """(horizontal, vertical) world axis indices of a view: sagittal = (x, z), coronal = (y, z)."""
    return (0, 2) if view == "sagittal" else (1, 2)


def _add_view(fig: go.Figure, cfg: Config, r: Results, view: str, row: int, col: int, side: int) -> None:
    """Draw one 2D view into subplot (row, col).  ``side`` = +1 puts the joint-label column on the right and the
    peak-moment column on the left; -1 mirrors that.  Axes are equal-scale with fixed ranges so the figure is a
    true-to-scale drawing of the forward kinematics."""
    h, v = _view_axes(view)
    fk = r.fk
    P = lambda p: (float(p[h]), float(p[v]))  # noqa: E731
    axn = "" if row == 1 else str(row)
    # ground (horizontal line through the contact point)
    cx, cz = P(fk.p_contact)
    span = 0.9
    fig.add_trace(go.Scatter(x=[cx - span, cx + span], y=[cz, cz], mode="lines",
                             line=dict(color=GREY, width=2, dash="dot"), hoverinfo="skip", showlegend=False),
                  row=row, col=col)
    # sole rectangle (follows the foot link)
    sc = np.vstack([sole_corners(cfg, fk)] * 2)[:5]
    fig.add_trace(go.Scatter(x=sc[:, h], y=sc[:, v], mode="lines", line=dict(color=LINK_GREY, width=6),
                             hoverinfo="skip", showlegend=False), row=row, col=col)
    top = foot_mount_top(cfg, fk)           # rigid post from the ankle roll axis down to the foot plate
    fig.add_trace(go.Scatter(x=[fk.p[-1][h], top[h]], y=[fk.p[-1][v], top[v]], mode="lines",
                             line=dict(color=LINK_GREY, width=7), hoverinfo="skip", showlegend=False), row=row, col=col)
    # links, coloured by safety factor (grey: no section / zero length)
    for lr in r.links:
        a, b = fk.beam_start[lr.index], fk.beam_end[lr.index]
        if np.linalg.norm(a - b) < 1e-6:
            continue
        fig.add_trace(go.Scatter(
            x=[a[h], b[h]], y=[a[v], b[v]], mode="lines", line=dict(color=sf_color(lr.min_sf), width=7),
            hovertemplate=f"{pretty(lr.name)}<br>peak M {lr.peak_bending.M_bend:.1f} N·m"
                          f"<br>SF {'–' if lr.min_sf is None else f'{lr.min_sf:.2f}'}<extra></extra>",
            showlegend=False), row=row, col=col)
    # ankle levers and rods (pivot -> lever tip -> foot pickup); the hover text gives the rod force
    for rd in r.rods or ():
        fig.add_trace(go.Scatter(
            x=[rd.pivot[h], rd.tip[h], rd.pickup[h]], y=[rd.pivot[v], rd.tip[v], rd.pickup[v]], mode="lines",
            line=dict(color=MASS_STYLE["linkage"][1], width=3), showlegend=False,
            hovertemplate=f"{pretty(rd.name)}<br>rod {rd.force:+.0f} N ({'tension' if rd.force > 0 else 'compression'})"
                          "<extra></extra>"), row=row, col=col)
    # structure COMs vs actuator masses: different symbol and colour, size ~ sqrt(mass)
    mp = fk.mass_points
    for kind, (sym, colr, label) in MASS_STYLE.items():
        ms = [m for m in mp if m.kind == kind]
        fig.add_trace(go.Scatter(
            x=[m.p[h] for m in ms], y=[m.p[v] for m in ms], mode="markers", name=label, legendgroup=kind,
            showlegend=(row == 1),
            marker=dict(symbol=sym, size=[4 + 2.5 * np.sqrt(m.mass) for m in ms], color=colr,
                        line=dict(width=0)),
            text=[f"{_mass_name(m.name)} {m.mass:.3f} kg" for m in ms], hoverinfo="text"), row=row, col=col)
    # joints
    fig.add_trace(go.Scatter(
        x=[p[h] for p in fk.p], y=[p[v] for p in fk.p], mode="markers",
        marker=dict(size=11, color="white", line=dict(width=2.5, color=[joint_color(cfg, r, i) for i in range(cfg.n)])),
        text=[pretty(n) for n in cfg.joint_names], hoverinfo="text", showlegend=False), row=row, col=col)
    fig.add_trace(go.Scatter(x=[cx], y=[cz], mode="markers", marker=dict(symbol="star", size=13, color=GRF_RED),
                             hovertext="Foot Contact Point", hoverinfo="text", showlegend=False), row=row, col=col)
    # GRF arrow (tail below the contact point so it does not hide the leg), plus label
    F = r.inputs.load.force
    Fn = float(np.linalg.norm(F))
    if Fn > 1e-6:
        Fp = np.hypot(F[h], F[v])                           # in-plane part of the force
        length, w2d, _ = _grf_style(cfg, F)
        ln = length * Fp / Fn
        if ln > 1e-4:
            d = np.array([F[h], F[v]]) / Fp
            tail = np.array([cx, cz]) - d * ln
            fig.add_annotation(x=cx, y=cz, ax=tail[0], ay=tail[1], xref=f"x{axn}", yref=f"y{axn}",
                               axref=f"x{axn}", ayref=f"y{axn}", showarrow=True, arrowhead=2, arrowsize=0.5,
                               arrowwidth=w2d, arrowcolor=GRF_RED)
            mid = (tail + np.array([cx, cz])) / 2
            fig.add_annotation(x=mid[0], y=mid[1], xref=f"x{axn}", yref=f"y{axn}", showarrow=False,
                               text=f"GRF {Fp:.0f} N" + (f" (|F|={Fn:.0f})" if abs(Fn - Fp) > 1 else ""),
                               font=dict(color=GRF_RED, size=11), xanchor="left", yanchor="middle",
                               xshift=int(w2d / 2 + 14), bgcolor="rgba(255,255,255,0.85)")
    # peak bending-moment labels in a second column on the opposite side, spread so they never overlap
    anchors = []
    for lr in r.links:
        if lr.length < 1e-3:
            continue
        pt = fk.beam_start[lr.index] + lr.peak_bending.s * lr.frame_world[:, 0]
        anchors.append((float(pt[h]), float(pt[v]), lr))
    allx = [p_[h] for p_ in fk.p] + [cx] + [m.p[h] for m in fk.mass_points]
    x_m = (min(allx) - 0.10) if side > 0 else (max(allx) + 0.10)
    ys = _spread([a_[1] for a_ in anchors], 0.05)
    for (ax_, ay_, lr), y_lab in zip(anchors, ys):
        colr = sf_color(lr.min_sf)
        fig.add_trace(go.Scatter(x=[ax_, x_m], y=[ay_, y_lab], mode="lines", line=dict(color="rgba(120,125,135,0.45)", width=1),
                                 hoverinfo="skip", showlegend=False), row=row, col=col)
        fig.add_annotation(x=x_m, y=y_lab, xref=f"x{axn}", yref=f"y{axn}", showarrow=False,
                           text=f"M {lr.peak_bending.M_bend:.1f}", font=dict(size=10, color="white"),
                           bgcolor=colr if lr.min_sf is not None else LINK_GREY, borderpad=2, opacity=0.95,
                           xanchor="right" if side > 0 else "left")
    # joint value labels: a tidy column to one side, leader lines to the joints, ordered by height
    pts = np.array([[p[h], p[v]] for p in fk.p])
    zs = pts[:, 1]
    ext = np.concatenate([pts[:, 0], [cx], [m.p[h] for m in mp], [x_m]])
    x_lab = (ext.max() + 0.12) if side > 0 else (ext.min() - 0.12)
    order = np.argsort(-zs, kind="stable")
    dy = 0.09                                   # fixed label pitch [m] so labels never overlap
    yc = 0.5 * (max(zs.max(), cz) + min(zs.min(), cz))
    slots = yc + (np.arange(cfg.n)[::-1] - (cfg.n - 1) / 2) * dy
    for rank, i in enumerate(order):
        y_lab = float(slots[rank])
        fig.add_trace(go.Scatter(x=[pts[i, 0], x_lab], y=[pts[i, 1], y_lab], mode="lines",
                                 line=dict(color="rgba(120,125,135,0.55)", width=1), hoverinfo="skip",
                                 showlegend=False), row=row, col=col)
        fig.add_annotation(x=x_lab, y=y_lab, xref=f"x{axn}", yref=f"y{axn}", showarrow=False,
                           text=_joint_label(cfg, r, int(i)), align="left" if side > 0 else "right",
                           xanchor="left" if side > 0 else "right",
                           font=dict(size=10, color=joint_color(cfg, r, int(i))),
                           bordercolor=joint_color(cfg, r, int(i)), borderwidth=1.5, borderpad=2,
                           bgcolor=joint_fill(cfg, r, int(i)))
    half_h = max(0.45, 0.5 * ((cfg.n - 1) * dy) + 0.14, 0.5 * (zs.max() - min(zs.min(), cz)) + 0.15)
    half_w = 1.75 * half_h                      # ~ plot aspect (width/height) so the equal-scale domain is full
    cxm = 0.5 * (ext.min() + ext.max()) + (0.3 if side > 0 else -0.3)   # leave room for the label column
    lo_x, hi_x = cxm - half_w, cxm + half_w
    ticks = [round(t_, 1) for t_ in np.arange(np.ceil(lo_x * 5) / 5, hi_x, 0.2)
             if lo_x + 0.07 * (hi_x - lo_x) <= t_ <= hi_x - 0.05 * (hi_x - lo_x)]   # none hugging the edges
    fig.update_xaxes(range=[lo_x, hi_x], tickmode="array", tickvals=ticks, row=row, col=col,
                     title_text="X Forward [m]" if view == "sagittal" else "Y Left [m]", zeroline=False,
                     scaleanchor=f"y{axn}", scaleratio=1, constrain="domain")
    fig.update_yaxes(range=[yc - half_h - 0.06, yc + half_h], row=row, col=col, title_text="Z Up [m]", zeroline=False,
                     constrain="domain")


def stick_figures(cfg: Config, r: Results) -> go.Figure:
    """Sagittal (top) and coronal (bottom) views of the current pose."""
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Sagittal (Side) View", "Coronal (Front) View"),
                        vertical_spacing=0.16)
    _add_view(fig, cfg, r, "sagittal", 1, 1, side=+1)
    _add_view(fig, cfg, r, "coronal", 2, 1, side=-1)
    fig.update_layout(height=1000, margin=dict(l=20, r=20, t=70, b=20),
                      legend=dict(orientation="h", x=0, y=1.06), plot_bgcolor="white", paper_bgcolor="white",
                      font=dict(family="Inter, Arial, sans-serif", color=INK))
    fig.update_xaxes(showgrid=True, gridcolor="#eceef1", automargin=True, title_standoff=14)
    fig.update_yaxes(showgrid=True, gridcolor="#eceef1", automargin=True, title_standoff=10)
    return fig


def fig_3d(cfg: Config, r: Results) -> go.Figure:
    """Orbitable 3D view with the frame triad of the pelvis and every joint."""
    fk = r.fk
    fig = go.Figure()
    # ground plane through the contact point (horizontal)
    c = fk.p_contact
    gx = np.array([c[0] - .3, c[0] + .3])    # small patch under the foot so it never sits behind the labels
    gy = np.array([c[1] - .3, c[1] + .3])
    fig.add_trace(go.Surface(x=gx, y=gy, z=np.full((2, 2), c[2]), showscale=False, opacity=0.25,
                             colorscale=[[0, "#bfc5ce"], [1, "#bfc5ce"]], hoverinfo="skip"))
    for lr in r.links:
        a, b = fk.beam_start[lr.index], fk.beam_end[lr.index]
        if np.linalg.norm(a - b) > 1e-6:
            fig.add_trace(go.Scatter3d(x=[a[0], b[0]], y=[a[1], b[1]], z=[a[2], b[2]], mode="lines",
                                       line=dict(color=sf_color(lr.min_sf), width=9), name="Links", legendgroup="links",
                                       showlegend=not any(t.name == "Links" for t in fig.data),
                                       hovertemplate=f"{pretty(lr.name)}<br>peak M {lr.peak_bending.M_bend:.1f} N·m<extra></extra>"))
    for rd in r.rods or ():
        fig.add_trace(go.Scatter3d(
            x=[rd.pivot[0], rd.tip[0], rd.pickup[0]], y=[rd.pivot[1], rd.tip[1], rd.pickup[1]],
            z=[rd.pivot[2], rd.tip[2], rd.pickup[2]], mode="lines", line=dict(color=MASS_STYLE["linkage"][1], width=5),
            showlegend=False, hovertemplate=f"{pretty(rd.name)}<br>rod {rd.force:+.0f} N<extra></extra>"))
    fig.add_trace(go.Scatter3d(x=fk.p[:, 0], y=fk.p[:, 1], z=fk.p[:, 2], mode="markers", text=[pretty(n) for n in cfg.joint_names],
                               hoverinfo="text", marker=dict(size=5, color=[joint_color(cfg, r, i) for i in range(cfg.n)],
                                                             line=dict(width=0)), name="Joints"))
    mp = fk.mass_points
    for kind, (sym, colr, label) in MASS_STYLE.items():
        ms = [m for m in mp if m.kind == kind]
        fig.add_trace(go.Scatter3d(
            x=[m.p[0] for m in ms], y=[m.p[1] for m in ms], z=[m.p[2] for m in ms], mode="markers", name=label,
            marker=dict(size=[6 + 4 * np.sqrt(m.mass) for m in ms], symbol=sym, color=colr, line=dict(width=0)),
            text=[f"{_mass_name(m.name)} {m.mass:.3f} kg" for m in ms], hoverinfo="text"))
    fig.add_trace(go.Scatter3d(x=[c[0]], y=[c[1]], z=[c[2]], mode="markers", marker=dict(size=6, color=GRF_RED),
                               name="Contact Point"))
    V = foot_box_vertices(cfg, fk)       # solid foot plate (grey, like the other links)
    tri = [(0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
           (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    fig.add_trace(go.Mesh3d(x=V[:, 0], y=V[:, 1], z=V[:, 2], i=[t[0] for t in tri], j=[t[1] for t in tri],
                            k=[t[2] for t in tri], color=LINK_GREY, flatshading=True, opacity=1.0, name="Foot Plate",
                            showlegend=True, hoverinfo="name"))
    top = foot_mount_top(cfg, fk)
    fig.add_trace(go.Scatter3d(x=[fk.p[-1][0], top[0]], y=[fk.p[-1][1], top[1]], z=[fk.p[-1][2], top[2]], mode="lines",
                               line=dict(color=LINK_GREY, width=9), hoverinfo="skip", showlegend=False))
    # frame triads: pelvis (world-aligned by R_base) and each joint frame (after rotation)
    tri_len = 0.05
    frames = [("pelvis", fk.p[0], fk.R_base)] + [(n, fk.p[i], fk.R[i]) for i, n in enumerate(cfg.joint_names)]
    for (name, o, Rm) in frames:
        for k, col in enumerate(("#d62728", "#2ca02c", "#1f77b4")):
            e = o + tri_len * Rm[:, k]
            fig.add_trace(go.Scatter3d(x=[o[0], e[0]], y=[o[1], e[1]], z=[o[2], e[2]], mode="lines",
                                       line=dict(color=col, width=4), showlegend=False,
                                       hovertemplate=f"{name} {'xyz'[k]}<extra></extra>"))
    F = r.inputs.load.force
    Fn = max(float(np.linalg.norm(F)), 1e-9)
    length, _, r_shaft = _grf_style(cfg, F)
    tail = c - F / Fn * length             # arrow starts below the foot and ends at the sole
    fig.add_trace(_arrow_mesh(tail, c, r_shaft, GRF_RED, f"GRF {np.linalg.norm(F):.0f} N"))
    # joint labels: a column beside the leg (screen-right of the default camera), ordered by height, with leaders
    eye = np.array(CAMERA_EYE)
    right = np.array([-eye[1], eye[0], 0.0])
    right /= np.linalg.norm(right)
    ctr = fk.p[:, :2].mean(axis=0)
    proj = float(np.max((np.vstack([fk.p[:, :2], c[None, :2], [m.p[:2] for m in fk.mass_points]]) - ctr) @ right[:2]))
    base = np.array([ctr[0], ctr[1]]) + right[:2] * (proj + 0.40)
    zs = fk.p[:, 2]
    order = np.argsort(-zs, kind="stable")
    pitch = max(0.075, (zs.max() - min(zs.min(), c[2]) + 0.1) / max(cfg.n - 1, 1))
    zc = 0.5 * (zs.max() + min(zs.min(), c[2]))
    slots = zc + (np.arange(cfg.n)[::-1] - (cfg.n - 1) / 2) * pitch
    lx, ly, lz, lt, lc = [], [], [], [], []
    for rank, i in enumerate(order):
        zl = float(slots[rank])
        fig.add_trace(go.Scatter3d(x=[fk.p[i, 0], base[0]], y=[fk.p[i, 1], base[1]], z=[fk.p[i, 2], zl], mode="lines",
                                   line=dict(color="rgba(120,125,135,0.6)", width=2), hoverinfo="skip", showlegend=False))
        lx.append(base[0]); ly.append(base[1]); lz.append(zl)
        lt.append(_joint_label(cfg, r, int(i)).replace("<b>", "").replace("</b>", ""))
        lc.append(joint_color(cfg, r, int(i)))
    pad = np.array([base[0], base[1]]) + right[:2] * 0.3
    fig.add_trace(go.Scatter3d(x=[pad[0], pad[0]], y=[pad[1], pad[1]], z=[min(lz) - 0.05, max(lz) + 0.05], mode="markers",
                               marker=dict(size=1, opacity=0), hoverinfo="skip", showlegend=False))
    for k_ in range(len(lt)):               # text only (no fill), coloured by actuator status
        fig.add_trace(go.Scatter3d(x=[lx[k_]], y=[ly[k_]], z=[lz[k_]], mode="text", text=[lt[k_]],
                                   textposition="middle right", textfont=dict(size=11, color=lc[k_]),
                                   hoverinfo="skip", showlegend=False))
    fig.update_layout(height=640, margin=dict(l=0, r=0, t=10, b=0), scene=dict(
        aspectmode="data", xaxis_title="X Forward", yaxis_title="Y Left", zaxis_title="Z Up",
        camera=dict(eye=dict(x=CAMERA_EYE[0], y=CAMERA_EYE[1], z=CAMERA_EYE[2]))),
        legend=dict(orientation="h"))
    return fig


def util_bars(cfg: Config, r: Results) -> go.Figure:
    """Actuator utilisation as % of continuous, with the 60 % line and one peak line per motor model."""
    acts = r.actuators or []
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=[pretty(a.name) for a in acts], x=[a.util_cont * 100 for a in acts], orientation="h",
        marker_color=[STATUS_COLOR[a.status] for a in acts],
        text=[f"{abs(a.value):.1f} {a.unit} · {a.util_cont * 100:.0f}% cont · {a.util_peak * 100:.0f}% peak" for a in acts],
        textposition="outside", cliponaxis=False, showlegend=False, textfont=dict(color=INK)))
    # one dashed line per motor model at peak/continuous; labels staggered in rows above the plot
    models: dict[tuple, str] = {}
    for spec in cfg.actuators:
        key = (round(spec.peak / spec.continuous * 100, 1), spec.peak, spec.continuous)
        models.setdefault(key, spec.model or pretty(spec.name))
    dashes = ["dash", "dot", "dashdot", "longdash"]
    rows = [("60% of Continuous", 60.0, GREY, "dot"), ("Continuous (100%)", 100.0, AMBER, "dash")]
    rows += [(f"{name} Peak {k[1]:g} ({k[0]:.0f}%)", k[0], RED, dashes[i % 4])
             for i, (k, name) in enumerate(sorted(models.items()))]
    for row_i, (label, x, colr, dash) in enumerate(rows):
        fig.add_vline(x=x, line=dict(color=colr, dash=dash, width=1.5))
        fig.add_annotation(x=x, xref="x", y=1, yref="paper", yshift=12 + 17 * row_i, text=label, showarrow=False,
                           font=dict(size=11, color=colr if colr != GREY else INK), xanchor="center", yanchor="bottom",
                           bgcolor="rgba(255,255,255,0.9)")
    xmax = max([a.util_cont * 100 for a in acts] + [k[0] for k in models] + [110]) * 1.12
    fig.update_layout(height=140 + 46 * max(len(acts), 1) + 17 * len(rows),
                      margin=dict(l=20, r=230, t=40 + 17 * len(rows), b=50),
                      xaxis=dict(title="% of Continuous Rating", range=[-0.04 * xmax, xmax], automargin=True,
                                 title_standoff=10, gridcolor="#eceef1", tickfont=dict(color=INK)),
                      yaxis=dict(autorange="reversed", automargin=True, tickfont=dict(color=INK)),
                      plot_bgcolor="white", paper_bgcolor="white", font=dict(color=INK))
    return fig


def link_diagram(lr) -> go.Figure:
    """Axial / shear, bending and torsion along one link (``lr`` is a ``structure.LinkResult``)."""
    s = [st.s * 1000 for st in lr.stations]
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Axial N / Shear V [N]", "Bending My, Mz [N·m]", "Torsion T [N·m]"))
    for name, vals, col in (("N", [t.N for t in lr.stations], 1), ("Vy", [t.Vy for t in lr.stations], 1),
                            ("Vz", [t.Vz for t in lr.stations], 1), ("My", [t.My for t in lr.stations], 2),
                            ("Mz", [t.Mz for t in lr.stations], 2), ("T", [t.T for t in lr.stations], 3)):
        fig.add_trace(go.Scatter(x=s, y=vals, mode="lines+markers", name=name), row=1, col=col)
    black = dict(color="black")
    L = max(s) if max(s) > 0 else 1.0
    fig.update_xaxes(title_text="Distance Along Link, s [mm]", title_font=black, tickfont=black, gridcolor="#e3e5e8",
                     linecolor="black", zerolinecolor="#9aa0a8", automargin=True, title_standoff=14, nticks=5,
                     range=[-0.08 * L, 1.08 * L])
    fig.update_yaxes(tickfont=black, gridcolor="#e3e5e8", linecolor="black", zerolinecolor="#9aa0a8", automargin=True,
                     nticks=6)
    fig.update_annotations(font=dict(color="black", size=13))        # subplot titles
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=60), plot_bgcolor="white", paper_bgcolor="white",
                      font=black, legend=dict(font=black))
    return fig


# ------------------------------------------------------------------ walking cycle plots
_WALK_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#17becf", "#e377c2"]


def _walk_axes(fig: go.Figure, gait, ytitle: str, xtitle: str, xticks: bool) -> None:
    """Common black-on-white styling; with ``xticks`` the x axis is labelled at the gait keyframes."""
    black = dict(color="black")
    fig.update_xaxes(title_text=xtitle, title_font=black, tickfont=black, gridcolor="#e3e5e8", linecolor="black",
                     zerolinecolor="#9aa0a8", automargin=True, title_standoff=14)
    fig.update_yaxes(title_text=ytitle, title_font=black, tickfont=black, gridcolor="#e3e5e8", linecolor="black",
                     zerolinecolor="#9aa0a8", automargin=True)
    if xticks and gait:
        fig.update_xaxes(tickmode="array", tickvals=[k.phase * 100 for k in gait.keys],
                         ticktext=[f"{k.phase * 100:.0f}%<br>{k.name}" for k in gait.keys])
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", font=black, legend=dict(font=black),
                      margin=dict(l=20, r=20, t=40, b=60))


def walk_load_plot(cfg: Config, gait, sw: dict) -> go.Figure:
    """|actuator load| through the stance phase, with each motor model's continuous and peak ratings."""
    fig = go.Figure()
    ph = sw["phase"] * 100
    color_of = {}
    for k, name in enumerate(sw["names"]):
        color_of[name] = _WALK_COLORS[k % len(_WALK_COLORS)]
        fig.add_trace(go.Scatter(x=ph, y=np.abs(sw["act"][:, k]), mode="lines", name=pretty(name),
                                 line=dict(color=color_of[name], width=2.5)))
    ymax = max(float(np.abs(sw["act"]).max()) * 1.25, 1.0)
    seen = set()
    for a in cfg.actuators:
        key = (a.model or a.name, a.peak, a.continuous)
        if key in seen:
            continue
        seen.add(key)
        col = color_of[a.name]
        for val, dash, lab in ((a.peak, "dash", "peak"), (a.continuous, "dot", "continuous")):
            if val <= ymax:
                fig.add_hline(y=val, line=dict(color=col, dash=dash, width=1.3),
                              annotation_text=f"{a.model or pretty(a.name)} {lab} {val:g}", annotation_position="top left",
                              annotation_font=dict(color=col, size=10))
    _walk_axes(fig, gait, "Actuator Load |N·m|", "Stance Phase", True)
    fig.update_yaxes(range=[0, ymax])
    fig.update_layout(height=420, legend=dict(orientation="h", y=-0.28))
    return fig
