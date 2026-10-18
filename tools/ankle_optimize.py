"""Choose the ankle lever length, lever neutral angle and rod pickup x position that keep the ankle motors farthest from
their limits.

Objective: minimise the peak motor utilisation over the task set (walking stance sweep + static presets),
``max(|tau_act| / peak torque, |w_act| / no-load speed)``, for both ankle motors.

Constraints, checked on a grid over the full ankle range of motion (J5 / J6 joint limits):

* the rods close the loop at every pose (Cervettini et al. 2025, Eq. 9) without passing a dead-centre;
* the lever-rod angle stays at or above ``--min-angle`` degrees everywhere (default 45).

Design variables:

* lever length ``l``;
* lever neutral angle ``phi``: the lever angle at the neutral ankle pose (foot flat), right-hand about +y as in
  leg/linkages.py, so ``phi > 0`` tips the lever tip DOWN (toward the foot) and ``phi < 0`` tips it UP;
* pickup ``b_x`` (both pickups, ahead of the ankle).

Everything else is as configured (pickup y / z, lever pivots, motors).  The rod length is re-derived for every design so
the loop closes with the lever at ``phi`` and the foot flat.  For each ``(l, phi)`` a coarse scan over ``b_x`` finds the
best passing value; if the next larger ``b_x`` fails the constraints the boundary between them is refined by bisection
(the motor load keeps falling as ``b_x`` grows, so the optimum sits on the constraint boundary).

    python tools/ankle_optimize.py                                     # lever 40..75 mm, phi -40..40 deg
    python tools/ankle_optimize.py --lever 30 100 10 --neutral -30 30 2.5 --min-angle 40
    python tools/ankle_optimize.py --neutral 0 0 1                     # lever horizontal at neutral (old behaviour)

Run from the repository root (``leg_calc``).
"""
import argparse
import copy
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ankle_bx_sweep import (ROOT, ankle_group, capability, load_config, params_for,  # noqa: E402
                            pose_metrics, task_demands)

_W = {}                                   # per-worker state: tasks, ratings, grids, base params, constraint


def evaluate(base, lever, bx, pitch, roll, tasks, tau_pk, w_max, neutral=0.0):
    """Metrics of one design.  ``ok`` is False when some ROM pose is unreachable or past a dead-centre."""
    b = copy.deepcopy(base)
    b["lever_length"] = lever
    b["neutral_angle"] = neutral
    p = params_for(b, bx, keep_rod=False)
    try:
        _, s0, _ = pose_metrics(np.zeros(2), p)
    except (ValueError, np.linalg.LinAlgError):
        return dict(ok=False)
    mu, tau_wc = 1.0, np.inf
    th_all = []
    for qp in pitch:
        for qr in roll:
            try:
                th, s, T = pose_metrics(np.array([qp, qr]), p)
            except (ValueError, np.linalg.LinAlgError):
                return dict(ok=False)
            if np.any(np.sign(s) != np.sign(s0)):
                return dict(ok=False)
            th_all.append(th)
            mu = min(mu, float(np.min(np.abs(s))))
            tau_wc = min(tau_wc, capability(T, tau_pk, w_max)[0][0])
    u_tau = u_w = 0.0
    case = ""
    push = None
    for q, tau, qd, label in tasks:
        _, s, T = pose_metrics(q, p)
        ta = np.linalg.solve(T.T, tau)
        ut = float(np.max(np.abs(ta) / tau_pk))
        if ut > u_tau:
            u_tau, case = ut, label
            # rod force at this pose, the larger of the two rods: motor torque / (lever * sin mu)
            push = float(np.max(np.abs(ta) / (lever * np.abs(s))))
        if qd is not None:
            u_w = max(u_w, float(np.max(np.abs(T @ qd) / w_max)))
    th_all = np.array(th_all)
    return dict(ok=True, mu=float(np.degrees(np.arcsin(min(mu, 1.0)))), tau_wc=float(tau_wc), u_tau=100 * u_tau,
                u_w=100 * u_w, util=100 * max(u_tau, u_w), case=case, rod_force=push, rod=p["rod_length"],
                travel=float(np.degrees(np.max(np.ptp(th_all, axis=0)))),
                th_lo=float(np.degrees(np.min(th_all))), th_hi=float(np.degrees(np.max(th_all))))


def _init(base, tasks, tau_pk, w_max, coarse, fine, min_angle):
    _W.update(base=base, tasks=tasks, tau_pk=tau_pk, w_max=w_max, coarse=coarse, fine=fine, min_angle=min_angle)


def _good(r):
    return r["ok"] and r["mu"] >= _W["min_angle"]


def best_design(lever_phi, bx_step=0.005, tol=0.0002):
    """Best ``b_x`` for one ``(lever, phi)``: returns ``(lever, phi, bx, metrics)`` or ``(lever, phi, None, None)``."""
    lever, phi = lever_phi
    ev = lambda bx, g: evaluate(_W["base"], lever, bx, *g, _W["tasks"], _W["tau_pk"], _W["w_max"], phi)  # noqa: E731
    bxs = np.arange(0.005, lever + 0.06 + 1e-9, bx_step)
    res = [ev(bx, _W["coarse"]) for bx in bxs]
    ok = [i for i, r in enumerate(res) if _good(r)]
    if not ok:
        return lever, phi, None, None
    i = min(ok, key=lambda k: res[k]["util"])
    bx = bxs[i]
    if i + 1 < len(bxs) and not _good(res[i + 1]):          # optimum on the constraint boundary: refine it
        lo, hi = bxs[i], bxs[i + 1]
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            r = ev(mid, _W["coarse"])
            if _good(r) and r["util"] <= res[i]["util"] + 1e-9:
                lo = mid
            else:
                hi = mid
        bx = lo
    r = ev(bx, _W["fine"])
    for _ in range(20):                                       # the fine grid can be slightly stricter
        if _good(r) or bx <= 0.005:
            break
        bx -= 0.0005
        r = ev(bx, _W["fine"])
    return (lever, phi, bx, r) if _good(r) else (lever, phi, None, None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lever", nargs=3, type=float, default=[40, 75, 5], metavar=("FROM", "TO", "STEP"),
                    help="lever lengths to compare, mm (default 40 75 5)")
    ap.add_argument("--neutral", nargs=3, type=float, default=[-40, 40, 5], metavar=("FROM", "TO", "STEP"),
                    help="lever neutral angles to compare, deg, + = tip down (default -40 40 5)")
    ap.add_argument("--min-angle", type=float, default=45.0, help="minimum lever-rod angle over the ROM, deg")
    ap.add_argument("--search-deg", type=float, default=5.0, help="ROM grid step during the search, deg")
    ap.add_argument("--check-deg", type=float, default=1.25, help="ROM grid step for the final check, deg")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()

    cfg = load_config(ROOT / "leg_config.yaml")
    spec = ankle_group(cfg)
    acts = [cfg.actuators[i] for i in spec.actuators]
    tau_pk = np.array([x.peak for x in acts])
    w_max = np.array([x.max_speed for x in acts])
    (p_lo, p_hi), (r_lo, r_hi) = cfg.joints[spec.joints[0]].limits, cfg.joints[spec.joints[1]].limits

    def grid(step):
        st = np.radians(step)
        return (np.unique(np.r_[np.arange(p_lo, p_hi, st), p_hi]), np.unique(np.r_[np.arange(r_lo, r_hi, st), r_hi]))

    tasks = task_demands(cfg, spec)
    base = dict(spec.params)
    now = evaluate(base, base["lever_length"], base["pickup_foot"][0][0], *grid(a.check_deg), tasks, tau_pk, w_max,
                   base.get("neutral_angle", 0.0))

    levers = np.arange(a.lever[0], a.lever[1] + 1e-9, a.lever[2]) / 1000
    phis = np.radians(np.arange(a.neutral[0], a.neutral[1] + 1e-9, a.neutral[2]))
    jobs = [(L, ph) for L in levers for ph in phis]
    with Pool(a.jobs, initializer=_init,
              initargs=(base, tasks, tau_pk, w_max, grid(a.search_deg), grid(a.check_deg), a.min_angle)) as pool:
        out = pool.map(best_design, jobs, chunksize=1)

    print(f"constraint: lever-rod angle >= {a.min_angle:g} deg over pitch {np.degrees(p_lo):g}..{np.degrees(p_hi):g}, "
          f"roll {np.degrees(r_lo):g}..{np.degrees(r_hi):g} deg;  current design {now['util']:.1f}% peak motor load\n")
    print("peak motor load % at the best b_x  (rows: lever mm, columns: neutral angle deg, + = tip down)")
    print("lever " + "".join(f"{np.degrees(ph):7.1f}" for ph in phis))
    table = {(L, ph): r for L, ph, bx, r in out}
    for L in levers:
        print(f"{L * 1000:5.1f} " + "".join(f"{table[(L, ph)]['util']:7.1f}" if table[(L, ph)] else "      -"
                                           for ph in phis))

    hdr = (f"\n{'lever':>6} {'phi':>6} {'b_x':>6} {'rods':>13} {'motor %':>7} {'speed %':>7} {'min mu':>6} "
           f"{'worst tau_p':>11} {'lever range deg':>16} {'rod force':>9}")
    fmt = lambda L, ph, bx, r: (f"{L * 1000:6.1f} {np.degrees(ph):6.1f} {bx * 1000:6.1f} "  # noqa: E731
                                f"{r['rod'][0] * 1000:6.1f}/{r['rod'][1] * 1000:5.1f} {r['util']:7.1f} {r['u_w']:7.1f} "
                                f"{r['mu']:6.1f} {r['tau_wc']:11.1f} {r['th_lo']:7.0f} ..{r['th_hi']:5.0f} "
                                f"{r['rod_force']:8.0f}N")
    print(hdr + "\n" + fmt(base["lever_length"], base.get("neutral_angle", 0.0), base["pickup_foot"][0][0], now)
          + "   <- current design")
    best_per_lever = []
    for L in levers:
        cands = [(ph, bx, r) for LL, ph, bx, r in out if LL == L and r]
        if cands:
            ph, bx, r = min(cands, key=lambda c: c[2]["util"])
            best_per_lever.append((L, ph, bx, r))
            print(fmt(L, ph, bx, r))
    L, ph, bx, r = min(best_per_lever, key=lambda t: t[3]["util"])
    print(f"\noptimum: lever {L * 1000:.1f} mm, neutral angle {np.degrees(ph):+.1f} deg, b_x {bx * 1000:.1f} mm -> "
          f"peak motor load {r['util']:.1f}% ({r['case']}), was {now['util']:.1f}%")


if __name__ == "__main__":
    main()
