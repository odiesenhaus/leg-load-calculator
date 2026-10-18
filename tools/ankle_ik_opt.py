"""Gradient-based optimisation of the RSU ankle (lever length, lever neutral angle, rod pickup x) on a closed-form IK.

Same problem as tools/ankle_optimize.py, but instead of a grid search over Newton-solved poses it uses

* the closed-form RSU inverse kinematics (Cervettini et al. 2025, Eq. 7-8): with ``d = pickup - lever pivot``, the rod
  constraint ``|d - l u(th)|^2 = r^2`` becomes ``d_x cos th - d_z sin th = k``, ``k = (|d|^2 + l^2 - r^2) / (2 l)``, so
  ``th = -beta +- arccos(k / rho)`` with ``rho = hypot(d_x, d_z)``, ``beta = atan2(d_z, d_x)``; the branch is the one that
  puts the lever at its neutral angle with the foot flat, and the IK exists only while ``|k / rho| <= 1`` (Eq. 9);
* the analytic transmission Jacobian from implicit differentiation of the same constraint:
  ``d th / d q_j = (e . dP/dq_j) / (l r sin mu)``, with ``e`` the rod vector and ``mu`` the lever-rod angle;
* all poses evaluated at once with numpy (a whole ROM grid in a few milliseconds), and
* SciPy's SLSQP (sequential quadratic programming) with finite-difference gradients of those smooth functions.

Problem (epigraph form, so the objective is smooth):

    minimise    t
    over        lever l, neutral angle phi, pickup b_x, t
    subject to  t >= |tau_act| / peak  and  t >= |w_act| / no-load speed   for every task sample and motor
                sin(mu) >= sin(--min-angle)                                for every ROM grid pose and rod
                bounds on l, phi, b_x

``tau_act = T^-T tau_joint`` and ``w_act = T qd_joint`` use the walking stance sweep and static presets of leg_calc (joint
torques do not depend on the ankle geometry).  The ROM grid is the J5 / J6 joint limits; the optimum is re-checked on a
finer grid.  SLSQP finds a LOCAL optimum, so it is started from several points.

Needs SciPy (not in the leg_calc venv; the system Python has it):

    python tools/ankle_ik_opt.py
    python tools/ankle_ik_opt.py --lever 40 90 --min-angle 40
    python tools/ankle_ik_opt.py --check            # compare the closed form with leg/linkages.py, then exit

Run from the repository root (``leg_calc``).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from leg import Inputs, analyze, default_load, gait, load_config  # noqa: E402


# ---------------------------------------------------------------------------------------------------------------------
# closed-form RSU model, vectorised over poses
# ---------------------------------------------------------------------------------------------------------------------
def rot(axis, ang):
    """Rotation matrices about the unit ``axis`` by the angles ``ang`` (N,) -> (N, 3, 3) (Rodrigues)."""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    s, c = np.sin(ang)[:, None, None], np.cos(ang)[:, None, None]
    return np.eye(3) + s * K + (1 - c) * (K @ K)


class Ankle:
    """Fixed geometry from the leg_calc transmission ``params``; ``solve`` takes the design variables."""

    def __init__(self, params):
        self.pa = np.asarray(params["ankle_pitch_axis"], float)
        self.ra = np.asarray(params["ankle_roll_axis"], float)
        self.j56 = np.asarray(params["j5_to_j6"], float)
        self.A = np.array([[0.0, y, z] for y, z in zip(params["lever_y"], params["actuator_axis_z"])])
        self.pick = np.asarray(params["pickup_foot"], float)

    def solve(self, Q, lever, phi, bx):
        """Lever angles, signed sin(lever-rod angle) and T = d th / d(pitch, roll) at the poses ``Q`` (N, 2).

        Returns ``th`` (N, 2), ``sin_mu`` (N, 2; > 0 on the assembly branch of the neutral pose), ``T`` (N, 2, 2), the
        existence margin ``1 - (k / rho)^2`` (N, 2; < 0 = rod cannot reach) and the rod lengths (2,)."""
        Q = np.atleast_2d(Q)
        Q0 = np.vstack([np.zeros(2), Q])                         # row 0: neutral pose, fixes rod length and branch
        R5 = rot(self.pa, Q0[:, 0])
        R6 = R5 @ rot(self.ra, Q0[:, 1])
        p6 = R5 @ self.j56
        w6 = R5 @ self.ra                                        # roll axis after pitch
        n = len(Q0)
        th, smu, marg = np.zeros((n, 2)), np.zeros((n, 2)), np.zeros((n, 2))
        T = np.zeros((n, 2, 2))
        rods = np.zeros(2)
        for i in range(2):
            b = np.array([bx, self.pick[i, 1], self.pick[i, 2]])
            P = p6 + R6 @ b                                      # pickup in the shank frame (J5 at the origin)
            d = P - self.A[i]
            u0 = np.array([np.cos(phi), 0.0, -np.sin(phi)])
            r = np.linalg.norm(d[0] - lever * u0)
            rods[i] = r
            k = (np.sum(d * d, axis=1) + lever ** 2 - r ** 2) / (2 * lever)
            rho = np.hypot(d[:, 0], d[:, 2])
            c = k / rho
            marg[:, i] = 1.0 - c ** 2
            beta = np.arctan2(d[:, 2], d[:, 0])
            ac = np.arccos(np.clip(c, -1.0, 1.0))
            s = 1.0 if abs(np.angle(np.exp(1j * (-beta[0] + ac[0] - phi)))) < 1e-6 else -1.0   # branch of the neutral pose
            t = -beta + s * ac
            t = phi + np.angle(np.exp(1j * (t - phi)))           # unwrap around the neutral angle
            u = np.stack([np.cos(t), np.zeros_like(t), -np.sin(t)], axis=1)
            e = d - lever * u                                    # rod vector, lever tip -> pickup
            sm = (-np.sin(t) * e[:, 0] - np.cos(t) * e[:, 2]) / r
            sm = sm * np.sign(sm[0])
            dP = np.stack([np.cross(self.pa, P), np.cross(w6, P - p6)], axis=2)      # (N, 3, 2): dP/dpitch, dP/droll
            den = lever * r * sm * np.sign((-np.sin(t[0]) * e[0, 0] - np.cos(t[0]) * e[0, 2]))
            with np.errstate(divide="ignore", invalid="ignore"):
                T[:, i, :] = np.einsum("nk,nkj->nj", e, dP) / den[:, None]
            th[:, i], smu[:, i] = t, sm
        return th[1:], smu[1:], T[1:], marg[1:], rods


# ---------------------------------------------------------------------------------------------------------------------
# problem data from leg_calc
# ---------------------------------------------------------------------------------------------------------------------
def ankle_group(cfg):
    for spec in cfg.transmissions:
        if spec.function == "leg.linkages:ankle_parallel_jacobian":
            return spec
    raise SystemExit("no ankle_parallel_jacobian transmission in the config")


def task_demands(cfg, spec):
    """Ankle (q, tau, qd) of the walking stance sweep and the static presets, plus labels."""
    j = spec.joints
    q, tau, qd, lab = [], [], [], []
    for g in cfg.gaits.values():
        sw = gait.stance_sweep(cfg, g)
        for k in range(len(sw["phase"])):
            q.append(np.radians(sw["q"][k][j]))
            tau.append(sw["tau"][k][j])
            qd.append(np.radians(sw["qd_deg"][k][j]))
            lab.append(f"{g.name or 'gait'} {sw['phase'][k]:.2f}")
    for name, pr in cfg.presets.items():
        if pr.motion:
            continue
        r = analyze(cfg, Inputs(q=pr.q, load=default_load(cfg, pr.load_fraction)))
        q.append(pr.q[j])
        tau.append(r.static.tau[j])
        qd.append(np.zeros(2))
        lab.append(name)
    return np.array(q), np.array(tau), np.array(qd), lab


def rom_grid(cfg, spec, step_deg):
    (p_lo, p_hi), (r_lo, r_hi) = cfg.joints[spec.joints[0]].limits, cfg.joints[spec.joints[1]].limits
    st = np.radians(step_deg)
    P = np.unique(np.r_[np.arange(p_lo, p_hi, st), p_hi])
    R = np.unique(np.r_[np.arange(r_lo, r_hi, st), r_hi])
    return np.array([(a, b) for a in P for b in R])


def metrics(ank, x, data):
    """Everything about design ``x = (lever m, phi rad, b_x m)``."""
    Qt, tau, qd, lab, Qr, tau_pk, w_max = data
    th, sm, T, mg, rods = ank.solve(np.vstack([Qt, Qr]), *x)
    nt = len(Qt)
    Tt = T[:nt]
    with np.errstate(all="ignore"):
        ta = np.linalg.solve(np.transpose(Tt, (0, 2, 1)), tau[:, :, None])[:, :, 0]        # motor torques at the tasks
        wa = np.einsum("nij,nj->ni", Tt, qd)                                                 # motor speeds at the tasks
        Tr = T[nt:]
        Tit = np.linalg.inv(Tr).transpose(0, 2, 1)
        tau_cap = 1.0 / np.max(np.abs(Tit[:, :, 0]) / tau_pk, axis=1)                      # pure pitch torque available
    u_tau, u_w = np.abs(ta) / tau_pk, np.abs(wa) / w_max
    k = np.unravel_index(np.argmax(u_tau), u_tau.shape)
    return dict(th=th, sin_mu_rom=sm[nt:], marg_rom=mg[nt:], u_tau=u_tau, u_w=u_w, rods=rods,
                util=float(max(u_tau.max(), u_w.max())), case=lab[k[0]],
                rod_force=float(np.abs(ta[k]) / (x[0] * abs(sm[k[0], k[1]]))),
                tau_wc=float(np.nanmin(tau_cap)), th_rom=th[nt:])


# ---------------------------------------------------------------------------------------------------------------------
# optimiser
# ---------------------------------------------------------------------------------------------------------------------
SCALE = np.array([1e-1, np.radians(10.0), 1e-1])      # optimiser variables: lever / 100 mm, phi / 10 deg, b_x / 100 mm


class Problem:
    """SLSQP wrapper: variables ``z = (x / SCALE, t)``; constraint vector cached per z, gradients by central differences."""

    def __init__(self, ank, data, min_angle):
        self.ank, self.data = ank, data
        self.smin = np.sin(np.radians(min_angle))
        self._z = None
        self._g = None
        self.evals = 0

    def cons(self, z):
        if self._z is not None and np.array_equal(z, self._z):
            return self._g
        self.evals += 1
        x, t = z[:3] * SCALE, z[3]
        m = metrics(self.ank, x, self.data)
        g = np.concatenate([t - m["u_tau"].ravel(), t - m["u_w"].ravel(),           # epigraph: t >= every motor load
                            (m["sin_mu_rom"].ravel() - self.smin) * 5.0,              # lever-rod angle >= min (scaled)
                            m["marg_rom"].ravel()])                                   # rods reach every pose
        g = np.nan_to_num(g, nan=-1.0, posinf=-1.0, neginf=-1.0)
        self._z, self._g = z.copy(), g
        return g

    def cons_jac(self, z, h=1e-6):
        J = np.zeros((len(self.cons(z)), len(z)))
        for j in range(len(z)):
            dz = np.zeros_like(z)
            dz[j] = h
            J[:, j] = (self.cons(z + dz) - self.cons(z - dz)) / (2 * h)
        self.cons(z)
        return J


def optimise(ank, data, min_angle, bounds_x, starts):
    from scipy.optimize import minimize          # only the optimiser needs SciPy; the model runs without it
    prob = Problem(ank, data, min_angle)
    lo, hi = np.array(bounds_x).T
    bounds = [(a / s, b / s) for a, b, s in zip(lo, hi, SCALE)] + [(0.0, 10.0)]
    best = None
    for x0 in starts:
        m0 = metrics(ank, np.array(x0), data)
        z0 = np.r_[np.array(x0) / SCALE, max(m0["util"], 0.1) if np.isfinite(m0["util"]) else 5.0]
        res = minimize(lambda z: z[3], z0, jac=lambda z: np.r_[0, 0, 0, 1.0], method="SLSQP", bounds=bounds,
                       constraints=[dict(type="ineq", fun=prob.cons, jac=prob.cons_jac)],
                       options=dict(maxiter=200, ftol=1e-9))
        x = res.x[:3] * SCALE
        feas = prob.cons(res.x).min() >= -1e-6
        print(f"  start lever {x0[0] * 1000:5.1f} mm, phi {np.degrees(x0[1]):+5.1f} deg, b_x {x0[2] * 1000:5.1f} mm -> "
              f"lever {x[0] * 1000:5.1f}, phi {np.degrees(x[1]):+5.1f}, b_x {x[2] * 1000:5.1f}: "
              f"load {100 * res.x[3]:5.1f}%  {'feasible' if feas else 'INFEASIBLE'}, {res.nit} iterations")
        if feas and (best is None or res.x[3] < best[1]):
            best = (x, res.x[3])
    return best, prob.evals


# ---------------------------------------------------------------------------------------------------------------------
def check(cfg, spec, ank):
    """Closed form vs the Newton solver in leg/linkages.py at random poses and designs."""
    from leg.linkages import ankle_parallel_jacobian, solve_ankle_actuator_angles
    rng = np.random.default_rng(0)
    worst_th = worst_T = 0.0
    for _ in range(20):
        L, phi, bx = rng.uniform(0.045, 0.075), np.radians(rng.uniform(-10, 10)), rng.uniform(0.02, 0.06)
        p = dict(spec.params, lever_length=L, neutral_angle=phi,
                 pickup_foot=[[bx, b[1], b[2]] for b in spec.params["pickup_foot"]])
        Q = np.radians(rng.uniform([-40, -25], [25, 25], size=(5, 2)))
        th, _, T, mg, rods = ank.solve(Q, L, phi, bx)
        p["rod_length"] = list(rods)
        for k, q in enumerate(Q):
            worst_th = max(worst_th, np.max(np.abs(th[k] - solve_ankle_actuator_angles(q, **p))))
            worst_T = max(worst_T, np.max(np.abs(T[k] - ankle_parallel_jacobian(q, **p))))
    print(f"closed form vs leg/linkages.py over 100 random poses / 20 designs: "
          f"max |lever angle| error {worst_th:.1e} rad, max |T| error {worst_T:.1e}")


def report(name, x, m, fine_ok, smin_fine):
    th = np.degrees(m["th_rom"])
    print(f"{name}: lever {x[0] * 1000:.1f} mm, neutral angle {np.degrees(x[1]):+.1f} deg (+ = tip down), "
          f"b_x {x[2] * 1000:.1f} mm, rods {m['rods'][0] * 1000:.1f} / {m['rods'][1] * 1000:.1f} mm\n"
          f"    peak motor load {100 * m['util']:.1f}% ({m['case']}), speed {100 * m['u_w'].max():.1f}%, "
          f"worst-case pitch torque {m['tau_wc']:.1f} N*m, push-off rod force {m['rod_force']:.0f} N\n"
          f"    lever swing {th.min():.0f} .. {th.max():.0f} deg, min lever-rod angle on the fine grid "
          f"{np.degrees(np.arcsin(min(smin_fine, 1))):.1f} deg, all poses reachable: {fine_ok}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lever", nargs=2, type=float, default=[40, 75], metavar=("MIN", "MAX"), help="mm (40 75)")
    ap.add_argument("--neutral", nargs=2, type=float, default=[-30, 30], metavar=("MIN", "MAX"), help="deg (-30 30)")
    ap.add_argument("--bx", nargs=2, type=float, default=[5, 165], metavar=("MIN", "MAX"),
                    help="mm (5 165 = the foot plate toe)")
    ap.add_argument("--min-angle", type=float, default=45.0, help="minimum lever-rod angle over the ROM, deg")
    ap.add_argument("--grid-deg", type=float, default=2.5, help="ROM grid step for the constraints, deg")
    ap.add_argument("--check-deg", type=float, default=0.5, help="ROM grid step for the final check, deg")
    ap.add_argument("--check", action="store_true", help="only validate the closed form against leg/linkages.py")
    a = ap.parse_args()

    cfg = load_config(ROOT / "leg_config.yaml")
    spec = ankle_group(cfg)
    ank = Ankle(spec.params)
    if a.check:
        check(cfg, spec, ank)
        return
    acts = [cfg.actuators[i] for i in spec.actuators]
    tau_pk = np.array([x.peak for x in acts])
    w_max = np.array([x.max_speed for x in acts])
    Qt, tau, qd, lab = task_demands(cfg, spec)
    data = (Qt, tau, qd, lab, rom_grid(cfg, spec, a.grid_deg), tau_pk, w_max)
    fine = (Qt, tau, qd, lab, rom_grid(cfg, spec, a.check_deg), tau_pk, w_max)

    def final(name, x):
        m = metrics(ank, x, fine)
        report(name, x, m, bool(m["marg_rom"].min() >= 0), float(m["sin_mu_rom"].min()))

    p = spec.params
    final("current design", np.array([p["lever_length"], p.get("neutral_angle", 0.0), p["pickup_foot"][0][0]]))

    bounds_x = [np.array(a.lever) / 1000, np.radians(a.neutral), np.array(a.bx) / 1000]
    starts = [np.array([L, np.radians(ph), 0.7 * L]) for L in np.linspace(*bounds_x[0], 3) for ph in (-10.0, 0.0, 10.0)]
    print(f"\nSLSQP, {len(data[4])} ROM poses x 2 rods as constraints, lever-rod angle >= {a.min_angle:g} deg:")
    t0 = time.perf_counter()
    best, evals = optimise(ank, data, a.min_angle, bounds_x, starts)
    dt = time.perf_counter() - t0
    print(f"  {evals} model evaluations in {dt:.1f} s ({1000 * dt / max(evals, 1):.1f} ms each)\n")
    if best is None:
        print("no feasible design found")
        return
    final("optimum", best[0])


if __name__ == "__main__":
    main()
