"""Walking / running stance as a sequence of quasi-static snapshots (no UI imports).

A ``Gait`` (``gaits:`` in leg_config.yaml) holds keyframes over ONE STANCE PHASE (phase 0 = foot strike .. 1 = toe-off,
as a fraction of the stance duration): joint angles, the ground reaction in body weights [fore-aft, lateral, vertical]
and the centre-of-pressure position along the foot.  ``sample`` interpolates linearly between keyframes;
``stance_sweep`` evaluates the whole model at many phases so the peak load, RMS load (heating), speed and power of every
actuator over a step can be read off, and ``summarize`` condenses that per actuator.

What this is and is not:

* Each phase is a static equilibrium with the given ground force.  Leg-segment inertia and the swing phase are not
  modelled, so swing-phase torques are missing and the whole-cycle RMS (which assumes an unloaded swing) is optimistic.
* Joint speeds are the keyframe angle rate: ``d q / d phase / stance_time``, from a lightly smoothed finely sampled
  curve.  ``stance_time`` therefore changes speeds and power but NEVER the torques; keyframe TIMES scale with it
  (phase is a fraction of stance), so a time-based reference such as "1 body weight within 150 ms" only holds at the
  stance time it was placed for.
* The bundled keyframes are typical adult human shapes scaled to the robot's weight, not data from this robot; they
  describe a heel-to-toe gait (the foot pitches), not a flat-foot gait.
"""
from __future__ import annotations

import numpy as np

from .analysis import Inputs, analyze
from .config import Config, Gait
from .statics import Load


def sample(gait: Gait, phase: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Linear interpolation of the keyframes at ``phase`` (clamped to 0..1).

    Returns ``(q [rad], force [body weights: fore-aft, lateral, vertical], cop_x [m ahead of the ankle])``.
    """
    ph = np.array([k.phase for k in gait.keys])
    phase = float(min(max(phase, 0.0), 1.0))
    q = np.array([np.interp(phase, ph, [k.q[i] for k in gait.keys]) for i in range(len(gait.keys[0].q))])
    f = np.array([np.interp(phase, ph, [k.force_bw[i] for k in gait.keys]) for i in range(3)])
    return q, f, float(np.interp(phase, ph, [k.cop_x for k in gait.keys]))


def nearest_name(gait: Gait, phase: float) -> str:
    """Name of the keyframe closest to ``phase`` (its percentage if it has no name)."""
    k = min(gait.keys, key=lambda k: abs(k.phase - phase))
    return k.name or f"{k.phase * 100:.0f}%"


def _smooth_rate(gait: Gait, phases: np.ndarray, stance_time: float) -> np.ndarray:
    """Joint rates [rad/s] at ``phases``: finely sampled keyframe angles, moving-average smoothed, differentiated.

    The curve is padded by odd (point) reflection so the end values and trends are preserved; without that the ends
    of the stance would be flattened by the averaging and the angle change would not integrate back correctly."""
    fine = np.linspace(0.0, 1.0, 401)
    q = np.array([sample(gait, p)[0] for p in fine])
    k = 15
    h = k // 2
    pad = np.vstack([2 * q[0] - q[1:h + 1][::-1], q, 2 * q[-1] - q[-h - 1:-1][::-1]])
    qs = np.array([np.convolve(pad[:, j], np.ones(k) / k, mode="valid") for j in range(q.shape[1])]).T
    qd = np.gradient(qs, fine, axis=0) / stance_time
    return np.array([np.interp(phases, fine, qd[:, j]) for j in range(q.shape[1])]).T


def stance_sweep(cfg: Config, g: Gait, n: int = 41, payload: float = 0.0, stance_time: float | None = None) -> dict:
    """Evaluate the leg at ``n`` evenly spaced phases of stance.  Returns arrays indexed by phase:

    phase, q (deg), force_bw, cop_x, tau (joint torques), act (actuator loads), speed (actuator speeds), util_cont,
    min_sf (smallest safety factor over all links), warnings (count), qd_deg (joint rates), plus stance_time, weight
    (one body weight incl. payload, N) and names (actuator names).  The sweep is uniform in phase, so time averages
    over the stance are plain means.
    """
    T = float(stance_time or g.stance_time)
    ph = np.linspace(0.0, 1.0, n)
    qd = _smooth_rate(g, ph, T)
    weight = (cfg.robot_total_mass + payload) * cfg.gravity
    out = {k: [] for k in ("q", "force_bw", "cop_x", "tau", "act", "speed", "util_cont", "min_sf", "warnings")}
    for p, qdi in zip(ph, qd):
        q, f_bw, cop = sample(g, p)
        r = analyze(cfg, Inputs(q=q, load=Load(f_bw * weight), cop_offset=(cop, 0.0), qd=qdi))
        if r.actuators is None:
            raise RuntimeError("; ".join(r.errors) or "actuator model unavailable")
        out["q"].append(np.degrees(q))
        out["force_bw"].append(f_bw)
        out["cop_x"].append(cop)
        out["tau"].append(r.static.tau)
        out["act"].append([a.value for a in r.actuators])
        out["speed"].append([a.speed for a in r.actuators])
        out["util_cont"].append([a.util_cont for a in r.actuators])
        sfs = [lr.min_sf for lr in r.links if lr.min_sf is not None]
        out["min_sf"].append(min(sfs) if sfs else np.inf)
        out["warnings"].append(len(r.warnings))
    res = {k: np.array(v) for k, v in out.items()}
    res.update(phase=ph, stance_time=T, weight=weight, names=[a.name for a in cfg.actuators],
               qd_deg=np.degrees(qd))
    return res


def summarize(cfg: Config, g: Gait, sw: dict) -> list[dict]:
    """Per-actuator peak / RMS load, peak speed, peak mechanical power and torque-speed-envelope check.

    * ``rms_stance`` is the root-mean-square load over the stance (the heating measure; compare with the continuous
      rating).  ``rms_cycle_pct`` scales it by sqrt(stance_fraction), assuming an unloaded swing (optimistic).
    * ``over_envelope`` is True if any point has |torque| above a simple envelope: the peak torque falling linearly to
      zero at the no-load speed.  The continuous limit used in the plots is flat to the rated speed, then linear to the
      no-load speed.  Both are approximations from datasheet numbers, not the real motor curve.
    """
    frac = g.stance_fraction
    rows = []
    for k, a in enumerate(cfg.actuators):
        v, w = sw["act"][:, k], sw["speed"][:, k]
        rms = float(np.sqrt(np.mean(v**2)))
        rated_w = a.rated_speed or 0.8 * a.max_speed
        rows.append({
            "name": a.name, "model": a.model, "unit": "N*m" if a.kind == "rotary" else "N",
            "peak": float(np.max(np.abs(v))), "peak_pct": float(np.max(np.abs(v)) / a.peak * 100),
            "rms_stance": rms, "rms_stance_pct": rms / a.continuous * 100,
            "rms_cycle_pct": rms * np.sqrt(frac) / a.continuous * 100,     # swing assumed unloaded
            "peak_speed": float(np.max(np.abs(w))), "peak_speed_pct": float(np.max(np.abs(w)) / a.max_speed * 100),
            "peak_power": float(np.max(np.abs(v * w))), "continuous": a.continuous, "peak_rating": a.peak,
            "max_speed": a.max_speed, "rated_speed": rated_w,
            "over_envelope": bool(np.any(np.abs(v) > a.peak * np.clip(1 - np.abs(w) / a.max_speed, 0, None) + 1e-9)),
        })
    return rows
