"""Numerical fingerprint of the model, to prove a refactor or documentation pass changed no physics.

It evaluates every preset in ``leg_config.yaml`` (joint torques, actuator loads, peak link stresses, bearing tilt
moments) and a stance sweep of every gait, and stores ~1000 numbers.

    python tools/fingerprint.py save   before.json     # on the known-good version
    python tools/fingerprint.py compare before.json    # after the change: prints IDENTICAL or the differences

Differences up to 1e-9 relative are ignored.  Run from the repository root (``leg_calc``).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from leg import Inputs, analyze, default_load, gait, load_config  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "leg_config.yaml"


def fingerprint() -> dict:
    cfg = load_config(CONFIG)
    out: dict = {"joint_names": cfg.joint_names, "mass": cfg.robot_total_mass}
    for name, p in cfg.presets.items():
        r = analyze(cfg, Inputs(q=p.q, load=default_load(cfg, p.load_fraction)))
        out["preset:" + name] = {
            "tau": r.static.tau.tolist(), "act": [a.value for a in r.actuators],
            "sigma": [None if lr.peak_stress is None else lr.peak_stress.sigma for lr in r.links],
            "flange": [f.tilt_moment for f in r.flanges]}
    for gname, g in cfg.gaits.items():
        sw = gait.stance_sweep(cfg, g, n=21)
        out["gait:" + gname] = {"tau": sw["tau"].tolist(), "act": sw["act"].tolist(), "speed": sw["speed"].tolist(),
                                "min_sf": sw["min_sf"].tolist()}
    return out


def _flat(d, prefix=""):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from _flat(v, f"{prefix}{k}/")
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from _flat(v, f"{prefix}{i}/")
    else:
        yield prefix, d


def main(mode: str, path: str) -> None:
    fp = fingerprint()
    if mode == "save":
        json.dump(fp, open(path, "w"))
        print("saved", sum(1 for _ in _flat(fp)), "numbers to", path)
        return
    old, new = dict(_flat(json.load(open(path)))), dict(_flat(fp))

    def same(a, b):
        return (isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(a - b) <= 1e-9 * max(1.0, abs(a))) or a == b

    bad = [(k, old.get(k), new.get(k)) for k in set(old) | set(new) if not same(old.get(k), new.get(k))]
    print("compared", len(old), "numbers:", "IDENTICAL" if not bad else f"{len(bad)} DIFFERENCES, first: {bad[:5]}")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("save", "compare"):
        sys.exit(__doc__)
    main(*sys.argv[1:])
