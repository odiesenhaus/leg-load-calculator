"""Shared helpers: build small, hand-checkable leg configs as plain dicts."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from leg import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

_AXES = [[1, 0, 0], [0, -1, 0], [0, 0, 1], [0, 1, 0], [0, -1, 0], [1, 0, 0]]
_NAMES = ["hip_roll", "hip_pitch", "hip_yaw", "knee_pitch", "ankle_pitch", "ankle_roll"]


def make_dict(thigh=0.3, shank=0.3, contact=(0.0, 0.0, -0.07), masses=None, coms=None,
              point_masses=None, section=None, beam_end=None, inertia=None, rotor=0.0, ratio=1.0,
              foot_origin=(0.0, 0.0, 0.0)):
    """6-joint leg with the template's axes; links are massless unless ``masses`` (per link) given."""
    masses = masses or [0.0] * 6
    coms = coms or [[0.0, 0.0, 0.0]] * 6
    origins = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, -thigh], [0, 0, -shank], list(foot_origin)]
    joints = []
    for i in range(6):
        link = {"name": f"link{i}", "mass": masses[i], "com": coms[i]}
        if point_masses and i in point_masses:
            link["point_masses"] = point_masses[i]
        if section and i in section:
            link["section"] = section[i]
        if inertia and i in inertia:
            link["inertia"] = inertia[i]
        joints.append({"name": _NAMES[i], "axis": _AXES[i], "origin": {"xyz": origins[i]},
                       "limits_deg": [-170, 170], "link": link})
    return {
        "robot": {"total_mass": 40.0},
        "joints": joints,
        "foot": {"contact_point": list(contact), "beam_end": list(beam_end or contact)},
        "actuators": [{"name": f"a{i}", "kind": "rotary", "continuous": 20.0, "peak": 60.0,
                       "max_speed": 20.0, "rotor_inertia": rotor} for i in range(6)],
        "transmissions": [{"joints": [_NAMES[i]], "actuators": [f"a{i}"], "type": "ratio", "ratio": ratio}
                          for i in range(6)],
        "presets": {},
    }


@pytest.fixture
def mk():
    def _mk(**kw):
        return load_config(copy.deepcopy(make_dict(**kw)))
    return _mk


@pytest.fixture
def template_cfg():
    return load_config(ROOT / "leg_config.yaml")


@pytest.fixture
def rng():
    return np.random.default_rng(1234)
