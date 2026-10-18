"""Per-joint share of the actuator capacity (actuators.joint_shares): coupled ankle joints must not inherit each other's load."""
import numpy as np
import pytest

from leg import Inputs, Load, analyze

W = 40.0 * 9.81


def _stand(cfg, cop=(0.0, 0.0), fz=W):
    return analyze(cfg, Inputs(q=np.zeros(6), load=Load(np.array([0.0, 0.0, fz])), cop_offset=cop))


def test_one_to_one_joints_equal_their_actuator(template_cfg):
    r = _stand(template_cfg, fz=0.5 * W)
    for j in range(4):                                         # hip x3 and knee: one motor each, ratio 1
        assert r.shares[j].util_cont == pytest.approx(r.actuators[j].util_cont)
        assert r.shares[j].status == r.actuators[j].status


def test_zero_roll_torque_shows_zero_share_even_when_pitch_overloads_the_motors(template_cfg):
    r = _stand(template_cfg, cop=(0.12, 0.0))                  # 120 mm forward CoP, single support: pitch torque only
    assert abs(r.static.tau[5]) < 1e-6                         # no roll torque
    assert r.shares[5].util_cont == pytest.approx(0.0, abs=1e-9) and r.shares[5].status == "ok"
    assert r.shares[4].util_cont > 1.0                         # pitch alone is past the continuous rating
    assert max(a.util_cont for a in r.actuators[4:]) == pytest.approx(r.shares[4].util_cont, rel=1e-6)


def test_roll_only_load_does_not_light_up_the_pitch_row(template_cfg):
    r = _stand(template_cfg, cop=(0.0, 0.02))                  # 20 mm lateral CoP shift: roll torque only (plus foot weight)
    assert r.shares[5].util_cont > 1.0                         # the weak roll axis: ~139 % of continuous
    assert r.shares[4].util_cont < 0.1
    assert max(a.util_cont for a in r.actuators[4:]) == pytest.approx(r.shares[5].util_cont, rel=0.05)


def test_combined_motor_load_lies_between_the_joint_shares(template_cfg):
    """f = T^-T tau is linear, so the combined load is the vector sum of the joints' contributions."""
    r = _stand(template_cfg, cop=(0.03, 0.02))
    s4, s5 = r.shares[4].util_cont, r.shares[5].util_cont
    comb = max(a.util_cont for a in r.actuators[4:])
    assert abs(s4 - s5) - 1e-9 <= comb <= s4 + s5 + 1e-9
    assert comb > max(s4, s5) * 0.99                            # here the two contributions add on one motor
