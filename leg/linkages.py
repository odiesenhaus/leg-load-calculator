"""Linkage-driven transmissions (pose-dependent ``T(q)``).

Select one in leg_config.yaml, for example::

    transmissions:
      - joints: [ankle_flexion, ankle_inversion]
        actuators: [ankle_upper_act, ankle_lower_act]
        type: function
        function: "leg.linkages:ankle_parallel_jacobian"
        params: {...}                      # passed as keyword arguments

Each function receives the group's joint angles ``q`` (rad, ordered as ``joints:``) and returns ``T(q)`` with
``qdot_act = T(q) @ qdot_joint`` (see leg/actuators.py).  For a table-driven ratio (a simple linkage knee) no code is
needed: use ``type: table`` in the YAML.

Parallel ankle model (``ankle_parallel_jacobian``)
--------------------------------------------------
Two rotary actuators sit on the shank with their output axes parallel to y.  Each drives a lever of length ``l`` whose tip
is joined by a fixed-length rod to a pickup point on the foot.  All geometry is given in the shank frame at J5 (ankle
pitch axis), before the ankle rotates:

* lever i pivots about the y-parallel axis through ``(0, lever_y[i], actuator_axis_z[i])`` and its tip is
  ``A_i + l (cos th, 0, -sin th)``, where ``th`` is the right-hand angle about +y (``th = 0``: lever pointing forward,
  perpendicular to the shin);
* the foot (frame J6, after pitch then roll) carries pickup i at ``pickup_foot[i]``;
* the rod constraint ``|pickup_world(q) - tip(th)| = rod_length[i]`` fixes ``th`` for every ankle pose; it is solved
  exactly by Newton's method starting from ``neutral_angle``, and ``T`` is the central-difference Jacobian
  ``d th / d (ankle pitch, ankle roll)``.

No limits on ``th`` are enforced (the linkage is assumed free over the ankle range of motion); a pose with no
solution, or a lever/rod dead-centre, raises ``ValueError`` which the caller reports as a transmission error.
"""
import numpy as np

from .kinematics import rot_axis


def solve_ankle_actuator_angles(q, *, ankle_pitch_axis, ankle_roll_axis, j5_to_j6, actuator_axis_z,
                                lever_length, rod_length, lever_y, pickup_foot, neutral_angle=0.0):
    """Lever angles ``th`` (one per actuator) for ankle pitch / roll ``q = [q_pitch, q_roll]`` [rad].

    Geometry parameters are described in the module docstring; ``ankle_pitch_axis`` / ``ankle_roll_axis`` are the joint
    axes in their own frames, ``j5_to_j6`` the offset from J5 to J6 in the shank frame, ``pickup_foot`` the two pickup
    points in the J6 frame.  Raises ``ValueError`` when the rods cannot reach the pose.
    """
    q = np.asarray(q, float)
    R5 = rot_axis(np.asarray(ankle_pitch_axis, float), q[0])
    R6 = R5 @ rot_axis(np.asarray(ankle_roll_axis, float), q[1])
    p6 = R5 @ np.asarray(j5_to_j6, float)
    th = np.zeros(len(rod_length))
    for i in range(len(rod_length)):
        P = p6 + R6 @ np.asarray(pickup_foot[i], float)           # pickup position, shank frame
        A = np.array([0.0, lever_y[i], actuator_axis_z[i]])        # lever pivot
        t = neutral_angle
        for _ in range(60):                                        # Newton on f(t) = |P - tip(t)|^2 - L^2
            tip = A + lever_length * np.array([np.cos(t), 0.0, -np.sin(t)])
            d = P - tip
            f = d @ d - rod_length[i] ** 2
            df = -2.0 * d @ (lever_length * np.array([-np.sin(t), 0.0, -np.cos(t)]))
            if abs(f) < 1e-14:
                break
            if abs(df) < 1e-9:
                raise ValueError(f"ankle rod {i} is at a dead-centre (lever and rod aligned)")
            t -= f / df
        else:
            raise ValueError(f"ankle rod {i} cannot reach this ankle pose (no solution)")
        th[i] = t
    return th


def ankle_parallel_jacobian(q, **params):
    """2x2 ``T(q) = d(lever angles) / d(ankle pitch, ankle roll)`` of the lever + rod parallel ankle.

    Central differences of ``solve_ankle_actuator_angles`` (pose dependent, exact to the rod-length constraint).
    Rows: the two actuators in the order of ``actuators:``; columns: ankle pitch, ankle roll."""
    q = np.asarray(q, float)
    eps = 1e-6
    T = np.zeros((2, 2))
    for j in range(2):
        dq = np.zeros(2)
        dq[j] = eps
        T[:, j] = (solve_ankle_actuator_angles(q + dq, **params)
                   - solve_ankle_actuator_angles(q - dq, **params)) / (2 * eps)
    return T


def knee_linkage_ratio(q: np.ndarray, **params) -> np.ndarray:
    """TODO(knee linkage): 1x1 matrix ``[[d actuator / d knee angle]]`` for a four-bar / slider-crank knee as a function of
    the knee angle ``q[0]`` [rad].  The current knee is a direct, in-line X8-150 (ratio 1), so nothing uses this."""
    raise NotImplementedError("fill in knee_linkage_ratio() in leg/linkages.py with the real geometry")
