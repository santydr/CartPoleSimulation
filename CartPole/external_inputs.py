"""
Optional external inputs for the AIS 2026 revision experiments.

Both classes are inactive by default: a CartPole instance only uses them if
`CartPole.disturbance` or `CartPole.sensor_model` is set explicitly (e.g. by
run_ais_batch.py). With both left as None the simulator behaves exactly as before.

- PulseTrain: force pulses applied to the cart (in the model's newtons).
- RobotLikeSensor: measurement model inspired by the physical robot:
  angle and position quantized with the resolution of the ADC and the encoder,
  optional white noise on the angle before quantization, and velocities
  obtained by finite differences of the quantized measurements over a fixed
  window (20 ms in the robot firmware: derivative over 2 control periods of 10 ms).
"""
from collections import deque

import numpy as np

from CartPole.state_utilities import (ANGLE_COS_IDX, ANGLE_IDX, ANGLE_SIN_IDX,
                                      ANGLED_IDX, POSITION_IDX, POSITIOND_IDX)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class PulseTrain:
    """Rectangular force pulses on the cart: force(t) in N (model units)."""

    def __init__(self, start_times, signs, amplitude, duration=0.05):
        self.start_times = np.asarray(start_times, dtype=float)
        self.signs = np.asarray(signs, dtype=float)
        self.amplitude = float(amplitude)
        self.duration = float(duration)

    def force(self, t):
        active = (t >= self.start_times) & (t < self.start_times + self.duration)
        if not np.any(active):
            return 0.0
        return float(self.amplitude * np.sum(self.signs[active]))

    __call__ = force


class RobotLikeSensor:
    """Quantized, optionally noisy measurements with finite-difference velocities."""

    def __init__(self, angle_lsb=2 * np.pi / 4302, position_lsb=0.396 / 4649,
                 angle_sigma=0.0, derivative_window=0.02, seed=0):
        self.angle_lsb = float(angle_lsb)
        self.position_lsb = float(position_lsb)
        self.angle_sigma = float(angle_sigma)
        self.derivative_window = float(derivative_window)
        self.rng = np.random.default_rng(seed)
        self.history = deque()

    def measure(self, s, t):
        out = np.array(s, copy=True)
        angle = out[ANGLE_IDX]
        if self.angle_sigma > 0.0:
            angle = angle + self.angle_sigma * self.rng.standard_normal()
        q_angle = _wrap(np.round(angle / self.angle_lsb) * self.angle_lsb)
        q_pos = np.round(out[POSITION_IDX] / self.position_lsb) * self.position_lsb

        self.history.append((t, q_angle, q_pos))
        # Keep the oldest sample that is at least derivative_window old
        while len(self.history) > 2 and self.history[1][0] <= t - self.derivative_window + 1e-9:
            self.history.popleft()
        t_old, a_old, x_old = self.history[0]
        dt = t - t_old
        if dt > 0:
            angle_d = _wrap(q_angle - a_old) / dt
            pos_d = (q_pos - x_old) / dt
        else:
            angle_d, pos_d = 0.0, 0.0

        out[ANGLE_IDX] = q_angle
        out[ANGLE_COS_IDX] = np.cos(q_angle)
        out[ANGLE_SIN_IDX] = np.sin(q_angle)
        out[ANGLED_IDX] = angle_d
        out[POSITION_IDX] = q_pos
        out[POSITIOND_IDX] = pos_d
        return out
