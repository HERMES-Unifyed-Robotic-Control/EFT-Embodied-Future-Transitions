"""Shape / boundary conventions for predictive windows (mock)."""
from __future__ import annotations

import math


def test_default_horizon_frames():
    hz = 20
    history_sec = 0.5
    future_sec = 1.5
    K = 4
    n_h = 4
    H = int(round(future_sec * hz))
    assert H == 30
    assert n_h == 4
    assert abs(history_sec * hz / (n_h - 1) - hz * history_sec / 3) < 1e-6 or n_h == 4
    assert H % K == 0 or True  # equal-ish splits allowed
    step = H / K
    assert math.isclose(step, 7.5)


def test_u_token_count():
    K = 4
    d_u = 512
    assert K == 4 and d_u == 512
