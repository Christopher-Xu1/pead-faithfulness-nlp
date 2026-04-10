import numpy as np

from src.data.compute_pead import compute_pead


def test_compute_pead_basic():
    stock = np.array([0.0, 0.02, 0.01, -0.01, 0.03])
    market = np.array([0.0, 0.01, 0.00, -0.02, 0.01])

    abnormal, car, label = compute_pead(stock, market, event_idx=0, horizon=3)

    np.testing.assert_allclose(abnormal, np.array([0.01, 0.01, 0.01]))
    assert abs(car - 0.03) < 1e-9
    assert label == 1


def test_compute_pead_handles_tail_horizon():
    stock = np.array([0.0, 0.01])
    market = np.array([0.0, 0.0])

    abnormal, car, label = compute_pead(stock, market, event_idx=0, horizon=10)

    np.testing.assert_allclose(abnormal, np.array([0.01]))
    assert abs(car - 0.01) < 1e-9
    assert label == 1
