"""Unit tests for least-squares fitters."""

import pytest

import numpy as np
from numpy import random

from expansion_fitter import fitters


def test_constant():
    """Set data to be constant, make sure we recover 0 slope, constant offset,
    with uncertainty for offset equal to the formula for the uncertainty in the mean."""
    n = 64
    value = 5
    err = 3
    x = np.arange(n) - (n - 1) / 2
    y = value * np.ones(n)
    y_err = err * np.ones(n)
    fitter = fitters.LineFitter(x, y, y_err)

    # Initial guess for parameters.
    pars0 = np.array([1., 1.])

    pars, cov = fitter.fit(pars0)
    m = pars[0]
    sig_m = np.sqrt(cov[0, 0])
    b = pars[1]
    sig_b = np.sqrt(cov[1, 1])
    assert b == pytest.approx(value, rel=0.001)
    assert m == pytest.approx(0, abs=0.0001)
    assert sig_b == pytest.approx(err / np.sqrt(n), rel=0.01)


def test_line():
    n = 256
    b_true = 5
    m_true = 0.5
    uncertainty = 0.001
    x = np.arange(n)
    y = m_true * x + b_true + random.randn(n) * uncertainty
    y_err = uncertainty * np.ones(n)
    fitter = fitters.LineFitter(x, y, y_err)

    # Initial guess for parameters.
    pars0 = np.array([1., 1.])

    pars, cov = fitter.fit(pars0)
    m = pars[0]
    sig_m = np.sqrt(cov[0, 0])
    b = pars[1]
    sig_b = np.sqrt(cov[1, 1])
    assert b == pytest.approx(b_true, 0.01)
    assert m == pytest.approx(m_true, 0.01)
    # The tolerence for this test based on the variance for a chi-squared distribution.
    assert fitter.chi_squared(pars) == pytest.approx(n - 2, abs=5 * np.sqrt(2 * n))


def test_line_slope_direct_measurement():
    """By putting all the data at the same x, data does not constrain slope. Information comes from
    the direct measurement."""
    n = 64
    value = 5
    err = 3
    x = np.zeros(n)
    y = value * np.ones(n)
    y_err = err * np.ones(n)
    slope_measurement = 3.
    slope_uncertainty = 1.
    fitter = fitters.LineFitterWithSlopeMeasurement(x, y, y_err, slope_measurement,
            slope_uncertainty)

    # Initial guess for parameters.
    pars0 = np.array([1., 1.])

    pars, cov = fitter.fit(pars0)
    m = pars[0]
    sig_m = np.sqrt(cov[0, 0])
    b = pars[1]
    sig_b = np.sqrt(cov[1, 1])
    assert b == pytest.approx(value, rel=0.001)
    assert m == pytest.approx(slope_measurement, abs=0.01)
    assert sig_b == pytest.approx(err / np.sqrt(n), rel=0.01)
    assert sig_m == pytest.approx(slope_uncertainty, rel=0.01)
