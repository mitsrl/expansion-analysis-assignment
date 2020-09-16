"""Least squares fitters."""

import numpy as np
from scipy import optimize, linalg

from expansion_fitter import friedmann_integrators


class BaseFitter:
    """This is an abstract base class. ABSs are classes that are useless on thier own, but provide
    code that can be used by subclasses.

    """

    def data(self):
        raise NotImplementedError("Not implemented in base class.")

    def uncertainties(self):
        raise NotImplementedError("Not implemented in base class.")

    def model(self, parameters):
        """Returns model predictions for the expectation value of the data, as a function of the
        parameters.

        Parameters are packed into a 1D array.

        """
        raise NotImplementedError("Not implemented in base class.")

    def weighted_residuals(self, parameters):
        return (self.data() - self.model(parameters)) / self.uncertainties()

    def fit(self, parameters0):
        result = optimize.least_squares(self.weighted_residuals, parameters0)
        pars_est = result.x
        jacobian = result.jac
        covariance_inv = np.dot(jacobian.T, jacobian)
        covariance = linalg.inv(covariance_inv)

        return pars_est, covariance

    def chi_squared(self, parameters):
        return np.sum(self.weighted_residuals(parameters)**2)


class LineFitter(BaseFitter):

    def __init__(self, x, y, y_err):

        self._x = x
        self._y = y
        self._y_err = y_err

    def data(self):
        return self._y

    def uncertainties(self):
        return self._y_err

    def model(self, parameters):
        m = parameters[0]
        b = parameters[1]
        return m * self._x + b


class LineFitterWithSlopeMeasurement(LineFitter):

    def __init__(self, x, y, y_err, slope_meas, slope_err):
        LineFitter.__init__(self, x, y, y_err)
        self._slope_meas = slope_meas

        self._slope_err = slope_err

    def model(self, parameters):
        y_model = LineFitter.model(self, parameters)
        return np.concatenate([y_model, [parameters[0]]])

    def data(self):
        y_data = LineFitter.data(self)
        return np.concatenate([y_data, [self._slope_meas]])

    def uncertainties(self):
        y_errs = LineFitter.uncertainties(self)
        return np.concatenate([y_errs, [self._slope_err]])




