# -*- coding: utf-8 -*-
'''
Created on 20 Oct 2015

@author: Kimon Tsitsikas

Copyright © 2015-2016 Kimon Tsitsikas and Éric Piel, Delmic

This file is part of Odemis.

Odemis is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License version 2 as published by the Free Software Foundation.

Odemis is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with Odemis. If not, see http://www.gnu.org/licenses/.
'''
from __future__ import division

from concurrent.futures._base import CancelledError, CANCELLED, FINISHED, RUNNING
import logging
import numpy
from odemis import model
from scipy.optimize import curve_fit, OptimizeWarning
import threading
import time
import warnings
from builtins import range

# TODO: this code is full of reliance on numpy being quite lax with wrong
# computation, and easily triggers numpy warnings. To force numpy to be
# stricter:
# import warnings
# warnings.filterwarnings('error')
# numpy.seterr(all='raise')

# TODO: what are the estimation of peak width based on space domain
# rough estimation of peak width based on fitting type
PEAK_WIDTHS = {'gaussian': 1e-4, 'lorentzian': 1e-4}

# These two fitting functions are called back from curve_fit()
# Note: when returning NaN, curve_fit() appears to not like the proposed parameters

def GaussianFit(data, *peaks):
    """
    Applies gaussian fitting to data given the "peaks" parameters.
    data (1d array of floats): the wavelength list for each N pixel
    peaks (list of floats): series of pos, width, amplitude and initial offset
    """
    gau = peaks[-1]  # Automatically converted to a vector in the addition
    for pos, width, amplitude in _Grouped(peaks[:-1], 3):
        sprime = pos * abs(width)
        gau += abs(amplitude) * _Normalize(numpy.exp(-(data - pos) ** 2 / sprime ** 2))
    return gau


def LorentzianFit(data, *peaks):
    """
    Applies lorentzian fitting to data given the "peaks" parameters.
    """
    lor = peaks[-1]
    for pos, width, amplitude in _Grouped(peaks[:-1], 3):
        wprime = pos * abs(width)
        lor += abs(amplitude) * _Normalize(wprime ** 2 / ((data - pos) ** 2 + wprime ** 2))

    return lor


def Smooth(signal, window_len=11, window='hanning'):
    """
    Based at https://github.com/dannyjacobs/capo/blob/master/dcj/cals/smooth.py

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    signal (list of floats): The input signal.
    window_len (int): The dimension of the smoothing window; should be an odd integer.
    window (str): The type of window from 'flat', 'hanning', 'hamming', 'bartlett',
    'blackman' flat window will produce a moving average smoothing.
    returns (list of floats): The smoothed signal
    """
    # First check if parameters can be processed
    if signal.ndim != 1:
        raise ValueError("Smooth only accepts 1 dimension arrays.")
    if signal.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return signal
    if window not in ('flat', 'hanning', 'hamming', 'bartlett', 'blackman'):
        raise ValueError("Window has to be one of 'flat', 'hanning', "
                         "'hamming', 'bartlett', 'blackman'")
    s = numpy.r_[2 * signal[0] - signal[window_len - 1::-1],
                 signal,
                 2 * signal[-1] - signal[-1:-window_len:-1]]
    if window == 'flat':  # moving average
        w = numpy.ones(window_len, 'd')
    else:
        f = getattr(numpy, window)
        w = f(window_len)
    y = numpy.convolve(w / w.sum(), s, mode='same')
    return y[window_len:-window_len + 1]

PEAK_FUNCTIONS = {'gaussian': GaussianFit, 'lorentzian': LorentzianFit}

def Detect(y_vector, x_vector=None, lookahead=5, delta=0):
    """
    Inspired by MATLAB script at http://billauer.co.il/peakdet.html

    Finds the local maxima and minima ("peaks") in the signal represented by
    y_vector.
    y_vector (list of floats): The signal where the peaks are to be found on.
    x_vector (list of floats): Represents the position of the corresponding element
    of y_vector.
    lookahead (int): Distance to look ahead from a peak candidate to determine
    if it is an actual peak.
    delta (int): Minimum difference between a candidate peak and the following
    points for this to be considered a peak.
    returns (2 lists of tuple of floats): Contain the positive and negative peaks
    respectively.
    """
    maxtab = []
    mintab = []
    dump = []

    length = len(y_vector)
    if x_vector is None:
        x_vector = numpy.arange(length)

    # First check if parameters can be processed
    if length != len(x_vector):
        raise ValueError("Input vectors y_vector and x_vector must have same length")
    elif lookahead < 1:
        raise ValueError("Lookahead must be greater than 1")
    elif not (numpy.isscalar(delta) and delta >= 0):
        raise ValueError("Delta must be a positive number")

    y_vector = numpy.asarray(y_vector)
    mn, mx = numpy.Inf, -numpy.Inf

    # Compare candidate peak to the lookahead amount of points in front of it
    for index, (x, y) in enumerate(zip(x_vector[:-lookahead], y_vector[:-lookahead])):
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x

        if y < mx - delta and mx != numpy.Inf:
            if y_vector[index:index + lookahead].max() < mx:
                maxtab.append((mxpos, mx))
                dump.append(True)
                mx = numpy.Inf
                mn = numpy.Inf

        if y > mn + delta and mn != -numpy.Inf:
            if y_vector[index:index + lookahead].min() > mn:
                mintab.append((mnpos, mn))
                dump.append(False)
                mn = -numpy.Inf
                mx = -numpy.Inf

    # remove the first value since it is always detected as peak
    try:
        if dump[0]:
            maxtab.pop(0)
        else:
            mintab.pop(0)
        del dump
    except IndexError:
        # just return empty lists
        pass

    return maxtab, mintab


h_planck = 6.6260715e-34  # J/s
c_light = 299792458  # m/s
e_charge = 1.602e-19  # J --> eV

class PeakFitter(object):
    def __init__(self):
        # will take care of executing peak fitting asynchronously
        # Maximum one task at a time as curve_fit() is not thread-safe
        self._executor = model.CancellableThreadPoolExecutor(max_workers=1)

    def __del__(self):
        if self._executor:
            self._executor.cancel()
            self._executor.shutdown()
            self._executor = None
        logging.debug("PeakFitter destroyed")

    def Fit(self, spectrum, wavelength, type='gaussian'):
        """
        Wrapper for _DoFit. It provides the ability to check the progress of fitting
        procedure or even cancel it.
        spectrum (1d array of floats): The data representing the spectrum.
        wavelength (1d array of floats): The wavelength values corresponding to the
        spectrum given.
        type (str): Type of fitting to be applied (for now only ‘gaussian’ and
        ‘lorentzian’ are available).
        returns (model.ProgressiveFuture):  Progress of DoFit
        """
        # Create ProgressiveFuture and update its state to RUNNING
        est_start = time.time() + 0.1
        f = model.ProgressiveFuture(start=est_start,
                                    end=est_start + self.estimateFitTime(spectrum))
        f._fit_state = RUNNING
        f._fit_lock = threading.Lock()
        f.task_canceller = self._CancelFit

        return self._executor.submitf(f, self._DoFit, f, spectrum, wavelength, type)

    def _DoFit(self, future, spectrum, wavelength, type='gaussian'):
        """
        Smooths the spectrum signal, detects the peaks and applies the type of peak
        fitting required. Finally returns the optimized peak parameters.
        future (model.ProgressiveFuture): Progressive future provided by the wrapper
        spectrum (1d array of floats): The data representing the spectrum.
        wavelength (1d array of floats): The wavelength values corresponding to the
        spectrum given.
        type (str): Type of fitting to be applied (for now only ‘gaussian’ and
        ‘lorentzian’ are available).
        returns:
             params (list of 3-tuple): Each peak parameters as (pos, width, amplitude)
             offset (float): global offset to add
        raises:
                KeyError if given type not available
                ValueError if fitting cannot be applied
        """
        try:
            # values based on experimental datasets
            if len(wavelength) >= 2000:
                divider = 20
            elif len(wavelength) >= 1000:
                divider = 25
            else:
                divider = 30
            init_window_size = max(3, len(wavelength) // divider)
            window_size = init_window_size
            logging.debug("Starting peak detection on data (len = %d) with window = %d",
                          len(wavelength), window_size)
            # TODO: Check if wavelength is in meters, or pixels. Apply Jacobian transformation only in case of meters.
            energy = applyJacobian_x(wavelength)
            spectra_energy = applyJacobian_y(wavelength, spectrum)
            try:
                width = PEAK_WIDTHS[type]
                FitFunction = PEAK_FUNCTIONS[type]
            except KeyError:
                raise KeyError("Given type %s not in available fitting types: %s" % (type, list(PEAK_FUNCTIONS.keys())))
            for step in range(5):
                if future._fit_state == CANCELLED:
                    raise CancelledError()
                smoothed = Smooth(spectrum, window_len=window_size)
                # Increase window size until peak detection finds enough peaks to fit
                # the spectrum curve
                peaks = Detect(smoothed, wavelength, lookahead=window_size, delta=5)[0]
                if not peaks:
                    window_size = int(round(window_size * 1.2))
                    logging.debug("Retrying to fit peak with window = %d", window_size)
                    continue

                constant = h_planck * c_light
                fit_list = []
                for (pos, amplitude) in peaks:
                    pos_e = constant/pos
                    width_e = constant/width
                    amplitude_e = (amplitude * pos**2) / constant
                    fit_list.append(pos_e)
                    fit_list.append(width_e)
                    fit_list.append(amplitude_e)
                logging.debug("-----------print fit_list in eV--------------%s", fit_list)
                # # Initialize offset to 0
                fit_list.append(0)

                if future._fit_state == CANCELLED:
                    raise CancelledError()

                try:
                    with warnings.catch_warnings():
                        # Hide scipy/optimize/minpack.py:690: OptimizeWarning: Covariance of the parameters could not be estimated
                        warnings.filterwarnings("ignore", "", OptimizeWarning)
                        # TODO, from scipy 0.17, curve_fit() supports the 'bounds' parameter.
                        # It could be used to ensure the peaks params are positives.
                        # (Once we don't support Ubuntu 12.04)
                        # params, _ = curve_fit(FitFunction, wavelength, spectrum, p0=fit_list)
                        params, _ = curve_fit(FitFunction, energy, spectra_energy, p0=fit_list)
                        logging.debug("-------params after fitting in eV--------%s", params)
                    break
                except Exception:
                    window_size = int(round(window_size * 1.2))
                    logging.debug("Retrying to fit peak with window = %d", window_size)
                    continue
            else:
                raise ValueError("Could not apply peak fitting of type %s." % type)
            # reformat parameters to (list of 3 tuples, offset)
            # peaks_params = []
            # for pos, width, amplitude in _Grouped(params[:-1], 3):
            #     # Note: to avoid negative peaks, the fit functions only take the
            #     # absolute of the amplitude/width. So now amplitude and width
            #     # have 50% chances to be negative => Force positive now.
            #     peaks_params.append((pos, abs(width), abs(amplitude)))

            peaks_params = []
            for pos_e, width_e, amplitude_e in _Grouped(params[:-1], 3):
                pos = constant/pos_e
                width = constant/width_e
                amplitude = amplitude_e * constant / pos**2
                # Note: to avoid negative peaks, the fit functions only take the
                # absolute of the amplitude/width. So now amplitude and width
                # have 50% chances to be negative => Force positive now.
                peaks_params.append((pos, abs(width), abs(amplitude)))

            params = peaks_params, params[-1]
            return params
        except CancelledError:
            logging.debug("Fitting of type %s was cancelled.", type)
        finally:
            with future._fit_lock:
                if future._fit_state == CANCELLED:
                    raise CancelledError()
                future._fit_state = FINISHED

    def _CancelFit(self, future):
        """
        Canceller of _DoFit task.
        """
        logging.debug("Cancelling fitting...")

        with future._fit_lock:
            if future._fit_state == FINISHED:
                return False
            future._fit_state = CANCELLED
            logging.debug("Fitting cancelled.")

        return True

    def estimateFitTime(self, data):
        """
        Estimates fitting duration
        """
        # really rough estimation
        return len(data) * 10e-3  # s

def applyJacobian_x(wavelength):
    constant = h_planck * c_light
    energy = numpy.divide(constant, wavelength)
    return energy

def applyJacobian_y(wavelength, spectrum):
    constant = h_planck * c_light
    wavelength_2 = numpy.power(wavelength, 2)
    spectra_space = numpy.multiply(spectrum, wavelength_2)
    spectra_energy = numpy.divide(spectra_space, constant)
    return spectra_energy

def Curve(wavelength, peak_parameters, offset, type='gaussian'):
    """
    Given the peak parameters and the wavelength values returns the actual
    dataset of curve points.
    wavelength (1d array of floats): The wavelength values corresponding to the
    spectrum given.
    peak_parameters (list of tuples): The parameters of the peak curves to
    be depicted.
    offset (float): peaks offset
    type (str): Type of fitting to be applied (for now only ‘gaussian’ and
    ‘lorentzian’ are available).
    returns (1d array of floats): Dataset of points representing the curve.
    raises:
            KeyError if given type not available
            ValueError if fitting cannot be applied
    """
    try:
        FitFunction = PEAK_FUNCTIONS[type]
    except KeyError:
        raise KeyError("Given type %s not in available fitting types: %s" % (type, list(PEAK_FUNCTIONS.keys())))

    logging.debug("-----------------INSIDE CURVE FUNCTION----------------------------")
    logging.debug("--------peak_parameters----%s", peak_parameters)

    # Flatten the peak parameters tuples
    peak_flat = [p for l in peak_parameters for p in l]
    peak_flat.append(offset)
    logging.debug("-------------PEAK FLAT--------------%s", peak_flat)

    constant = h_planck * c_light
    peaks_params = []
    for (pos, width, amplitude) in _Grouped(peak_flat[:-1], 3):
        pos_e = constant / pos
        width_e = constant / width
        amplitude_e = amplitude * (pos ** 2) / constant
        peaks_params.append((pos_e, abs(width_e), abs(amplitude_e)))
        peak_pos = pos

    peak_flat_e = [p for l in peaks_params for p in l]
    peak_flat_e.append(offset)

    logging.debug("-------- peak parameters after the conversion to energy----- %s", peaks_params)
    logging.debug("-------- peak flat after the conversion to energy----- %s", peak_flat_e)
    energy = numpy.divide(constant, wavelength)

    # curve = FitFunction(wavelength, *peak_flat)
    curve_e = FitFunction(energy, *peak_flat_e)
    wavelength_2 = numpy.power(wavelength, 2)
    curve_c = numpy.multiply(curve_e, constant)
    curve = numpy.divide(curve_c, wavelength_2)
    logging.debug("------------CURVE---------%s", curve)
#         residual = numpy.sqrt((abs(output - spectrum) ** 2).sum() / len(spectrum))
#         logging.info("Residual error of spectrum fitting is %f", residual)
    return curve


def _Grouped(iterable, n):
    """
    Iterate over the iterable, n elements at a time
    """
    return zip(*[iter(iterable)] * n)


def _Normalize(vector):
    normfac = numpy.max(vector)
    if normfac == 0:
        # TODO: raise a ValueError instead? But that confuses curve_fit a lot
        logging.debug("Tried to normalize null vector")
        return float("NaN")

    vecnorm = vector / normfac
    return vecnorm
