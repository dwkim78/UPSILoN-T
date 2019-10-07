"""
Original code from UPSILoN (Kim & Bailer-Jones 2016).
https://github.com/dwkim78/upsilon
"""

import warnings
import multiprocessing

import numpy as np
import scipy.stats as ss

from collections import OrderedDict
from scipy.optimize import leastsq
from upsilont.features.period_LS_pyfftw import fasper, significance


def get_train_feature_name():
    """
    Return a list of features' names.

    Features' name that are used to train a model and predict a class.
    Sorted by the names.

    Returns
    -------
    feature_names : list
        A list of features' names.
    """

    features = ['amplitude', 'hl_amp_ratio', 'kurtosis', 'period',
                'phase_cusum', 'phase_eta', 'phi21', 'phi31', 'quartile31',
                'r21', 'r31', 'shapiro_w', 'skewness', 'slope_per10',
                'slope_per90', 'stetson_k']
    features.sort()

    return features


def get_all_feature_name():
    """
    Return a list of entire features.

    A set of entire features regardless of being used to train a model or
    predict a class.

    Returns
    -------
    feature_names : list
        A list of features' names.
    """

    features = get_train_feature_name()

    features.append('cusum')
    features.append('eta')
    features.append('n_points')
    features.append('period_SNR')
    features.append('period_log10FAP')
    features.append('period_uncertainty')
    features.append('weighted_mean')
    features.append('weighted_std')

    features.sort()

    return features


class VariabilityFeatures:
    def __init__(self, date, mag, err=None, n_threads=4, min_period=0.03):
        """
        Extract variability features of a light curve.

        Parameters
        ----------
        date : array_like
            An array of observed date, in days.
        mag : array_like
            An array of observed magnitude.
        err : array_like, optional
            An array of magnitude error. If None, std(mag) will be used.
        n_threads : int, optional
            The number of cores to use to derive periods.
        min_period : float, optional
            The minimum period to calculate.
        """

        # Variable to calculate.
        self.n_points = None
        self.weight = None
        self.weighted_mean = None
        self.weighted_std = None
        self.weighted_sum = None
        self.mean = None
        self.median = None
        self.std = None
        self.skewness = None
        self.kurtosis = None
        self.shapiro_w = None
        self.quartile31 = None
        self.stetson_k = None
        self.hl_amp_ratio = None
        self.cusum = None
        self.eta = None
        self.phase_eta = None
        self.slope_per10 = None
        self.slope_per90 = None
        self.phase_cusum = None
        self.f = None
        self.period = None
        self.period_uncertainty = None
        self.period_log10FAP = None
        self.period_SNR = None
        self.amplitude = None
        self.r21 = None
        self.r31 = None
        self.f_phase = None
        self.phi21 = None
        self.phi31 = None

        # Set basic values.
        if not isinstance(date, np.ndarray):
            date = np.array(date)
        if not isinstance(mag, np.ndarray):
            mag = np.array(mag)

        self.date = date
        self.mag = mag
        if err is not None:
            if not isinstance(err, np.ndarray):
                err = np.array(err)
            self.err = err
        else:
            self.err = np.ones(len(self.mag)) * np.std(self.mag)

        # Check length.
        if (len(self.date) != len(self.mag)) or \
           (len(self.date) != len(self.err)) or \
           (len(self.mag) != len(self.err)):
            raise RuntimeError('The length of date, mag, and err must be same.')

        # if the number of data points is too small.
        min_n_data = 80
        if len(self.date) < min_n_data:
            warnings.warn('The number of data points are less than %d.'
                          % min_n_data)

        n_threads = int(n_threads)
        if n_threads > multiprocessing.cpu_count():
            self.n_threads = multiprocessing.cpu_count()
        else:
            if n_threads <= 0:
                self.n_threads = 1
            else:
                self.n_threads = n_threads

        min_period = float(min_period)
        if min_period <= 0:
            self.min_period = 0.03
        else:
            self.min_period = min_period

    def run(self):
        """Run feature extraction modules."""

        # shallow_run must be executed prior to deep_run
        # since shallow_run calculates several values needed for deep_run.
        self.shallow_run()
        self.deep_run()

    def shallow_run(self):
        """Derive not-period-based features."""
        # Number of data points
        self.n_points = len(self.date)

        # Weight calculation.
        # All zero values.
        if not self.err.any():
            self.err = np.ones(len(self.mag)) * np.std(self.mag)
        # Some zero values.
        elif not self.err.all():
            np.putmask(self.err, self.err == 0, np.median(self.err))

        self.weight = 1. / self.err
        self.weighted_sum = np.sum(self.weight)

        # Simple statistics, mean, median and std.
        self.mean = np.mean(self.mag)
        self.median = np.median(self.mag)
        self.std = np.std(self.mag)

        # Weighted mean and std.
        self.weighted_mean = np.sum(self.mag * self.weight) / self.weighted_sum
        self.weighted_std = np.sqrt(np.sum((self.mag - self.weighted_mean) ** 2
                                           * self.weight) / self.weighted_sum)

        # Skewness and kurtosis.
        self.skewness = ss.skew(self.mag)
        self.kurtosis = ss.kurtosis(self.mag)

        # Normalization-test. Shapiro-Wilk test.
        shapiro = ss.shapiro(self.mag)
        self.shapiro_w = shapiro[0]

        # Percentile features.
        self.quartile31 = np.percentile(self.mag, 75) - \
            - np.percentile(self.mag, 25)

        # Stetson K.
        self.stetson_k = self.get_stetson_k(self.mag, self.median, self.err)

        # Ratio between higher and lower amplitude than average.
        self.hl_amp_ratio = self.half_mag_amplitude_ratio(
            self.mag, self.median, self.weight)

        # Cusum
        self.cusum = self.get_cusum(self.mag)

        # Eta
        self.eta = self.get_eta(self.mag, self.weighted_std)

    def deep_run(self):
        """Derive period-based features."""
        # Lomb-Scargle period finding.
        self.get_period_LS(self.date, self.mag, self.n_threads, self.min_period)

        # Features based on a phase-folded light curve
        # such as Eta, slope-percentile, etc.
        # Should be called after the getPeriodLS() is called.

        # Created phased a folded light curve.
        # We use period * 2 to take eclipsing binaries into account.
        phase_folded_date = self.date % (self.period * 2.)
        sorted_index = np.argsort(phase_folded_date)

        folded_date = phase_folded_date[sorted_index]
        folded_mag = self.mag[sorted_index]

        # phase Eta
        self.phase_eta = self.get_eta(folded_mag, self.weighted_std)

        # Slope percentile.
        self.slope_per10, self.slope_per90 = \
            self.slope_percentile(folded_date, folded_mag)

        # phase Cusum
        self.phase_cusum = self.get_cusum(folded_mag)

    def get_period_LS(self, date, mag, n_threads, min_period):
        """
        Period finding using the Lomb-Scargle algorithm.

        Finding two periods. The second period is estimated after whitening
        the first period. Calculating various other features as well
        using derived periods.

        Parameters
        ----------
        date : array_like
            An array of observed date, in days.
        mag : array_like
            An array of observed magnitude.
        n_threads : int
            The number of threads to use.
        min_period : float
            The minimum period to calculate.
        """

        # DO NOT CHANGE THESE PARAMETERS.
        oversampling = 3.
        hifac = int((max(date) - min(date)) / len(date) / min_period * 2.)

        # Minimum hifac
        if hifac < 100:
            hifac = 100

        # Lomb-Scargle.
        fx, fy, nout, jmax, prob = fasper(date, mag, oversampling, hifac,
                                              n_threads)

        self.f = fx[jmax]
        self.period = 1. / self.f
        self.period_uncertainty = self.get_period_uncertainty(fx, fy, jmax)
        self.period_log10FAP = \
            np.log10(significance(fx, fy, nout, oversampling)[jmax])
        self.period_SNR = (fy[jmax] - np.median(fy)) / np.std(fy)

        # Fit Fourier Series of order 3.
        order = 3
        # Initial guess of Fourier coefficients.
        p0 = np.ones(order * 2 + 1)
        date_period = (date % self.period) / self.period
        p1, success = leastsq(self.residuals, p0,
                              args=(date_period, mag, order))

        # Derive Fourier features for the first period.
        # Petersen, J. O., 1986, A&A
        self.amplitude = np.sqrt(p1[1] ** 2 + p1[2] ** 2)
        self.r21 = np.sqrt(p1[3] ** 2 + p1[4] ** 2) / self.amplitude
        self.r31 = np.sqrt(p1[5] ** 2 + p1[6] ** 2) / self.amplitude
        self.f_phase = np.arctan(-p1[1] / p1[2])
        self.phi21 = np.arctan(-p1[3] / p1[4]) - 2. * self.f_phase
        self.phi31 = np.arctan(-p1[5] / p1[6]) - 3. * self.f_phase

    def get_period_uncertainty(self, fx, fy, jmax, fx_width=100):
        """
        Get uncertainty of a period.

        The uncertainty is defined as the half width of the frequencies
        around the peak, that becomes lower than average + standard deviation
        of the power spectrum.

        Since we may not have fine resolution around the peak,
        we do not assume it is gaussian. So, no scaling factor of
        2.355 (= 2 * sqrt(2 * ln2)) is applied.

        Parameters
        ----------
        fx : array_like
            An array of frequencies.
        fy : array_like
            An array of amplitudes.
        jmax : int
            An index at the peak frequency.
        fx_width : int, optional
            Width of power spectrum to calculate uncertainty.

        Returns
        -------
        p_uncertain : float
            Period uncertainty.
        """

        # Get subset
        start_index = jmax - fx_width
        end_index = jmax + fx_width
        if start_index < 0:
            start_index = 0
        if end_index > len(fx) - 1:
            end_index = len(fx) - 1

        fx_subset = fx[start_index:end_index]
        fy_subset = fy[start_index:end_index]
        fy_mean = np.median(fy_subset)
        fy_std = np.std(fy_subset)

        # Find peak
        max_index = np.argmax(fy_subset)

        # Find list whose powers become lower than average + std.
        index = np.where(fy_subset <= fy_mean + fy_std)[0]

        # Find the edge at left and right. This is the full width.
        left_index = index[(index < max_index)]
        if len(left_index) == 0:
            left_index = 0
        else:
            left_index = left_index[-1]
        right_index = index[(index > max_index)]
        if len(right_index) == 0:
            right_index = len(fy_subset) - 1
        else:
            right_index = right_index[0]

        # We assume the half of the full width is the period uncertainty.
        half_width = (1. / fx_subset[left_index]
                      - 1. / fx_subset[right_index]) / 2.
        period_uncertainty = half_width

        return period_uncertainty

    def residuals(self, pars, x, y, order):
        """
        Residual of Fourier Series.

        Parameters
        ----------
        pars : array_like
            Fourier series parameters.
        x : array_like
            An array of date.
        y : array_like
            An array of true values to fit.
        order : int
            An order of Fourier Series.
        """

        return y - self.fourier_series(pars, x, order)

    def fourier_series(self, pars, x, order):
        """
        Function to fit Fourier Series.

        Parameters
        ----------
        x : array_like
            An array of date divided by period. It doesn't need to be sorted.
        pars :  array_like
            Fourier series parameters.
        order : int
            An order of Fourier series.
        """

        sums = pars[0]
        for i in range(order):
            sums += pars[i * 2 + 1] * np.sin(2 * np.pi * (i + 1) * x) \
                   + pars[i * 2 + 2] * np.cos(2 * np.pi * (i + 1) * x)

        return sums

    def get_stetson_k(self, mag, avg, err):
        """
        Return Stetson K feature.

        Parameters
        ----------
        mag : array_like
            An array of magnitude.
        avg : float
            An average value of magnitudes.
        err : array_like
            An array of magnitude errors.

        Returns
        -------
        stetson_k : float
            Stetson K value.
        """

        residual = (mag - avg) / err
        stetson_k = np.sum(np.fabs(residual)) \
                    / np.sqrt(np.sum(residual * residual)) / np.sqrt(len(mag))

        return stetson_k

    def half_mag_amplitude_ratio(self, mag, avg, weight):
        """
        Return ratio of amplitude of higher and lower magnitudes.


        A ratio of amplitude of higher and lower magnitudes than average,
        considering weights. This ratio, by definition, should be higher
        for EB than for others.

        Parameters
        ----------
        mag : array_like
            An array of magnitudes.
        avg : float
            An average value of magnitudes.
        weight : array_like
            An array of weight.

        Returns
        -------
        hl_ratio : float
            Ratio of amplitude of higher and lower magnitudes than average.
        """

        # For lower (fainter) magnitude than average.
        index = np.where(mag > avg)
        lower_weight = weight[index]
        lower_weight_sum = np.sum(lower_weight)
        lower_mag = mag[index]
        lower_weighted_std = np.sum((lower_mag
                                     - avg) ** 2 * lower_weight) / \
                             lower_weight_sum

        # For higher (brighter) magnitude than average.
        index = np.where(mag <= avg)
        higher_weight = weight[index]
        higher_weight_sum = np.sum(higher_weight)
        higher_mag = mag[index]
        higher_weighted_std = np.sum((higher_mag
                                      - avg) ** 2 * higher_weight) / \
                              higher_weight_sum

        # Return ratio.
        return np.sqrt(lower_weighted_std / higher_weighted_std)

    def get_eta(self, mag, std):
        """
        Return Eta feature.

        Parameters
        ----------
        mag : array_like
            An array of magnitudes.
        std : array_like
            A standard deviation of magnitudes.

        Returns
        -------
        eta : float
            The value of Eta index.
        """

        diff = mag[1:] - mag[:len(mag) - 1]
        eta = np.sum(diff * diff) / (len(mag) - 1.) / std / std

        return eta

    def slope_percentile(self, date, mag):
        """
        Return 10% and 90% percentile of slope.

        Parameters
        ----------
        date : array_like
            An array of phase-folded date. Sorted.
        mag : array_like
            An array of phase-folded magnitudes. Sorted by date.

        Returns
        -------
        per_10 : float
            10% percentile values of slope.
        per_90 : float
            90% percentile values of slope.
        """

        date_diff = date[1:] - date[:len(date) - 1]
        mag_diff = mag[1:] - mag[:len(mag) - 1]

        # Remove zero mag_diff.
        index = np.where(mag_diff != 0.)
        date_diff = date_diff[index]
        mag_diff = mag_diff[index]

        # Derive slope.
        slope = date_diff / mag_diff

        percentile_10 = np.percentile(slope, 10.)
        percentile_90 = np.percentile(slope, 90.)

        return percentile_10, percentile_90

    def get_cusum(self, mag):
        """
        Return max - min of cumulative sum.

        Parameters
        ----------
        mag : array_like
            An array of magnitudes.

        Returns
        -------
        mm_cusum : float
            Max - min of cumulative sum.
        """

        c = np.cumsum(mag - self.weighted_mean) / len(mag) / self.weighted_std

        return np.max(c) - np.min(c)

    def get_features(self, for_train=True):
        """
        Return all features with its names. Sorted by the names.

        Parameters
        ----------
        for_train : boolean
            If true, returns features for training. If false, returns all
            features.

        Returns
        -------
        features : OrderedDict
            Features dictionary.
        """

        # Get all the names of features.
        all_vars = vars(self)
        features = {}
        if for_train:
            feature_names_list_all = get_train_feature_name()
        else:
            feature_names_list_all = get_all_feature_name()
        for name in all_vars.keys():
            if name in feature_names_list_all:
                features[name] = all_vars[name]

        # Sort by the keys (i.e. feature names).
        features = OrderedDict(sorted(features.items(), key=lambda t: t[0]))

        return features
