import numpy as np

from os.path import join, dirname


def load_lc(filename='lm0134l19756.time'):
    """
    Read an EROS light curve and return its data.

    Parameters
    ----------
    filename : str, optional
        A light-curve filename.

    Returns
    -------
    dates : numpy.ndarray
        An array of dates.
    magnitudes : numpy.ndarray
        An array of magnitudes.
    errors : numpy.ndarray
        An array of magnitudes errors.
    """

    module_path = dirname(__file__)
    file_path = join(module_path, 'lightcurves', filename)

    data = np.loadtxt(file_path)
    date = data[:, 0]
    mag = data[:, 1]
    err = data[:, 2]

    return date, mag, err
