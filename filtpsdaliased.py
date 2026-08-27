import numpy as np


def filtpsdaliased(x, xdata):
    """Aliased Lorentzian PSD used for optical trap calibration.

    This is the direct Python equivalent of the MATLAB function in
    `filtpsdaliased.m`.
    """
    gamma = float(np.asarray(x, dtype=float)[0])
    fc = float(np.asarray(x, dtype=float)[1])
    f = np.asarray(xdata, dtype=float)
    kt = 4.1
    et = 0.650e-3
    f_nyquist = 2.0 * f[-1]

    spectrum = np.zeros_like(f, dtype=float)
    for i, fi in enumerate(f):
        total = 0.0
        for j in range(-40, 41):
            freq_term = fi + (j - 1) * f_nyquist
            denominator = (np.pi ** 2) * gamma * ((freq_term ** 2) + (fc ** 2))
            total += (kt / denominator) * (np.sinc(freq_term * et)) ** 2
        spectrum[i] = total
    return spectrum
