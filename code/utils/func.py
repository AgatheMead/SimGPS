"""
Utilities for GPS signal structure
"""
import numpy as np
import matplotlib.pyplot as plt


# circular correlation
def circular_correlation(data1, data2):
    return np.correlate(data1, np.hstack((data2[1:], data2)), mode='valid') / len(data1)


# calculate phase differences of two signals with similar frequency
# both signals should be only one period
def phase_diff(data1, data2):

    len_data = len(data1)
    cross_corr = circular_correlation(data1, data2)
    max_idx = np.argmax(cross_corr)
    phase_diff = min(max_idx, len_data - max_idx) / len_data
    return phase_diff / np.pi * 360     # convert to degree


# calculate phase differences of two signals with similar frequency
# insensitive to bit modulation
def phase_diff_insensitive(data1, data2):

    phase_diff1 = phase_diff(data1, data2)
    phase_diff2 = phase_diff(-data1, data2)
    return min(phase_diff1, phase_diff2)


# add noise to signal based on SNR
def awgn(signal, snr):
    # signal power
    p_signal = 10 * np.log10(np.mean(signal ** 2))  # in dB
    p_noise = p_signal - snr
    sigma = np.sqrt(10 ** (p_noise / 10))
    noise = np.random.normal(0, sigma, len(signal))
    return signal + noise


if __name__ == '__main__':

    fi = 10
    fs = 1e3
    t = np.arange(0, 1/fi, 1/fs)
    sig1 = np.sin(2 * np.pi * fi * t)
    sig2 = -np.sin(2 * np.pi * (fi * t + 0.4))
    print(phase_diff_insensitive(sig1, sig2))


