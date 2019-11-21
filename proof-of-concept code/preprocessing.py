import numpy as np
from scipy import signal as sig
import pywt, time, pandas as pd


def sine_sync(signal):
    filt = np.ones((1000,)) / 1000

    filt_signal = np.convolve(signal[:, 0], filt, mode='valid')

    # zero_crossings = np.where(np.diff(np.sign(filt_signal)))[0]
    neg_pos_zero_crossing = np.where(np.diff(np.sign(filt_signal)) > 0)[0][0]

    first_portion = signal[neg_pos_zero_crossing::]
    second_portion = signal[0:neg_pos_zero_crossing]

    sync_sig = np.concatenate((first_portion[:, 0], second_portion[:, 0]))

    return sync_sig


def denoise_wavelet(signal, wavelet_level):
    n = len(signal)
    # wavelet_level MUST be greater than or equal to 2!!!
    wavelet = pywt.Wavelet('db4')

    coeffs = pywt.wavedec(signal, wavelet, level=wavelet_level, axis=0)

    new_d_coeffs = [None] * (wavelet_level)

    for dLevel in range(1, wavelet_level + 1):
        mad = float(pd.DataFrame(np.abs(coeffs[dLevel])).mad())
        threshold = (1 / (0.6745 * mad)) * np.sqrt(2 * np.log(n))

        new_d_out = np.asarray(list(coeffs[dLevel]))
        new_d_out[np.abs(new_d_out) <= threshold] = 0

        new_d_coeffs[dLevel - 1] = new_d_out

    # Signal re-construction
    zero_approx = np.zeros(coeffs[0].shape)
    new_coeffs = [zero_approx, new_d_coeffs[0], new_d_coeffs[1]]

    reconstructed_signal = pywt.waverec(new_coeffs, wavelet, axis=0)
    return reconstructed_signal
