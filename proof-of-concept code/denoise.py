import numpy as np
import time
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from preprocessing import sine_sync, denoise_wavelet

# def remove_sine_50_hz(signal, order, Fs):
#
#     sos = sig.butter(order, 60, btype='highpass', fs=Fs, output='sos')
#
#     return sig.sosfiltfilt(sos, signal.T).T

t = np.linspace(0, 1, 800000)

for sample in range(0, 50):

    train_data = pq.read_pandas('train.parquet', columns=[str(sample)]).to_pandas()
    data_array = np.asarray(train_data)
    sync_sig = sine_sync(data_array)
    wavelet_sig = denoise_wavelet(sync_sig, wavelet_level=2)

    plt.subplot(311)
    plt.plot(t, data_array, label='original')
    plt.grid()
    plt.legend(loc='upper right')
    plt.subplot(312)
    plt.plot(t, sync_sig, label='synchronized')
    plt.grid()
    plt.legend(loc='upper right')
    plt.subplot(313)
    plt.plot(t, wavelet_sig, label='de-noised')
    plt.grid()
    plt.xlabel('Cycle')
    plt.show()

    time.sleep(2)
    plt.close()

