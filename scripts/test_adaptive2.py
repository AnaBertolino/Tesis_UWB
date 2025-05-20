import adaptivesswt
import numpy as np
import scipy
import matplotlib.pyplot as plt
from src.filters import extended_kalman_filter, kalman_filter, bandpass_filter
ruta_raw = "data/Mendeley Data/Raw Radar Data/Bandwidth3/DeltaR=10cm_Angle=0_Band3_Supine_Trial8.mat"
# ruta_raw =  "data/Mendeley Data/Raw Radar Data/Bandwidth1/DeltaR=8cm_Angle=30_Band1_Lateral_Trial1.mat" # Esta es la q funciona bárbaro

data = scipy.io.loadmat(ruta_raw)
signal = data['bScan']

fs = 18.75 #Hz
signal_slow_time = signal[:, np.argmax(np.var(signal, axis=0))]
signal_slow_time_raw = signal_slow_time - np.mean(signal_slow_time)

signal_slow_time = bandpass_filter(signal_slow_time_raw, fs, 0.05, 3)

ext_kalman_filtered_out, ext_kalman_filtered_states = extended_kalman_filter(signal_slow_time, fs, 0.99, plot=True)
kalman_filtered = np.array(ext_kalman_filtered_states[:, 0])[:, 0]

t = np.arange(len(kalman_filtered)) / fs
#kalman_filtered = kalman_filter(signal_slow_time, fs, 0.0001, 1, plot=True)
plt.figure()
plt.plot(t, kalman_filtered)
#plt.show()



config = adaptivesswt.Configuration(
    min_freq=0.01,
    max_freq=1,
    num_freqs=128,
    ts=1/fs,
    wcf=1,
    wbw=8,
    wavelet_bounds=(-8, 8),
    threshold=1/50,
)

sst, cwt, freqs, tail, _ = adaptivesswt.sswt(kalman_filtered, **config.asdict())

plt.figure()
plt.pcolormesh(t, freqs, np.abs(sst), cmap='plasma', shading='gouraud')
plt.show(block=False)
plt.show()