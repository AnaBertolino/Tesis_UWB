from scipy.io import whosmat
import matplotlib.pyplot as plt
from src.utils import parse_oximeter_txt, cargar_medicion, slow_time_max_var
from scipy.signal import savgol_filter
from scipy.signal import spectrogram
from scipy.signal import detrend
import numpy as np
from src.preprocess import remove_clutter_MA

def graficar_spectrograma(senal, fs):
    f, t, Sxx = spectrogram(senal, fs=fs, nperseg=256, noverlap=128)
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading='gouraud')
    plt.title("Espectrograma")
    plt.ylabel('Frecuencia [Hz]')
    plt.xlabel('Tiempo [s]')
    plt.colorbar(label='Potencia [dB]')
    plt.tight_layout()
    plt.show(block=False)

def preprocesar_senal(senal):
    # Remueve componente DC y tendencias lineales
    #processed_signal = detrend(senal, 1)
    processed_signal = remove_clutter_MA(senal, 30)
    processed_signal = processed_signal[60:, :]
    #processed_signal = processed_signal - np.mean(processed_signal, axis=0)
    #processed_signal = lowpass_filter(processed_signal, 0.01, fs_st)
    return processed_signal
# Path to a sample .mat file from the dataset
# Update this path to point to a file you downloaded from the GitHub
mat_file_path = "data/Multi-Radar-Dataset-main/Scenario 1/jxk_zxy_1sit_2stand/20210105_radar2_1sit_2stand_jxk_zxy_static_random_1.mat"
txt_file_path = "data/Multi-Radar-Dataset-main/Scenario 1/jxk_zxy_1sit_2stand/jxk/2021-01-06 084931.509WaveFile.txt"


times, hr = parse_oximeter_txt(txt_file_path)

# Savitzky-Golay filter for smoothing (window size must be odd and less than data length)
window_size = 11 if len(hr) >= 11 else len(hr) - (len(hr) + 1) % 2
smoothed_hr = savgol_filter(hr, window_size, polyorder=1)

hr_hz = [x / 60 for x in hr]
smoothed_hr_hz = smoothed_hr / 60

# Load the .mat file
data = cargar_medicion(mat_file_path, 'data')
#data = data[:, :-1]  # remove last fast-time sample (438th column)
data_processed = preprocesar_senal(data) # no me va a servir el procesamiento que estaba haciendo antes pq ahora el clutter es dinámico
data_processed = data

# Plot
plt.figure(figsize=(10, 5))
plt.imshow(data_processed, aspect='auto', cmap='jet', origin='lower',
           extent=[0, data.shape[1], 0, data.shape[0]])
plt.colorbar(label='Amplitude (dB)')
plt.xlabel('Fast Time (range bins)')
plt.ylabel('Slow Time (sweeps)')
plt.title('Radar Slow-Time vs Fast-Time Matrix')
plt.tight_layout()
plt.show()
fs_st = 20 #sampling frecuency in slow time
#data_processed = preprocesar_senal(data)
signal = data_processed[:, slow_time_max_var(data_processed)[0]]
print(slow_time_max_var(data_processed)[0])

graficar_spectrograma(signal, fs_st)

# Plot the heart rate over time
plt.figure(figsize=(12, 5))
plt.plot(times, hr_hz, label="Raw (Hz)", alpha=0.6, marker='o')
plt.plot(times, smoothed_hr_hz, label="Smoothed (Hz)", linewidth=2)
plt.title("Oximeter Heart Rate in Hz")
plt.xlabel("Time")
plt.ylabel("Heart Rate (Hz)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show(block=False)
plt.show()

plt.figure()
plt.plot(signal)
plt.show(block=False)
plt.show()


'''
# Let's assume there's a variable named 'UWB_data' or similar inside
# You may need to replace 'UWB_data' with the actual key name
sample_key = [key for key in data.keys() if not key.startswith('__')][0]
uwb_signal = data[sample_key]

# Print shape and preview of the signal
print(f"\nData shape: {uwb_signal.shape}")
print("First 5 rows:\n", uwb_signal[:5])

# Plot the first signal trace (first row, for example)
plt.plot(uwb_signal[0])
plt.title(f"Sample Radar Signal - {sample_key}")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
'''