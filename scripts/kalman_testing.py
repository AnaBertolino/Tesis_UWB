import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, detrend
from src.filters import kalman_filter, bandpass_filter, extended_kalman_filter,  notch_filter
from src.preprocess import remove_clutter_MA
from src.utils import frecuencias_dominantes, slow_time_max_var
import scipy.io
from scipy.interpolate import interp1d
from src.analysis import fft_spectrum, czt_spectrum
from scipy.signal import CZT


# PARÁMETROS FIJOS
fs_st = 18.75 #Hz
fs_ref = 1/0.325 #Hz

# Uso del script
Q_value = 0.005
R_value = 1
window_size = 5

# --- 1. Leer datos desde archivos .mat ---
def cargar_medicion(ruta, key):
    data = scipy.io.loadmat(ruta)
    return data[key]  # Ajusta esto según la estructura del archivo


ruta_raw = "data/Mendeley Data/Raw Radar Data/Bandwidth3/DeltaR=10cm_Angle=0_Band3_Supine_Trial2.mat"
ruta_ref = "data/Mendeley Data/Reference Data/Bandwidth3/Ref_DeltaR=10cm_Angle=0_Band3_Supine_Trial2.mat"
ruta_calib = "data/Mendeley Data/Calibration Data/Angle=30/Calibration_Band1_DeltaR=8cm.mat"

medicion = cargar_medicion(ruta_raw, "bScan")
referencia = cargar_medicion(ruta_ref, "Ref")[0, :]
#referencia = referencia - np.mean(referencia)
calibracion = cargar_medicion(ruta_calib, "bScan")


referencia = detrend(referencia)
#referencia = bandpass_filter(referencia, fs_ref, 0.2, 1)
#_, referencia = extended_kalman_filter(referencia, fs_ref, 0.5)
#referencia = np.array(referencia[:, 0])[:, 0]
#referencia = kalman_filter(referencia, fs_ref, Q_value, R_value)
#referencia = referencia[50:]



plt.figure()
plt.plot(referencia)
plt.show(block=False)

# --- 2. Procesamiento (aquí puedes probar distintos métodos) ---
#medicion_filtrada = remove_clutter_MA(medicion, k=30, R_ref=calibracion)
#medicion_filtrada = medicion




# medicion_filtrada tiene la matriz, necesito analizar la señal de mayor varianza en slow time
max_var_idx, var = slow_time_max_var(medicion)
s = medicion[:, max_var_idx]
medicion_filtrada = detrend(s)
# Defino mi señal a realizarle el análisis espectral

s = bandpass_filter(medicion_filtrada, fs_st, 0.15, 1)
#s = notch_filter(s, fs_st, [0.15], 5)
#s = s[200:]



# Calcular la potencia de la señal y del ruido
P_signal = np.mean((medicion[:, max_var_idx])**2)  # Potencia de la señal
P_noise = np.mean((calibracion[:, max_var_idx])**2)   # Potencia del ruido

# Calcular SNR en dB
SNR_dB = 10 * np.log10(P_signal / P_noise)

print(f"SNR: {SNR_dB:.2f} dB")

'''
plt.plot(medicion[:, max_var_idx], label="Medición")
plt.plot(calibracion[:, max_var_idx], label="Calibración")
plt.plot(s, label="Señal procesada")

plt.legend()  # Agrega la leyenda automáticamente
plt.show()
'''

def calculate_dominant_frequency(signal, fs, nperseg):
    """Calcula la frecuencia dominante de la señal usando FFT."""
    freqs, psd = welch(signal, fs, nfft=8192, window=('kaiser', 0.7), nperseg=100)
    #freqs, psd = fft_spectrum(signal, fs, plot=False)
    print(len(freqs))
    print(len(psd))
    #freqs_s_czt, espectro_s_czt = czt_spectrum(signal, 500, fs, 0.01, 1)

    #chirp_z = CZT(n=len(signal))

    #espectro = chirp_z(signal)

    dominant_freq = frecuencias_dominantes(freqs, psd, rango=(0.1, 0.4), n=1)
    return freqs, psd, dominant_freq, 0, 0

def process_signal(signal, reference_signal, fs_signal, fs_reference, Q_value, R_value):
    """Aplica filtros y analiza la señal."""
    kalman_filtered1 = kalman_filter(signal, fs_signal, Q_value, R_value)
    ext_kalman_filtered_out, ext_kalman_filtered_states = extended_kalman_filter(signal, fs_st, 0.8, plot=False)
    
    #kalman_filtered = signal

    kalman_filtered = np.array(ext_kalman_filtered_states[:, 0])[:, 0]
    #kalman_filtered = ext_kalman_filtered_out
    
     # Asegurar el mismo nperseg para ambas señales
    min_length = min(len(kalman_filtered), len(reference_signal))
    nperseg = min(1024, min_length)
    
    # Cálculo de espectro
    #freqs_signal, psd_filtered, dominant_freq_filtered, freqs_czt, espectro_czt = calculate_dominant_frequency(kalman_filtered, fs_signal, nperseg)
    #freqs_reference, psd_reference, _, freqs_czt_ref, espectro_czt_ref = calculate_dominant_frequency(reference_signal, fs_reference, nperseg)
    #freqs_reference, psd_reference = fft_spectrum(reference_signal, fs_ref, plot=False)
    #freqs_signal, psd_filtered = fft_spectrum(kalman_filtered, fs_st, plot=False)
    #freqs_signal1, psd_filtered1 = fft_spectrum(kalman_filtered1, fs_st, plot=False)

    # Hago la PSD con Welch 
    freqs_reference, psd_reference = welch(reference_signal, fs_ref, nfft=8192, window=('kaiser', 0.7), nperseg=30)
    freqs_signal, psd_filtered = welch(kalman_filtered, fs_st, nfft=8192, window=('kaiser', 0.7), nperseg=30)
    freqs_signal1, psd_filtered1 = welch(kalman_filtered1, fs_st, nfft=8192, window=('kaiser', 0.7), nperseg=30)
    # Interpolar PSD de la referencia para alinearla con la PSD filtrada
    interp_psd_reference = interp1d(freqs_reference, psd_reference, bounds_error=False, fill_value="extrapolate")
    psd_reference_interp = interp_psd_reference(freqs_signal)
    #print("Valores únicos en psd_reference_interp:", np.unique(psd_reference_interp))

    # Graficar señal filtrada vs original
    plt.figure(figsize=(12, 5))
    plt.plot(signal, label="Señal original", alpha=0.5)
    plt.plot(kalman_filtered, label="Señal filtrada (ekf)", linewidth=2)
    plt.plot(kalman_filtered1, label="Señal filtrada (kalman)", linewidth=2)
    plt.legend()
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.title("Filtro de Kalman aplicado a una señal UWB")
    plt.show(block=False)
    
    
    # Graficar espectro
    plt.figure(figsize=(10, 5))
    plt.plot(freqs_signal, psd_filtered / max(psd_filtered), label="Espectro señal filtrada")
    plt.plot(freqs_reference, psd_reference / max(psd_reference), label="Espectro señal referencia", linestyle='dashed')
    plt.plot(freqs_signal1, psd_filtered1 / max(psd_filtered1), label="Espectro señal filtrada kalman original")
    plt.legend()
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Densidad espectral de potencia")
    plt.title("Comparación del espectro de la señal filtrada vs referencia")
    plt.show(block=False)
    
    return 0 #dominant_freq_filtered



# Simulación de una señal de referencia
dominant_freq = process_signal(s, referencia, fs_st, fs_ref, Q_value, R_value)

# CZT parameters
f1 = 0.01       # Start frequency
f2 = 1       # End frequency
m = 500        # Number of frequency points (resolution)

# Convert frequencies to angles on the unit circle
theta1 = 2 * np.pi * f1 / fs_st
theta2 = 2 * np.pi * f2 / fs_st

a = np.exp(1j * theta1)                          # Starting point on unit circle
w = np.exp(-1j * (theta2 - theta1) / m)          # Ratio between successive points


plt.show()

print(f"Frecuencia dominante de la señal filtrada: {dominant_freq} Hz")


