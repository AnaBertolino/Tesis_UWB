import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, detrend
from src.filters import kalman_filter, bandpass_filter, extended_kalman_filter
from src.preprocess import remove_clutter_MA
from src.utils import frecuencias_dominantes, slow_time_max_var
import scipy.io
from scipy.interpolate import interp1d
from src.analysis import fft_spectrum, czt_spectrum



# PARÁMETROS FIJOS
fs_st = 18.75 #Hz
fs_ref = 1/0.325 #Hz

# Uso del script
Q_value = 0.001
R_value = 1
window_size = 5

# --- 1. Leer datos desde archivos .mat ---
def cargar_medicion(ruta, key):
    data = scipy.io.loadmat(ruta)
    return data[key]  # Ajusta esto según la estructura del archivo

ruta_raw = "data/Mendeley Data/Raw Radar Data/Bandwidth1/DeltaR=10cm_Angle=0_Band1_Lateral_Trial7.mat"
ruta_ref = "data/Mendeley Data/Reference Data/Bandwidth1/Ref_DeltaR=10cm_Angle=0_Band1_Lateral_Trial7.mat"
ruta_calib = "data/Mendeley Data/Calibration Data/Angle=30/Calibration_Band1_DeltaR=8cm.mat"

medicion = cargar_medicion(ruta_raw, "bScan")
referencia = cargar_medicion(ruta_ref, "Ref")[0, :]
referencia = referencia - np.mean(referencia)
calibracion = cargar_medicion(ruta_calib, "bScan")

referencia = kalman_filter(referencia, fs_ref, Q_value, R_value)
referencia = detrend(referencia)
referencia = bandpass_filter(referencia, fs_ref, 0.2, 1.5)


plt.figure()
plt.plot(referencia)
plt.show(block=False)

# --- 2. Procesamiento (aquí puedes probar distintos métodos) ---
medicion_filtrada = remove_clutter_MA(medicion, k=30, R_ref=calibracion)
#medicion_filtrada = medicion

# medicion_filtrada tiene la matriz, necesito analizar la señal de mayor varianza en slow time
max_var_idx, var = slow_time_max_var(medicion_filtrada)

# Defino mi señal a realizarle el análisis espectral
s = medicion_filtrada[:, max_var_idx]
s = s[200:]

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
    freqs, psd = welch(signal, fs, nfft=8192, window=('kaiser', 0.7), nperseg=nperseg)
    freqs, psd = fft_spectrum(signal, fs, plot=False)
    freqs_s_czt, espectro_s_czt = czt_spectrum(signal, 500, fs, 0.01, 1)

    chirp_z = CZT(n=len(signal))

    espectro = chirp_z(signal)

    dominant_freq = frecuencias_dominantes(freqs_s_czt, espectro_s_czt, rango=(0.1, 0.4), n=1)
    return freqs, psd, dominant_freq, freqs_s_czt, espectro_s_czt

def process_signal(signal, reference_signal, fs_signal, fs_reference, Q_value, R_value):
    """Aplica filtros y analiza la señal."""
    kalman_filtered = kalman_filter(signal, fs_signal, Q_value, R_value)
    ext_kalman_filtered_out, ext_kalman_filtered_states = extended_kalman_filter(signal, 0.9999, plot=False)
    #kalman_filtered = signal

    kalman_filtered = ext_kalman_filtered_states[:, 0]
    #kalman_filtered = ext_kalman_filtered_out
    
     # Asegurar el mismo nperseg para ambas señales
    min_length = min(len(kalman_filtered), len(reference_signal))
    nperseg = min(1024, min_length)
    
    # Cálculo de espectro
    freqs_signal, psd_filtered, dominant_freq_filtered, freqs_czt, espectro_czt = calculate_dominant_frequency(kalman_filtered, fs_signal, nperseg)
    freqs_reference, psd_reference, _, freqs_czt_ref, espectro_czt_ref = calculate_dominant_frequency(reference_signal, fs_reference, nperseg)
    freqs_reference, psd_reference = fft_spectrum(reference_signal, fs_ref, plot=False)
    # Interpolar PSD de la referencia para alinearla con la PSD filtrada
    interp_psd_reference = interp1d(freqs_reference, psd_reference, bounds_error=False, fill_value="extrapolate")
    psd_reference_interp = interp_psd_reference(freqs_signal)
    #print("Valores únicos en psd_reference_interp:", np.unique(psd_reference_interp))

    # Graficar señal filtrada vs original
    plt.figure(figsize=(12, 5))
    plt.plot(signal, label="Señal original", alpha=0.5)
    plt.plot(kalman_filtered, label="Señal filtrada (Kalman)", linewidth=2)
    plt.legend()
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.title("Filtro de Kalman aplicado a una señal UWB")
    plt.show(block=False)
    
    # Graficar espectro
    plt.figure(figsize=(10, 5))
    plt.plot(freqs_signal, 20*np.log10(psd_filtered / max(psd_filtered)), label="Espectro señal filtrada")
    plt.plot(freqs_reference, 20*np.log10(psd_reference), label="Espectro señal referencia", linestyle='dashed')
    plt.legend()
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Densidad espectral de potencia")
    plt.title("Comparación del espectro de la señal filtrada vs referencia")
    plt.show(block=False)
    
    return dominant_freq_filtered

