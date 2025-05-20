import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from src.filters import kalman_filter
from src.preprocess import remove_clutter_MA
from src.utils import frecuencias_dominantes, slow_time_max_var
import scipy.io
from scipy.interpolate import interp1d
from scipy.signal import detrend

# PARÁMETROS FIJOS
fs_st = 18.75 #Hz
fs_ref = 1/0.325 #Hz

# --- 1. Leer datos desde archivos .mat ---
def cargar_medicion(ruta, key):
    data = scipy.io.loadmat(ruta)
    return data[key]  # Ajusta esto según la estructura del archivo

ruta_raw = "data/Mendeley Data/Raw Radar Data/Bandwidth1/DeltaR=8cm_Angle=30_Band1_Lateral_Trial1.mat"
ruta_ref = "data/Mendeley Data/Reference Data/Bandwidth1/Ref_DeltaR=8cm_Angle=30_Band1_Lateral_Trial1.mat"
ruta_calib = "data/Mendeley Data/Calibration Data/Angle=30/Calibration_Band1_DeltaR=8cm.mat"

medicion = cargar_medicion(ruta_raw, "bScan")
referencia = cargar_medicion(ruta_ref, "Ref")[0, :]
calibracion = cargar_medicion(ruta_calib, "bScan")

# --- 2. Procesamiento (aquí puedes probar distintos métodos) ---
medicion_filtrada_ma = remove_clutter_MA(medicion, k=30, R_ref=calibracion)
#medicion_filtrada = medicion

# medicion_filtrada tiene la matriz, necesito analizar la señal de mayor varianza en slow time
max_var_idx, var = slow_time_max_var(medicion_filtrada_ma)

medicion_filtrada_detrend = detrend(medicion, 1)

# Defino mi señal a realizarle el análisis espectral
s = medicion_filtrada_ma[:, max_var_idx]
s_detrend = medicion_filtrada_detrend[:, max_var_idx]
s_detrend = s_detrend - np.mean(s_detrend)
clutter_ma = medicion[:, max_var_idx] - s
clutter_detrend = medicion[:, max_var_idx] - s_detrend

#s = s[200:]

# Calcular la potencia de la señal y del ruido
P_signal = np.mean((medicion[:, max_var_idx])**2)  # Potencia de la señal
P_noise = np.mean((calibracion[:, max_var_idx])**2)   # Potencia del ruido

# Calcular SNR en dB
SNR_dB = 10 * np.log10(P_signal / P_noise)

print(f"SNR: {SNR_dB:.2f} dB")


#plt.plot(medicion[:, max_var_idx], label="Medición")
plt.figure(figsize=(8, 6))
plt.plot(calibracion[:, max_var_idx], label="Calibración")
plt.plot(clutter_ma, label="clutter MA")
plt.plot(clutter_detrend, label="clutter detrend")
plt.title("Estimación de clutters luego del procesamiento")

plt.legend()  # Agrega la leyenda automáticamente
plt.show(block=False)

plt.figure(figsize=(8, 6))
plt.plot(s, label="clutter MA")
plt.plot(s_detrend, label="detrend")
plt.title("Señales estimadas")

plt.legend()  # Agrega la leyenda automáticamente
plt.show(block=False)

plt.show()

'''

def calculate_dominant_frequency(signal, fs, nperseg):
    """Calcula la frecuencia dominante de la señal usando FFT."""
    freqs, psd = welch(signal, fs, nperseg=nperseg)
    dominant_freq = frecuencias_dominantes(freqs, psd, rango=(0.1, 0.4), n=1)
    return freqs, psd, dominant_freq

def process_signal(signal, reference_signal, fs_signal, fs_reference, Q_value, R_value):
    """Aplica filtros y analiza la señal."""
    kalman_filtered = kalman_filter(signal, fs_signal, Q_value, R_value)
    #kalman_filtered = signal
    
     # Asegurar el mismo nperseg para ambas señales
    min_length = min(len(kalman_filtered), len(reference_signal))
    nperseg = min(1024, min_length)
    
    # Cálculo de espectro
    freqs_signal, psd_filtered, dominant_freq_filtered = calculate_dominant_frequency(kalman_filtered, fs_signal, nperseg)
    freqs_reference, psd_reference, _ = calculate_dominant_frequency(reference_signal, fs_reference, nperseg)

    # Interpolar PSD de la referencia para alinearla con la PSD filtrada
    interp_psd_reference = interp1d(freqs_reference, psd_reference, bounds_error=False, fill_value="extrapolate")
    psd_reference_interp = interp_psd_reference(freqs_signal)
    print("Valores únicos en psd_reference_interp:", np.unique(psd_reference_interp))

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
    plt.plot(freqs_signal, psd_filtered/1e6, label="Espectro señal filtrada")
    plt.plot(freqs_signal, psd_reference_interp, label="Espectro señal referencia", linestyle='dashed')
    plt.legend()
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Densidad espectral de potencia")
    plt.title("Comparación del espectro de la señal filtrada vs referencia")

    plt.show(block=False)

    plt.show()
    
    return dominant_freq_filtered

# Uso del script
Q_value = 0.004
R_value = 0.15
window_size = 5

# Simulación de una señal de referencia
dominant_freq = process_signal(s, referencia, fs_st, fs_ref, Q_value, R_value)
print(f"Frecuencia dominante de la señal filtrada: {dominant_freq} Hz")


'''