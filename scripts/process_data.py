import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Importar tus funciones definidas en módulos (asegúrate de organizarlas en archivos .py)
from src.preprocess import remove_clutter_MA 
from src.utils import frecuencias_dominantes, slow_time_max_var
from src.analysis import fft_spectrum, music_spectrum
from scipy.signal import detrend
from src.filters import lowpass_filter, notch_filter, kalman_filter, bandpass_filter
from scipy.signal import spectrogram

def graficar_espectrograma(senal, fs):
    f, t, Sxx = spectrogram(senal, fs=fs, nperseg=256, noverlap=128)
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.title("Espectrograma")
    plt.ylabel('Frecuencia [Hz]')
    plt.xlabel('Tiempo [s]')
    plt.colorbar(label='Potencia [dB]')
    plt.tight_layout()
    plt.show(block=False)
    plt.show()

# PARÁMETROS FIJOS
fs_st = 18.75 #Hz
fs_ref = 1/0.325 #Hz

Q_value = 0.00163
R_value = 0.15

# --- 1. Leer datos desde archivos .mat ---
def cargar_medicion(ruta, key):
    data = scipy.io.loadmat(ruta)
    return data[key]  # Ajusta esto según la estructura del archivo

ruta_raw = "data/Mendeley Data/Raw Radar Data/Bandwidth2/DeltaR=10cm_Angle=0_Band2_Lateral_Trial6.mat"
ruta_ref = "data/Mendeley Data/Reference Data/Bandwidth2/Ref_DeltaR=10cm_Angle=0_Band2_Lateral_Trial6.mat"

medicion = cargar_medicion(ruta_raw, "bScan")
referencia = cargar_medicion(ruta_ref, "Ref")[0, 50:]

ref_kalman = kalman_filter(referencia, fs_ref, Q_value=Q_value, R_value=R_value)


# --- 2. Procesamiento (aquí puedes probar distintos métodos) ---
medicion_filtrada = remove_clutter_MA(medicion, k=30)
medicion_filtrada = detrend(medicion, 1)
#medicion_filtrada = medicion

# medicion_filtrada tiene la matriz, necesito analizar la señal de mayor varianza en slow time
max_var_idx, var = slow_time_max_var(medicion_filtrada)

# Defino mi señal a realizarle el análisis espectral
s = medicion_filtrada[:, max_var_idx]
#s = s[200:]
#s = lowpass_filter(s, 1.5, fs_st)
s = bandpass_filter(s, fs_st, 0.1, 1.5)

s = kalman_filter(s, fs_st, Q_value=Q_value, R_value=R_value)

t_s = np.arange(len(s)) / fs_st
t_ref = np.arange(len(referencia)) / fs_ref
t_ref += 1.92

#graficar_espectrograma(s, fs_st)
#graficar_espectrograma(medicion[:, max_var_idx], fs_st)

plt.figure()
plt.plot(t_s, medicion[:, max_var_idx] - np.mean(medicion[:, max_var_idx]))
plt.plot(t_s[12:] - 12/fs_st, s[12:] - np.mean(s))
plt.plot(t_ref, (ref_kalman - np.mean(ref_kalman)) * 50)
plt.show(block=False)
plt.show()

# --- 3. Análisis espectral ---
frecuencias_s, espectro_s = fft_spectrum(s, fs=fs_st, window='hanning', plot=0)  # Ajusta la fs

frecuencias_ref, espectro_ref = fft_spectrum(ref_kalman, fs=fs_ref, plot=0)  # Ajusta la fs

plt.figure()
plt.plot(frecuencias_s, espectro_s)
plt.plot(frecuencias_ref, espectro_ref)
plt.show(block=False)
plt.show()

# Identificar frecuencia dominante
f_dom = frecuencias_dominantes(frecuencias_s, espectro_s, rango=(0.15, 0.4), n=1)

# --- 4. Comparación con referencia ---
f_ref = frecuencias_dominantes(frecuencias_ref, espectro_ref, rango=(0.15, 0.4), n=1)

# Calcular error cuadrático medio (ECM)
ecm = mean_squared_error(f_ref, f_dom)
print(f"Error cuadrático medio entre referencia y medición: {ecm:.6f}")
print(f_ref)
print(f_dom)
