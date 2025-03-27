import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Importar tus funciones definidas en módulos (asegúrate de organizarlas en archivos .py)
from src.preprocess import remove_clutter_MA 
from src.utils import frecuencias_dominantes, slow_time_max_var
from src.analysis import fft_spectrum, music_spectrum

# PARÁMETROS FIJOS
fs_st = 18.75 #Hz
fs_ref = 1/0.325 #Hz

# --- 1. Leer datos desde archivos .mat ---
def cargar_medicion(ruta, key):
    data = scipy.io.loadmat(ruta)
    return data[key]  # Ajusta esto según la estructura del archivo

ruta_raw = "data/Mendeley Data/Raw Radar Data/Bandwidth1/DeltaR=8cm_Angle=30_Band1_Facedown_Trial1.mat"
ruta_ref = "data/Mendeley Data/Reference Data/Bandwidth1/Ref_DeltaR=8cm_Angle=30_Band1_Facedown_Trial1.mat"

medicion = cargar_medicion(ruta_raw, "bScan")
referencia = cargar_medicion(ruta_ref, "Ref")[0, :]

# --- 2. Procesamiento (aquí puedes probar distintos métodos) ---
medicion_filtrada = remove_clutter_MA(medicion, k=30)
#medicion_filtrada = medicion

# medicion_filtrada tiene la matriz, necesito analizar la señal de mayor varianza en slow time
max_var_idx, var = slow_time_max_var(medicion_filtrada)

# Defino mi señal a realizarle el análisis espectral
s = medicion_filtrada[:, max_var_idx]
s = s[200:]


# --- 3. Análisis espectral ---
frecuencias_s, espectro_s = fft_spectrum(s, fs=fs_st, window='hanning', plot=0)  # Ajusta la fs

frecuencias_ref, espectro_ref = fft_spectrum(referencia, fs=fs_ref, plot=0)  # Ajusta la fs

# Identificar frecuencia dominante
f_dom = frecuencias_dominantes(frecuencias_s, espectro_s, rango=(0.1, 0.4), n=1)

# --- 4. Comparación con referencia ---
f_ref = frecuencias_dominantes(frecuencias_ref, espectro_ref, rango=(0.1, 0.4), n=1)

# Calcular error cuadrático medio (ECM)
ecm = mean_squared_error(f_ref, f_dom)
print(f"Error cuadrático medio entre referencia y medición: {ecm:.6f}")
print(f_ref)
print(f_dom)
