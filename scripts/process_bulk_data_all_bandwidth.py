import os
import scipy.io
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.metrics import mean_squared_error
from scipy.signal import detrend, welch
import scipy.signal as sp

# Importar funciones de procesamiento
from src.preprocess import remove_clutter_MA 
from src.utils import frecuencias_dominantes, slow_time_max_var, cargar_medicion
from src.analysis import fft_spectrum, czt_spectrum
from src.filters import lowpass_filter, notch_filter, kalman_filter, bandpass_filter, extended_kalman_filter

# Definir métricas de error
def mse(y_true, y_pred):
    return mean_squared_error([y_true], [y_pred])

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error([y_true], [y_pred]))

def mae(y_true, y_pred):
    return np.abs(y_true - y_pred)

def mre(y_true, y_pred):
    return np.abs((y_true - y_pred) / y_true)

def acc(y_true, y_pred):
    return 1 - np.abs((y_true - y_pred) / y_true)  # Precisión relativa

METRICAS = {
    "MSE": mse,
    "RMSE": rmse,
    "MAE": mae,
    "MRE": mre,
    "ACC": acc
}

# Seleccionar la métrica a utilizar
METRICA = "MRE"  # Cambia esto según lo que necesites

# Parámetros fijos
fs_st = 18.75  # Frecuencia de muestreo slow time
fs_ref = 1 / 0.325  # Frecuencia de muestreo de referencia

# Parámetros para el filtro de Kalman
Q_value = 0.00163
R_value = 0.15

# Directorios base
data_base = "data/Mendeley Data/"
bandas = ["Bandwidth1", "Bandwidth2", "Bandwidth3"]

# Cargar señales de referencia de calibración
calibration_base = os.path.join(data_base, "Calibration Data")
calibration_ref = {}

for angle_folder in ["Angle=0", "Angle=30"]:
    angle = int(angle_folder.split("=")[1])  # Extraer el ángulo (0 o 30)
    path = os.path.join(calibration_base, angle_folder)

    for file in os.listdir(path):
        match = re.search(r"Calibration_Band(\d+)_DeltaR=(\d+)cm", file)
        if match:
            band, delta_r = match.groups()
            band = f"Bandwidth{band}"
            delta_r = float(delta_r)

            ruta_calibracion = os.path.join(path, file)
            calibration_ref[(band, delta_r, angle)] = cargar_medicion(ruta_calibracion, "bScan")


# Función para extraer información del nombre del archivo
def extraer_info_archivo(nombre):
    match = re.search(r"DeltaR=(\d+(\.\d+)?)cm_Angle=(\d+)_Band(\d+)_(\w+)_Trial(\d+)", nombre)
    if match:
        delta_r, _, angle, band, position, trial = match.groups()
        return float(delta_r), int(angle), f"Bandwidth{band}", position, int(trial)
    return None, None, None, None, None

# Lista para almacenar resultados
resultados = []

# Iterar sobre cada ancho de banda
for banda in bandas:
    raw_dir = os.path.join(data_base, f"Raw Radar Data/{banda}")
    ref_dir = os.path.join(data_base, f"Reference Data/{banda}")

    archivos_raw = [f for f in os.listdir(raw_dir) if f.endswith(".mat")]

    for archivo_raw in archivos_raw:
        ruta_raw = os.path.join(raw_dir, archivo_raw)
        archivo_ref = archivo_raw.replace("DeltaR", "Ref_DeltaR")
        ruta_ref = os.path.join(ref_dir, archivo_ref)

        if not os.path.exists(ruta_ref):
            print(f"Archivo de referencia no encontrado para: {archivo_raw}")
            continue

        # Cargar datos
        medicion = cargar_medicion(ruta_raw, "bScan")
        referencia = cargar_medicion(ruta_ref, "Ref")[0, :]
        referencia = detrend(referencia)
        referencia = bandpass_filter(referencia, fs_ref, 0.2, 1)
        #_, referencia = extended_kalman_filter(referencia, fs_ref, 0.5)
        #referencia = np.array(referencia[:, 0])[:, 0]
        #referencia = kalman_filter(referencia, fs_ref, Q_value, R_value)
        #referencia = referencia[50:]

        # Extraer metadatos del archivo
        delta_r, angulo, banda, posicion, trial = extraer_info_archivo(archivo_raw)

        # Obtener la señal de referencia correspondiente (si existe)
        R_ref = calibration_ref.get((banda, delta_r, angulo), None)

        # Procesamiento con la señal de referencia opcional

        #medicion_filtrada = remove_clutter_MA(medicion, k=30)
        medicion_filtrada = detrend(medicion, 1)
        # medicion_filtrada = remove_clutter_MA(medicion, k=30, R_ref=R_ref)

        max_var_idx, _ = slow_time_max_var(medicion_filtrada)
        #s = medicion_filtrada[:, max_var_idx][200:]
        s = medicion_filtrada[:, max_var_idx]
        s = s - np.mean(s)

        # Calculo lo que necesito para f_card antes de pasar la señal por el filtro pasabajos
        s_card = s
        
        #s = lowpass_filter(s, 1.5, fs_st)
        s = bandpass_filter(s, fs_st, 0.15, 1.5)
        #s = notch_filter(s, fs_st, [0.15], 5)

        ref_kalman = kalman_filter(referencia, fs_ref, Q_value=Q_value, R_value=R_value)
        #s_kalman = kalman_filter(s, fs_st, Q_value=Q_value, R_value=R_value)


        # Hago un downsampling a 6Hz
        f_downsample = 6
        downsampleDecRate = int(fs_st/f_downsample)
        fs_downsample = fs_st / downsampleDecRate
        downsampleDecSignal = sp.decimate(s, downsampleDecRate,ftype='fir')

        ext_kalman_filtered_out, ext_kalman_filtered_states = extended_kalman_filter(s, fs_st, 0.999)
        s_kalman = np.array(ext_kalman_filtered_states[:, 0])[:, 0]
        # Análisis espectral

        # FFT
        #frecuencias_s, espectro_s = fft_spectrum(s_kalman, fs=fs_st, window='hanning', plot=0)
        #frecuencias_s, espectro_s = fft_spectrum(downsampleDecSignal, fs=fs_downsample, window='hanning', plot=0)
        #frecuencias_ref, espectro_ref = fft_spectrum(ref_kalman, fs=fs_ref, plot=0)

        # Welch
        #freqs_s_welch, psd_s_welch = welch(s_kalman, fs_st, nfft=8192, window=('kaiser', 0.7))
        freqs_s_welch, psd_s_welch = welch(s_kalman, fs_st, nfft=8192, window=('kaiser', 0.7), nperseg=60)
        freqs_ref_welch, psd_ref_welch = welch(ref_kalman, fs_ref, nfft=8192, window=('kaiser', 0.7), nperseg=19)

        # CZT
        #freqs_s_czt, espectro_s_czt = czt_spectrum(s_kalman, 2048, fs_st, 0.05, 0.5)
        #freqs_ref_czt, espectro_ref_czt = czt_spectrum(ref_kalman, 2048, fs_ref, 0.05, 0.5)

        # Frecuencias dominantes
        #f_dom = frecuencias_dominantes(frecuencias_s, espectro_s, rango=(0.1, 0.4), n=1)
        #f_ref = frecuencias_dominantes(frecuencias_ref, espectro_ref, rango=(0.1, 0.4), n=1)

        f_dom = frecuencias_dominantes(freqs_s_welch, psd_s_welch, rango=(0.1, 0.4), n=1)
        f_ref = frecuencias_dominantes(freqs_ref_welch, psd_ref_welch, rango=(0.1, 0.4), n=1)

        # Calcular armónicos a eliminar (1x, 2x, 3x respiratoria)
        harmonics = [f_dom[0] * i for i in range(1, 4) if f_dom[0] * i < fs_st / 2]

        # Filtrado para frecuencia cardíaca
        s_card = notch_filter(s_card, fs=fs_st, freqs=harmonics, Q=30)
        s_card = lowpass_filter(s_card, 2.5, fs_st)

        # Espectro cardíaco
        freqs_card, spec_card = fft_spectrum(s_card, fs=fs_st, plot=0)
        f_card = frecuencias_dominantes(freqs_card, spec_card, rango=(0.8, 2.5), n=1)

        # Calcular error usando la métrica seleccionada
        error = METRICAS[METRICA](f_ref[0], f_dom[0])

        # Guardar resultado
        resultados.append([archivo_raw, banda, delta_r, angulo, posicion, trial, error, f_ref[0], f_dom[0], f_card[0]])


        #print(f"Procesado: {archivo_raw} - {METRICA}: {error:.6f}")

# Convertir a DataFrame
df_resultados = pd.DataFrame(resultados, columns=["Archivo", "Banda", "DeltaR", "Ángulo", "Posición", "Trial", METRICA, "F_ref", "F_dom", "F_card"])

# Guardar en CSV
df_resultados.to_csv(f"resultados_Kalman_{METRICA}.csv", index=False)

# Mostrar las combinaciones con menor error
df_mean_mse = df_resultados.groupby(["Banda", "DeltaR", "Ángulo", "Posición"]).mean(numeric_only=True).reset_index()
print(df_mean_mse.sort_values(by=METRICA).head(5))

# Gráficos
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_resultados, x="Banda", y=METRICA)
plt.title(f"{METRICA} por Banda")
plt.show(block=False)

df_resultados["Ángulo_jitter"] = df_resultados["Ángulo"] + np.random.uniform(-5, 5, size=len(df_resultados))

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_resultados, x="Ángulo_jitter", y=METRICA, hue="DeltaR", style="DeltaR", s=100)
#sns.scatterplot(data=df_resultados, x="Ángulo", y="MSE", hue="Banda", style="Banda", s=100)
plt.title(f"{METRICA} vs Ángulo con diferenciación por Banda")
plt.xlabel("Ángulo")
plt.grid(True)
plt.show(block=False)


pivot_table = df_resultados.pivot_table(values=METRICA, index="DeltaR", columns="Banda")
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, cmap="coolwarm")
plt.title(f"Heatmap de {METRICA} según DeltaR y Banda")
plt.show(block=False)

pivot_table2 = df_resultados.pivot_table(values=METRICA, index="DeltaR", columns="Banda", aggfunc="max")
# Then, merge those max values back to the original DataFrame to filter matching rows
rows_with_max = df_resultados[
    df_resultados.apply(
        lambda row: row[METRICA] == pivot_table2.loc[row["DeltaR"], row["Banda"]], axis=1
    )
]
print(rows_with_max)
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table2, annot=True, cmap="coolwarm")
plt.title(f"Heatmap de {METRICA} según DeltaR y Banda (máximo)")
plt.show(block=False)

# Ordenar por frecuencia respiratoria
df_ordenado = df_resultados.sort_values(by="F_dom").reset_index(drop=True)
df_filtrado = df_resultados[df_resultados["Posición"] != "Facedown"].copy()

#df_top50 = df_resultados.nsmallest(75, METRICA)
# Filtrar solo supine y lateral
#df_tendencia = df_top50[df_top50["Posición"].isin(["Supine", "Lateral", "Facedown"])]

df_tendencia = df_ordenado

plt.figure(figsize=(12, 6))

# Puntos de dispersión
sns.scatterplot(
    data=df_tendencia.sort_values(by="F_dom"),
    x="F_dom", y="F_card",
    hue="DeltaR", style="Posición", s=100
)

# Línea de tendencia para cada posición
for pos in ["Supine", "Lateral", "Facedown"]:
    subset = df_tendencia[df_tendencia["Posición"] == pos]
    sns.regplot(
        data=subset,
        x="F_dom", y="F_card",
        scatter=False,
        label=f"Tendencia {pos.capitalize()}"
    )

plt.xlabel("Frecuencia Respiratoria Estimada (Hz)")
plt.ylabel("Frecuencia Cardíaca Estimada (Hz)")
plt.title(f"Relación F_card vs F_resp - Top 75 mediciones con menor {METRICA}")
plt.grid(True)
plt.legend(title="DeltaR / Posición", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()


plt.show()

corr = df_resultados["F_dom"].corr(df_resultados["F_card"])
print(f"Correlación lineal entre F_dom y F_card: {corr:.2f}")
