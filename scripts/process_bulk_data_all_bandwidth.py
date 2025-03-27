import os
import scipy.io
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.metrics import mean_squared_error

# Importar funciones de procesamiento
from src.preprocess import remove_clutter_MA 
from src.utils import frecuencias_dominantes, slow_time_max_var, cargar_medicion
from src.analysis import fft_spectrum

# Parámetros fijos
fs_st = 18.75  # Frecuencia de muestreo slow time
fs_ref = 1 / 0.325  # Frecuencia de muestreo de referencia

# Directorios base
data_base = "data/Mendeley Data/"
bandas = ["Bandwidth1", "Bandwidth2", "Bandwidth3"]

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

    # Obtener archivos de datos crudos
    archivos_raw = [f for f in os.listdir(raw_dir) if f.endswith(".mat")]

    for archivo_raw in archivos_raw:
        ruta_raw = os.path.join(raw_dir, archivo_raw)
        
        # Buscar archivo de referencia
        archivo_ref = archivo_raw.replace("DeltaR", "Ref_DeltaR") 
        ruta_ref = os.path.join(ref_dir, archivo_ref)
        
        if not os.path.exists(ruta_ref):
            print(f"Archivo de referencia no encontrado para: {archivo_raw}")
            continue

        # Cargar datos
        medicion = cargar_medicion(ruta_raw, "bScan")
        referencia = cargar_medicion(ruta_ref, "Ref")[0, :]

        # Procesamiento
        medicion_filtrada = remove_clutter_MA(medicion, k=30)
        max_var_idx, _ = slow_time_max_var(medicion_filtrada)
        s = medicion_filtrada[:, max_var_idx][200:]  # Ajusta si es necesario

        # Análisis espectral
        frecuencias_s, espectro_s = fft_spectrum(s, fs=fs_st, window='hanning', plot=0)
        frecuencias_ref, espectro_ref = fft_spectrum(referencia, fs=fs_ref, plot=0)

        # Frecuencias dominantes
        f_dom = frecuencias_dominantes(frecuencias_s, espectro_s, rango=(0.1, 0.4), n=1)
        f_ref = frecuencias_dominantes(frecuencias_ref, espectro_ref, rango=(0.1, 0.4), n=1)

        # Calcular ECM
        #ecm = mean_squared_error(f_ref, f_dom) / f_ref[0]
        ecm = np.abs(f_ref[0] - f_dom[0])

        # Extraer metadatos del archivo
        delta_r, angulo, banda, posicion, trial = extraer_info_archivo(archivo_raw)

        # Guardar resultado
        resultados.append([archivo_raw, banda, delta_r, angulo, posicion, trial, ecm, f_ref[0], f_dom[0]])

        print(f"Procesado: {archivo_raw} - MSE: {ecm:.6f}")

# Convertir a DataFrame
df_resultados = pd.DataFrame(resultados, columns=["Archivo", "Banda", "DeltaR", "Ángulo", "Posición", "Trial", "MSE", "F_ref", "F_dom"])

# Guardar en CSV
df_resultados.to_csv("resultados_abs_err.csv", index=False)

# Mostrar las combinaciones con menor error
df_mean_mse = df_resultados.groupby(["Banda", "DeltaR", "Ángulo", "Posición"]).mean(numeric_only=True).reset_index()
print(df_mean_mse.sort_values(by="MSE").head(5))

# Gráficos
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_resultados, x="Banda", y="MSE")
plt.title("MSE por Banda")
plt.show(block=False)

df_resultados["Ángulo_jitter"] = df_resultados["Ángulo"] + np.random.uniform(-5, 5, size=len(df_resultados))


plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_resultados, x="Ángulo_jitter", y="MSE", hue="DeltaR", style="DeltaR", s=100)
#sns.scatterplot(data=df_resultados, x="Ángulo", y="MSE", hue="Banda", style="Banda", s=100)
plt.title("MSE vs Ángulo con diferenciación por Banda")
plt.xlabel("Ángulo")
plt.grid(True)
plt.show(block=False)

pivot_table = df_resultados.pivot_table(values="MSE", index="DeltaR", columns="Banda")
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, cmap="coolwarm")
plt.title("Heatmap de MSE según DeltaR y Banda")
plt.show(block=False)

plt.show()