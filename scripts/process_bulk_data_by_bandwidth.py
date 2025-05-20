import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error

# Importar tus funciones definidas en módulos
from src.preprocess import remove_clutter_MA 
from src.utils import frecuencias_dominantes, slow_time_max_var, cargar_medicion
from src.analysis import fft_spectrum

# Parámetros fijos
fs_st = 18.75  # Frecuencia de muestreo slow time
fs_ref = 1 / 0.325  # Frecuencia de muestreo de referencia

# Directorios de datos
raw_dir = "data/Mendeley Data/Raw Radar Data/Bandwidth1"
ref_dir = "data/Mendeley Data/Reference Data/Bandwidth1"


# Obtener lista de archivos .mat en la carpeta raw
archivos_raw = [f for f in os.listdir(raw_dir) if f.endswith(".mat")]

# Lista para almacenar resultados
resultados = []



# Iterar sobre cada archivo en raw_dir
for archivo_raw in archivos_raw:
    ruta_raw = os.path.join(raw_dir, archivo_raw)
    
    # Buscar el archivo correspondiente en Reference Data
    archivo_ref = archivo_raw.replace("DeltaR", "Ref_DeltaR")  # Ajusta si el nombre es diferente
    ruta_ref = os.path.join(ref_dir, archivo_ref)
    
    # Verificar si existe el archivo de referencia
    if not os.path.exists(ruta_ref):
        print(f"Archivo de referencia no encontrado para: {archivo_raw}")
        continue  # Saltar si no hay referencia


    # Cargar datos
    medicion = cargar_medicion(ruta_raw, "bScan")
    referencia = cargar_medicion(ruta_ref, "Ref")[0, :]
    

    # Procesamiento de la señal
    medicion_filtrada = remove_clutter_MA(medicion, k=30)

    # Obtener la columna de mayor varianza
    max_var_idx, _ = slow_time_max_var(medicion_filtrada)

    # Definir señal a analizar
    s = medicion_filtrada[:, max_var_idx]
    s = s[200:]  # Ajusta si es necesario

    # Análisis espectral
    frecuencias_s, espectro_s = fft_spectrum(s, fs=fs_st, window='hanning', plot=0)
    frecuencias_ref, espectro_ref = fft_spectrum(referencia, fs=fs_ref, plot=0)

    # Obtener frecuencia dominante
    f_dom = frecuencias_dominantes(frecuencias_s, espectro_s, rango=(0.1, 0.4), n=1)
    f_ref = frecuencias_dominantes(frecuencias_ref, espectro_ref, rango=(0.1, 0.4), n=1)

    # Calcular ECM
    #ecm = mean_squared_error(f_ref, f_dom)
    ecm = np.abs(f_ref[0] - f_dom[0])/f_ref[0]

    # Guardar resultado
    resultados.append([archivo_raw, ecm, f_ref[0], f_dom[0]])

    print(f"Procesado: {archivo_raw} - MSE: {ecm:.6f}")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

# Función para extraer información del nombre del archivo
def extraer_factores(nombre):
    match = re.search(r"DeltaR=(\d+(\.\d+)?)cm_Angle=(\d+)_Band1_(\w+)_Trial(\d+)", nombre)
    if match:
        delta_r, _, angle, position, trial = match.groups()
        return float(delta_r), int(angle), position, int(trial)
    return None, None, None, None

# Convertimos resultados en DataFrame
df_resultados = pd.DataFrame(resultados, columns=["Archivo", "MSE", "F_ref", "F_dom"])

# Extraemos los factores
df_resultados[["DeltaR", "Ángulo", "Posición", "Trial"]] = df_resultados["Archivo"].apply(lambda x: pd.Series(extraer_factores(x)))

# Guardar en CSV
df_resultados.to_csv("resultados_mse.csv", index=False)


# Guardar resultados en un archivo CSV

#df_resultados.to_csv("resultados_mse.csv", index=False)

#print("Procesamiento finalizado. Resultados guardados en 'resultados_mse.csv'.")

df_mean_mse = df_resultados.groupby(["DeltaR", "Ángulo", "Posición"]).mean(numeric_only=True).reset_index()
print(df_mean_mse.sort_values(by="MSE").head(5))  # Ver las 5 mejores combinaciones


plt.figure(figsize=(12, 6))
sns.boxplot(data=df_resultados, x="DeltaR", y="MSE")
plt.title("Valor absoluto del error relativo vs DeltaR")
plt.ylabel("Valor absoluto del error relativo")
plt.show(block=False) 

df_resultados["Ángulo_jitter"] = df_resultados["Ángulo"] + np.random.uniform(-5, 5, size=len(df_resultados))

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_resultados, x="Ángulo_jitter", y="MSE", hue="DeltaR", style="DeltaR", s=100)
plt.title("Valor absoluto del error relativo vs Ángulo (con jitter y diferenciación por DeltaR)")
plt.xlabel("Ángulo (con jitter)")
plt.ylabel("Valor absoluto del error relativo")
plt.grid(True)
plt.show(block=False) 


pivot_table = df_resultados.pivot_table(values="MSE", index="DeltaR", columns="Ángulo")
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, cmap="coolwarm")
plt.title("Heatmap de Valor absoluto del error relativo según DeltaR y Ángulo")
plt.ylabel("Valor absoluto del error relativo")
plt.show(block=False) 

plt.show()



