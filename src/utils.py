import numpy as np
import scipy

def frecuencias_dominantes(frecuencias, magnitud, rango=(0.1, 0.4), n=1):

    """
    Encuentra las n frecuencias dominantes dentro de un rango específico.

    Parámetros:
    - frecuencias: array de frecuencias.
    - magnitud: array de magnitudes correspondiente a las frecuencias.
    - rango: tupla (min, max) con los límites del rango de interés.
    - n: cantidad de frecuencias dominantes a obtener.

    Retorna:
    - Lista con las n frecuencias dominantes en orden descendente de magnitud.
    """
    # Filtrar frecuencias dentro del rango
    mask = (frecuencias >= rango[0]) & (frecuencias <= rango[1])
    frecuencias_filtradas = frecuencias[mask]
    magnitud_filtrada = magnitud[mask]

    if len(frecuencias_filtradas) == 0:
        raise ValueError("No hay frecuencias en el rango especificado.")

    # Obtener los índices de las n magnitudes más grandes
    idx_top_n = np.argsort(magnitud_filtrada)[-n:][::-1]  # Orden descendente

    return frecuencias_filtradas[idx_top_n]

def slow_time_max_var(R):
    """
    Encuentra la columna de una matriz con la mayor varianza.

    Parámetros:
    - R: np.array -> Matriz de entrada (N x M).

    Retorna:
    - indice_max_var: int -> Índice de la columna con mayor varianza.
    - varianzas: np.array -> Vector de varianzas de cada columna.
    """
    varianzas = np.var(R, axis=0)  # Calcula la varianza de cada columna
    indice_max_var = np.argmax(varianzas)  # Encuentra el índice de la máxima varianza
    
    return indice_max_var, varianzas
# Función para cargar mediciones desde un archivo .mat
def cargar_medicion(ruta, key):
    data = scipy.io.loadmat(ruta)
    #print(data.keys())
    return data[key]  # Ajusta según la estructura del archivo
