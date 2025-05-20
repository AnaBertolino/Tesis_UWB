import numpy as np

def remove_clutter_MA(R, k=30, R_ref=None):
    """
    Aplica un filtro de media móvil para eliminar el clutter de una matriz de datos R.
    Si se proporciona una señal de referencia R_ref, se usa su promedio inicial para 
    mejorar la convergencia del filtro.

    Parámetros:
    - R: np.array -> Matriz de datos (N x M), donde N es el número de muestras y M el número de canales.
    - k: int -> Tamaño de la ventana de media móvil (por defecto 30).
    - R_ref: np.array (opcional) -> Matriz de referencia (N_ref x M) para inicializar el filtro.

    Retorna:
    - R_clean: np.array -> Matriz con el clutter eliminado.
    """
    N, M = R.shape
    b = np.zeros((N, M))       # Buffer de media móvil
    R_clean = np.zeros((N, M)) # Matriz de salida

    # Inicializar b[0, :] con el promedio de la referencia si está disponible
    if R_ref is not None:
        b[0, :] = np.mean(R_ref, axis=0) 

    for i in range(N - 1):
        b[i + 1, :] = (1 - (1/k)) * b[i, :] + R[i, :] / k
        R_clean[i + 1, :] = R[i + 1, :] - b[i, :]

    return R_clean

