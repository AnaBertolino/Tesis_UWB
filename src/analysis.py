import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd

def music_spectrum(signal, fs, num_sources, M=100, plot=True):
    """
    Aplica el algoritmo MUSIC para estimar las frecuencias dominantes en una señal.

    Parámetros:
    - signal: np.array -> Señal de entrada en dominio temporal.
    - fs: float -> Frecuencia de muestreo en Hz.
    - num_sources: int -> Número de fuentes de señal (frecuencias a detectar).
    - M: int -> Tamaño de la matriz de Hankel (por defecto 100).
    - plot: bool -> Si True, grafica el espectro MUSIC.

    Retorna:
    - frequencies: np.array -> Vector de frecuencias analizadas.
    - P_MUSIC: np.array -> Espectro MUSIC normalizado.
    """

    # Construcción de la matriz de Hankel
    X = np.array([signal[i:i+M] for i in range(len(signal)-M)])

    # Matriz de autocorrelación
    R = np.dot(X.T, X) / X.shape[0]

    # Descomposición en valores propios (SVD)
    U, S, Vh = svd(R)

    # Subespacio de ruido
    noise_subspace = U[:, num_sources:]   # Autovectores asociados al ruido

    # Rango de frecuencias a evaluar
    frequencies = np.linspace(0, fs/2, 500)  # Hasta Nyquist
    P_MUSIC = []

    # Cálculo del espectro MUSIC
    for f in frequencies:
        v = np.exp(1j * 2 * np.pi * f * np.arange(M) / fs)  # Vector de prueba
        psd = 1 / np.sum(np.abs(np.dot(noise_subspace.T, v))**2)  # Pseudo-espectro
        P_MUSIC.append(psd)

    P_MUSIC = np.array(P_MUSIC)
    P_MUSIC /= np.max(P_MUSIC)  # Normalización

    # Gráfico opcional
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(frequencies, 10*np.log10(P_MUSIC), label="MUSIC Spectrum")
        plt.xlabel("Frecuencia (Hz)")
        plt.ylabel("Potencia (dB)")
        plt.title("Espectro MUSIC")
        plt.legend()
        plt.grid()
        plt.show()

    return frequencies, P_MUSIC

def esprit_spectrum(signal, fs, num_sources, M=100, plot=True):
    """
    Aplica el algoritmo ESPRIT para estimar las frecuencias dominantes en una señal.

    Parámetros:
    - signal: np.array -> Señal de entrada en dominio temporal.
    - fs: float -> Frecuencia de muestreo en Hz.
    - num_sources: int -> Número de fuentes de señal (frecuencias a detectar).
    - M: int -> Tamaño de la matriz de Hankel (por defecto 100).
    - plot: bool -> Si True, grafica el espectro ESPRIT.

    Retorna:
    - estimated_frequencies: np.array -> Frecuencias estimadas en Hz.
    """

    # Construcción de la matriz de Hankel
    X = np.array([signal[i:i+M] for i in range(len(signal)-M)])

    # Matriz de autocorrelación
    R = np.dot(X.T, X) / X.shape[0]

    # Descomposición en valores propios (SVD)
    U, S, Vh = svd(R)

    # Subespacio de señal
    signal_subspace = U[:, :num_sources]

    # Definir las matrices desplazadas
    U1 = signal_subspace[:-1, :]
    U2 = signal_subspace[1:, :]

    # Resolver el sistema de ecuaciones U1 * Φ = U2
    Phi = np.linalg.pinv(U1) @ U2  # Estimación de la matriz de rotación

    # Eigenvalores de Phi
    eigvals = np.linalg.eigvals(Phi)

    # Estimación de frecuencias
    estimated_frequencies = np.angle(eigvals) * fs / (2 * np.pi)

    # Construcción del espectro
    freq_range = np.linspace(0, fs/2, 500)
    P_ESPRIT = np.zeros_like(freq_range)

    for f in estimated_frequencies:
        closest_idx = np.argmin(np.abs(freq_range - np.abs(f)))
        P_ESPRIT[closest_idx] += 1  # Reforzamos los picos en las frecuencias estimadas

    # Normalización del espectro
    P_ESPRIT /= np.max(P_ESPRIT)

    # Gráfico opcional
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(freq_range, 10 * np.log10(P_ESPRIT + 1e-10), label="ESPRIT Spectrum")
        plt.xlabel("Frecuencia (Hz)")
        plt.ylabel("Potencia (dB)")
        plt.title("Espectro ESPRIT")
        plt.legend()
        plt.grid()
        plt.show()

    return estimated_frequencies

def fft_spectrum(signal, fs, window=None, plot=True):
    """
    Calcula la FFT de una señal y devuelve su espectro de frecuencias.

    Parámetros:
    - signal: np.array -> Señal de entrada en el dominio temporal.
    - fs: float -> Frecuencia de muestreo en Hz.
    - window: str o None -> Tipo de ventana a aplicar (ej: "hann", "hamming", None para no aplicar).
    - plot: bool -> Si True, grafica el espectro de la FFT.

    Retorna:
    - frequencies: np.array -> Vector de frecuencias analizadas.
    - magnitude: np.array -> Espectro de amplitud normalizado.
    """

    # Remover la media para eliminar la componente DC
    signal = signal - np.mean(signal)

    # Aplicar ventana si se especifica
    if window is not None:
        try:
            win_func = getattr(np, window)(len(signal))
            signal = signal * win_func
        except AttributeError:
            raise ValueError(f"Ventana '{window}' no reconocida. Usa 'hann', 'hamming', etc.")

    # Cálculo de la FFT
    N = len(signal)
    spectrum = np.fft.fft(signal)
    magnitude = np.abs(spectrum[:N // 2])  # Magnitud de la FFT (mitad positiva)
    frequencies = np.fft.fftfreq(N, d=1/fs)[:N // 2]  # Escala de frecuencias positivas

    # Normalizar espectro
    magnitude /= np.max(magnitude)

    # Graficar espectro si plot=True
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(frequencies, 10 * magnitude, label="FFT Spectrum")
        plt.xlabel("Frecuencia (Hz)")
        plt.ylabel("Amplitud (dB)")
        plt.title("Espectro de Frecuencia")
        plt.legend()
        plt.grid()
        plt.show()

    return frequencies, magnitude