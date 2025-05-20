import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
import scipy.fftpack
from scipy.signal import CZT

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
    # Cantidad de puntos para la FFT con zero padding (por ejemplo, el siguiente número potencia de 2)
    N = len(signal)
    N_fft = 2**np.ceil(np.log2(N)).astype(int) * 4  # 4x padding (ajustable)

    # FFT con zero padding
    spectrum = np.fft.fft(signal, n=N_fft)
    magnitude = np.abs(spectrum[:N_fft // 2])
    frequencies = np.fft.fftfreq(N_fft, d=1/fs)[:N_fft // 2]

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

def circulant_multiply(c, x):
    # Compute the product y = Gx of a circulant matrix G and a vector x, where G is generated by its first column
    #     c = (c[0], c[1], ..., c[n-1]).
    if len(x) != len(c):
        raise Exception("should have len(x) equal to len(c), but instead len(x) = %d, len(c) = %d" % (len(x), len(c)))
    
    return scipy.fftpack.ifft( scipy.fftpack.fft(c) * scipy.fftpack.fft(x) )

def toeplitz_multiply_e(r, c, x):
    # Compute the product y = Tx of a Toeplitz matrix T and a vector x, where T is specified by its first row
    #     r = (r[0], r[1], r[2], ..., r[N-1])
    # and its first column
    #     c = (c[0], c[1], c[2], ..., c[M-1]),
    # where r[0] = c[0].
    N = len(r)
    M = len(c)
    
    if r[0] != c[0]:
        raise Exception("should have r[0] == c[0], but r[0] = %f and c[0] = %f" % (r[0], c[0]))
    if len(x) != len(r):
        raise Exception("should have len(x) equal to len(r), but instead len(x) = %d, len(r) = %d" % (len(x), len(r)))
    
    n = (2 ** np.ceil(np.log2(M+N-1))).astype(np.int64)
    
    # Form an array C by concatenating c, n - (M + N - 1) zeros, and the reverse of the last N-1 elements of r, ie.
    #     C = (c[0], c[1], ..., c[M-1], 0,..., 0, r[N-1], ..., r[2], r[1]).
    C = np.concatenate(( np.pad(c, (0, n - (M + N - 1)), 'constant'), np.flip(r[1:]) ))
    
    X = np.pad(x, (0, n-N), 'constant')
    Y = circulant_multiply(C, X)
    
    # The result is the first M elements of C * X.    
    return Y[:M]

def czt_spectrum(x, m, fs, f_start, f_end):
    """
    Computes the Chirp Z-transform of x, returning spectrum and frequency axis in Hz.
    
    Parameters:
        x       : input signal
        M       : number of frequency points
        fs      : sampling rate in Hz
        f_start : start frequency in Hz
        f_end   : end frequency in Hz
    
    Returns:
        X       : CZT of the signal (complex spectrum)
        freqs   : frequencies in Hz corresponding to X
    """
    # Normalized frequency range (cycles/sample)

    theta1 = 2 * np.pi * f_start / fs
    theta2 = 2 * np.pi * f_end / fs

    # CZT parameters
    a = np.exp(1j * theta1)                          # Starting point on unit circle
    w = np.exp(-1j * (theta2 - theta1) / m)          # Ratio between successive points

    # Apply CZT
    czt_transform = CZT(n=len(x), m=m, w=w, a=a)
    X_czt = czt_transform(x)

    # Frequency axis
    frequencies = np.linspace(f_start, f_end, m)

    return frequencies, X_czt
    