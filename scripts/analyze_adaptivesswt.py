import numpy as np
import matplotlib.pyplot as plt
from src.utils import  slow_time_max_var
import scipy.io

from scipy.signal import detrend
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from adaptivesswt import adaptive_sswt
from adaptivesswt.configuration import Configuration
from adaptivesswt.sswt import sswt
from typing import Callable, Optional, Tuple
from scipy import signal as sp
from adaptivesswt.sswt import reconstruct


# 1. Carga de señal
def cargar_medicion(ruta, key):
    data = scipy.io.loadmat(ruta)
    return data[key] 

def preprocesar_senal(senal):
    # Remueve tendencias lineales
    processed_signal = detrend(senal, 1)
    return processed_signal


def graficar_espectrograma(senal, fs):
    f, t, Sxx = spectrogram(senal, fs=fs, nperseg=256, noverlap=128)
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading='gouraud')
    plt.title("Espectrograma")
    plt.ylabel('Frecuencia [Hz]')
    plt.xlabel('Tiempo [s]')
    plt.colorbar(label='Potencia [dB]')
    plt.tight_layout()
    plt.show(block=False)

def graficar_espectrograma_asst(
    signal: np.ndarray,
    #t: np.ndarray,
    #f: Tuple,
    #transform: Callable,
    config: Configuration,
    method_fig: Optional[plt.figure] = None,
    if_fig: Optional[plt.figure] = None,
    **kwargs,):
    """
    Grafica el espectrograma adaptativo de una señal utilizando adaptivesswt.

    Parámetros:
        signal: np.ndarray
            Señal en el dominio temporal.
        fs: float
            Frecuencia de muestreo.
        gamma: float
            Parámetro de rigidez para la SST adaptativa.
        Q: int
            Número de componentes estimados (Q=1 para una sola, Q=2 para dos componentes, etc.)
    """
    sst, cwt, freqs, _, _ = sswt(signal, **config.asdict())
    asst, aFreqs, _, _ = adaptive_sswt(
    signal,
    kwargs['bMaxIters'],
    kwargs['method'],
    kwargs['threshold'],
    kwargs['itl'],
    **config.asdict(),
)

    time = np.arange(signal.shape[0]) / fs_st

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(time, aFreqs, np.abs(sst), cmap='plasma', shading='gouraud')
    plt.ylabel("Frecuencia [Hz]")
    plt.xlabel("Tiempo [s]")
    plt.title("Espectrograma")
    plt.colorbar(label="Magnitud")
    plt.tight_layout()
    plt.show()

def graficar_reconstruccion(
signal: np.ndarray,
config: Configuration,
method_fig: Optional[plt.figure] = None,
if_fig: Optional[plt.figure] = None,
**kwargs,):
    
    sst, cwt, freqs, _, _ = sswt(signal, **config.asdict())
    asst, aFreqs, _, _ = sswt(
    signal,
    **config.asdict(),
    )

    signalRespSynth = reconstruct(asst, config.c_psi, freqs)

    plt.figure()
    plt.plot(signalRespSynth)
    plt.plot(-(signal - np.mean(signal)))
    plt.show(block=False)
    plt.show()

    
        


if __name__ == "__main__":
    ruta_raw = "data/Mendeley Data/Raw Radar Data/Bandwidth1/DeltaR=8cm_Angle=30_Band1_Lateral_Trial1.mat"
    ruta_ref = "data/Mendeley Data/Reference Data/Bandwidth1/Ref_DeltaR=8cm_Angle=30_Band1_Lateral_Trial1.mat"

    medicion = cargar_medicion(ruta_raw, "bScan")
    referencia = cargar_medicion(ruta_ref, "Ref")[0, :]

    # PARÁMETROS FIJOS
    fs_st = 18.75 #Hz
    fs_ref = 1/0.325 #Hz

    f_resp = 6 # Hz -> esta es para hacer un downsampling

    senal = preprocesar_senal(medicion)
    senal = senal[:, slow_time_max_var(senal)[0]] # acá tomo la señal en slow time de la matriz

    respDecRate = int(fs_st/f_resp)
    respFs = fs_st / respDecRate
    respDecSignal = sp.decimate(senal, respDecRate,ftype='fir')
    #graficar_espectrograma(senal, fs_st)
    #graficar_espectrograma(referencia, fs_ref)
    config = Configuration(
    min_freq=0.01,
    max_freq=1,
    num_freqs=256,
    ts=1/fs_st,
    wcf=1,
    wbw=2,
    wavelet_bounds=(-8, 8),
    threshold=1/50,
    )
    threshold = config.threshold * 5 # Threshold for ASST
    bMaxIters = 128  # Max iterations in b
    method = 'proportional'
    itl = 'ITL'
    graficar_espectrograma_asst(senal, config=config, bMaxIters=bMaxIters, method=method, threshold=threshold, itl=itl)
    graficar_reconstruccion(respDecSignal, config=config, bMaxIters=bMaxIters, method=method, threshold=threshold, itl=itl)


