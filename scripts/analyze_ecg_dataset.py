from scipy.io import whosmat
import matplotlib.pyplot as plt
from src.utils import parse_oximeter_txt, cargar_medicion, slow_time_max_var
from scipy.signal import savgol_filter
from scipy.signal import spectrogram
from scipy.signal import detrend
import numpy as np
from src.preprocess import remove_clutter_MA
from scipy.io import loadmat
from scipy import signal as sp
from adaptivesswt.adaptivesswt import adaptive_sswt
from adaptivesswt.configuration import Configuration
from adaptivesswt.sswt import sswt
from adaptivesswt.utils.plot_utils import plot_batched_tf_repr, plot_tf_repr

def graficar_spectrograma(senal, fs):
    f, t, Sxx = spectrogram(senal, fs=fs, nperseg=256, noverlap=128)

    plt.figure(figsize=(10, 5))
    
    # Use meshgrid to create X, Y arrays
    T, F = np.meshgrid(t, f)  # X (time), Y (freq)

    # Now plot safely!
    plt.pcolormesh(T, F, 10 * np.log10(np.abs(Sxx) + 1e-12), shading='gouraud')
    plt.title("Espectrograma")
    plt.ylabel('Frecuencia [Hz]')
    plt.xlabel('Tiempo [s]')
    plt.colorbar(label='Potencia [dB]')
    plt.tight_layout()
    plt.show(block=False)
    plt.show()

mat_file_path = "data/dataset_shi/datasets/measurement_data_person1/PCG_front_radar_back/apnea_after_sport/DATASET_2016-12-20_14-44-53_Person 1.mat"

data = loadmat(mat_file_path)

print(whosmat(mat_file_path))

respiration = data['respiration']
radar_I = data['radar_I']
radar_Q = data['radar_Q']
fs = data['Fs']
fs = 2000
print(fs)

signal = radar_I + 1j * radar_Q
signalPhase = np.unwrap(np.angle(signal))
signalPhase = np.array(signalPhase.flatten())

fpcg = 800
fpulse = 40
fresp = 6

# acá las reutiliza para ahorrar recursos y no recorrer vectores tan grandes, es simplemente una optimización pero podría
# recorrer la señal original todas las veces.
pcgDecRate = int(fs/fpcg)
pcgFs = fs / pcgDecRate
pcgDecSignal = sp.decimate(signalPhase, pcgDecRate,ftype='fir')

pulseDecRate = int(pcgFs/fpulse)
pulseFs = pcgFs / pulseDecRate
pulseDecSignal = sp.decimate(pcgDecSignal, pulseDecRate,ftype='fir')

respDecRate = int(pulseFs/fresp)
respFs = pulseFs / respDecRate
respDecSignal = sp.decimate(pulseDecSignal, respDecRate,ftype='fir')

#plt.plot((respDecSignal))
#plt.show()

configPCG = Configuration(
    min_freq=15,
    max_freq=200,
    num_freqs=50,
    ts=1 / pcgFs,
    wcf=1,
    wbw=8,
    wavelet_bounds=(-8, 8),
    threshold=abs(pcgDecSignal).max() / 1e6,
    transform='tsst',
)

pcgIters = 2
pcgMethod = 'threshold'
pcgThreshold = abs(pcgDecSignal).max() / 1e5
pcgItl = False

sst, _, freqs, _, _ = sswt(pcgDecSignal, **configPCG.asdict(), tsst=False)
asst, afreqs, _, _ = adaptive_sswt(pcgDecSignal, pcgIters, pcgMethod, pcgThreshold, pcgItl, **configPCG.asdict(), tsst=False)


time = np.linspace(0, len(pcgDecSignal)*configPCG.ts, len(pcgDecSignal))

fig = plt.figure(figsize=(17/2.54,6/2.54), dpi=300)
gs = fig.add_gridspec(1, 1)
stAx = plt.subplot(gs[0, 0],)
plot_tf_repr(asst, time, afreqs, stAx)
plt.show(block=False)
plt.show()


configPulse = Configuration(
    min_freq=1,
    max_freq=3,
    num_freqs=16,
    ts=1 / pulseFs,
    wcf=1,
    wbw=10,
    wavelet_bounds=(-8, 8),
    threshold=abs(pulseDecSignal).max() / 1e5,
    num_processes=4,
)

pulseIters = 1
pulseMethod = 'proportional'
pulseThreshold = abs(pulseDecSignal).max() / 10000
pulseItl = True


sst, _, freqs, _, _ = sswt(pulseDecSignal, **configPulse.asdict(), tsst=False)
asst, afreqs, _, _ = adaptive_sswt(pulseDecSignal, pulseIters, pulseMethod, pulseThreshold, pulseItl, **configPulse.asdict(), tsst=False)


time = np.linspace(0, len(pulseDecSignal)*configPulse.ts, len(pulseDecSignal))

fig = plt.figure(figsize=(17/2.54,6/2.54), dpi=300)
gs = fig.add_gridspec(1, 1)
stAx = plt.subplot(gs[0, 0],)
plot_tf_repr(asst, time, afreqs, stAx)
plt.show(block=False)
plt.show()
