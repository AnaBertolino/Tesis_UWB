from scipy.signal import butter, lfilter
from filterpy.kalman import KalmanFilter, ExtendedKalmanFilter
import numpy as np
import matplotlib.pyplot as plt
from src.ekf_signal import RadarSignalEKF

from scipy.signal import iirnotch, filtfilt

def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def bandpass_filter(data, fs, f_low, f_high, order=4):
    nyq = 0.5 * fs  # Frecuencia de Nyquist
    low = f_low / nyq
    high = f_high / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def kalman_filter(signal, fs, Q_value, R_value, plot=False):

    """
    Aplica un filtro de Kalman a una señal dada.
    
    Parámetros:
    - signal: np.array, señal ruidosa de entrada.
    - fs: float, frecuencia de muestreo.
    - Q_value: float, valor de la covarianza del proceso.
    - R_value: float, valor de la covarianza de medición.
    - plot: bool, si True grafica la señal original y la filtrada.
    
    Retorna:
    - np.array con la señal filtrada.
    """
    # Definir el filtro de Kalman
    kf = KalmanFilter(dim_x=3, dim_z=1)  # 3 estados, 1 medición
    
    # Matriz de transición de estado (modelo de movimiento)
    dt = 1 / fs  # Intervalo de tiempo entre muestras
    kf.F = np.array([[1, dt, 0],
                     [0, 1, dt],
                     [0, 0, 1]])
    
    # Matriz de observación (solo medimos amplitud)
    kf.H = np.array([[1, 0, 0]])
    
    # Covarianza del proceso (ruido del modelo)
    kf.Q = np.eye(3) * Q_value
    
    # Covarianza de la medición (ruido de medición)
    kf.R = np.array([[R_value]])
    
    # Estado inicial
    kf.x = np.array([[0], [0], [1]])  # (Amplitud, velocidad, frecuencia)
    kf.P = np.eye(3)  # Covarianza inicial
    
    # Aplicamos el filtro de Kalman a la señal ruidosa
    filtered_signal = []
    for z in signal:
        kf.predict()
        kf.update(np.array([[z]]))
        filtered_signal.append(kf.x[0, 0])  # Guardamos la amplitud estimada
    
    filtered_signal = np.array(filtered_signal)
    
    # Graficamos si se solicita
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(signal, label="Señal ruidosa", alpha=0.5)
        plt.plot(filtered_signal, label="Señal filtrada (Kalman)", linewidth=2)
        plt.legend()
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Amplitud")
        plt.title("Filtro de Kalman aplicado a una señal UWB")
        plt.show()
    
    return filtered_signal

def extended_kalman_filter(s, fs, epsilon_a, plot=False):

    ekf = RadarSignalEKF(1/fs, epsilon_a)
    std_d = np.sqrt(np.var(s)) # Normalizo la señal con su varianza
    s_norm = s / std_d
    #print(np.var(s))

    ''' # Esto es para el modelo con rho
    def hx(x):
        x1, x2, _ = x
        return np.array([-(1 + rho**2) * x1 + rho**2 * x2])
    
    def H_jacobian(x):
        return np.array([[-(1 + rho**2), rho**2, 0]])
    '''
    # Esto es para el modelo de fasores
    def hx(x):
        return np.array([x[0]])
    
    # Jacobian of h(x)
    def H_jacobian(x):
        return np.array([[1.0, 0.0, 0.0]])

    
    estimated_states = []
    filtered_outputs = []
    # At each time step:
    for t in range(len(s_norm)):
        z = s_norm[t]      # from radar
        ekf.predict()
        ekf.update(z, HJacobian=H_jacobian, Hx=hx)

        ekf.predict_update(
            z,
            H_jacobian,
            hx,
            args=(), hx_args=()
        )
            # Save filtered state and estimated measurement
        estimated_states.append(ekf.x.copy() * std_d) 
        filtered_outputs.append(hx(ekf.x)[0])  # estimated y(t)

    
    if(plot):
        plt.figure()
        plt.plot(s_norm, label="Measurement")
        plt.plot(np.array(estimated_states)[:, 0], label=f'$x_{1}(t)$')
        plt.legend()
        plt.title('Estimated States over Time')
        plt.xlabel('Time step')
        plt.grid(True)
        plt.show()
    return np.array(filtered_outputs), np.array(estimated_states)



def notch_filter(signal, fs, freqs, Q=30):
    """Aplica múltiples filtros notch a la señal en las frecuencias indicadas."""
    filtered = signal.copy()
    for f in freqs:
        b, a = iirnotch(w0=f/(fs/2), Q=Q)
        filtered = filtfilt(b, a, filtered)
    return filtered
