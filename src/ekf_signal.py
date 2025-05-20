from filterpy.kalman import ExtendedKalmanFilter as EKF
from numpy import array, sqrt
import sympy
from sympy.abc import alpha, x, y, v, w, R, theta
from sympy import symbols, Matrix
import numpy as np
class RadarSignalEKF(EKF):
    def __init__(self, dt, epsilon_a, f_norm=0.3):
        super().__init__(dim_x=3, dim_z=1)

        self.dt = dt
        self.epsilon_a = epsilon_a  # std deviation for process noise
        self.f_norm = f_norm # Frecuencia nominal para normalizar


        # Initial state estimate
        self.x = np.zeros((3, 1))  # [x1, x2, x3]
        #self.x = np.array([1, 1, 10])
        self.x = np.array([1, 1, 1])

        # Initial covariance matrix
        self.P *= 10

        self.R = 5e4

        # Process noise
        self.Q = np.diag([1.0, 1.0, 2.0]) * 2e-2

    ''' # Esto es para el modelo con rho
    def predict(self):
        x1, x2, x3 = self.x.flatten()

        # Nonlinear state transition
        x1_new = 2 * np.cos(x3) * x1 - x2
        x2_new = x1
        x3_new = x3  # If Omega_0 is unknown, you could model this as slowly drifting

        self.x = np.array([[x1_new], [x2_new], [x3_new]])

        # Jacobian of the transition function F
        F = np.array([
            [2 * np.cos(x3), -1, -2 * x1 * np.sin(x3)],
            [1, 0, 0],
            [0, 0, 1]
        ])
        
        self.P = F @ self.P @ F.T + self.Q

    '''
    # Esto es para el modelo de fasores
    def predict(self):
        x1, x2, x3norm = self.x.flatten()
        x3 = x3norm * self.f_norm


        # Nonlinear state transition
        cos_x3 = np.cos(x3)
        sin_x3 = np.sin(x3)

        x1_new = cos_x3 * x1 - sin_x3 * x2
        x2_new = sin_x3 * x1 + cos_x3 * x2
        #print(x3)
        x3norm_new = (0.9 * x3norm)   # You assume w(t) ~ N(0, Q), modeled via self.Q

        self.x = np.array([[x1_new], [x2_new], [x3norm_new]])

        # Jacobian of f(x)
        F = np.array([
            [cos_x3, -sin_x3, -sin_x3 * x1 - cos_x3 * x2],
            [sin_x3,  cos_x3,  cos_x3 * x1 - sin_x3 * x2],
            [0,       0,       1]
        ])

        # Covariance prediction
        self.P = F @ self.P @ F.T + self.Q


