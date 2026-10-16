"""Simple sensor model: exact state readings, with optional random noise."""

import numpy as np


def read_sensors(system_true_state, sensor_noise_std, sensor_random_generator):
    """Return a noisy measurement of the system state."""
    
    sensor_noise = sensor_random_generator.normal(0.0, np.asarray(sensor_noise_std))
    return np.asarray(system_true_state).copy() + sensor_noise
