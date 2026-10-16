"""Simple sensor model: exact state readings, with optional random noise."""

import numpy as np


def read_sensors(system_true_state, sensor_noise_std, sensor_random_generator):
    """
    Return a separate measurement, leaving the actual physics untouched.

    We assume all four states are available. Real velocity measurements may
    require estimation; this first version does not implement that estimator.
    """
    sensor_noise = sensor_random_generator.normal(0.0, np.asarray(sensor_noise_std))
    return np.asarray(system_true_state).copy() + sensor_noise
