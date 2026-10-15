"""Simple sensor model: exact state readings, with optional random noise."""

import numpy as np


def read_sensors(true_state, noise_std, random_generator):
    """
    Return a separate measurement, leaving the actual physics untouched.

    We assume all four states are available. Real velocity measurements may
    require estimation; this first version does not implement that estimator.
    """
    noise = random_generator.normal(0.0, np.asarray(noise_std))
    return np.asarray(true_state).copy() + noise
