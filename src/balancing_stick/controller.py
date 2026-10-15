"""Design a controller around the upright position."""

import control
import numpy as np

from .physics import derivatives


def design_controller(physics, settings):
    """
    Calculate LQR gains using a local linear approximation of the physics.

    LQR balances two goals: keeping the state near zero and using less force.
    It is intended for small tilts near upright, not swinging up from below.
    """
    origin = np.zeros(4)
    epsilon = 1e-6
    a = np.zeros((4, 4))

    # Perturb one state at a time to measure how its derivatives change.
    # These numerical slopes form the linear model: state_dot = A state + B F.
    for column in range(4):
        offset = np.zeros(4)
        offset[column] = epsilon
        a[:, column] = (
            derivatives(0, origin + offset, 0, physics)
            - derivatives(0, origin - offset, 0, physics)
        ) / (2 * epsilon)

    b = ((derivatives(0, origin, epsilon, physics)
          - derivatives(0, origin, -epsilon, physics))
         / (2 * epsilon)).reshape(4, 1)
    q = np.diag(settings["state_weights"])
    r = np.array([[settings["force_weight"]]])
    gains, _, _ = control.lqr(a, b, q, r, method="scipy")
    return np.asarray(gains).ravel()


def calculate_force(measurement, gains, max_force):
    """Push the cart based on errors, respecting the motor's force limit."""
    # The target state is all zeros, so measurement is also the state error.
    requested_force = -np.dot(gains, measurement)
    return float(np.clip(requested_force, -max_force, max_force))
