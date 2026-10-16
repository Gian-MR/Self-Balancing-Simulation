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
    """
    Request cart force based on errors, capped by the controller setting.

    With the DC motor enabled, its voltage/current limits further affect the
    force actually delivered; this function only produces the request.
    """
    # The target state is all zeros, so measurement is also the state error.
    requested_force = -np.dot(gains, measurement)
    return float(np.clip(requested_force, -max_force, max_force))


def upright_angle(angle):
    """Angle error to the nearest upright position, always in [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


class SwingUpController:
    """Pump pendulum energy, then hand over to LQR near the upright position."""

    def __init__(self, physics, settings, swing_settings):
        self.physics = physics
        self.settings = settings
        self.swing = swing_settings
        self.gains = design_controller(physics, settings)
        self.phase = "swing-up"
        self.cart_target = 0.0

    def force(self, measurement, time):
        position, speed, angle, angular_speed = measurement
        angle = upright_angle(angle)
        s = self.swing
        # Hysteresis prevents rapid switching at the capture boundary.
        if self.phase == "balance":
            if abs(angle) > np.deg2rad(s["release_angle_degrees"]):
                self.phase = "swing-up"
        elif (abs(angle) < np.deg2rad(s["capture_angle_degrees"])
              and abs(angular_speed) < s["capture_angular_speed"]):
            self.phase = "balance"
            # Catch where the cart is now instead of demanding an instant return
            # to the original center, which could knock the pendulum over.
            self.cart_target = position
        if self.phase == "balance":
            error = np.array([position - self.cart_target, speed, angle, angular_speed])
            return calculate_force(error, self.gains, self.settings["max_force"])

        p = self.physics
        length = p["stick_length"]
        h = (p["stick_mass"] / 2 + p["top_mass"]) * length
        inertia = (p["stick_mass"] / 3 + p["top_mass"]) * length**2
        energy = inertia * angular_speed**2 / 2 + h * p["gravity"] * np.cos(angle)
        upright_energy = h * p["gravity"]
        # The moving pivot changes pendulum energy according to
        # E_dot = -h*cos(angle)*angular_speed*cart_acceleration - friction losses.
        acceleration = (s["energy_gain"] * (energy - upright_energy)
                        * angular_speed * np.cos(angle)
                        - s["position_gain"] * position - s["speed_gain"] * speed)
        # Exactly hanging and stationary has no preferred swing direction.
        # A brief initial cart acceleration breaks that symmetry.
        if time < s["kick_duration"]:
            acceleration = s["kick_acceleration"]
        acceleration = np.clip(acceleration, -s["max_acceleration"], s["max_acceleration"])
        # Convert desired cart acceleration to force using the coupled physics.
        angular_acceleration = (h * p["gravity"] * np.sin(angle)
                                - p["pivot_friction"] * angular_speed
                                - h * np.cos(angle) * acceleration) / inertia
        total_mass = p["cart_mass"] + p["stick_mass"] + p["top_mass"]
        force = (total_mass * acceleration + h * np.cos(angle) * angular_acceleration
                 + p["cart_friction"] * speed - h * np.sin(angle) * angular_speed**2)
        return float(np.clip(force, -self.settings["max_force"], self.settings["max_force"]))
