"""Design a controller around the upright position."""

import control
import numpy as np

from .physics import derivatives


def design_controller(physics, controller_settings):
    """
    Calculate LQR gains using a local linear approximation of the physics.

    LQR balances two goals: keeping the state near zero and using less force.
    It is intended for small tilts near upright, not swinging up from below.
    """
    upright_state = np.zeros(4)
    derivative_step = 1e-6
    system_state_matrix = np.zeros((4, 4))

    # Perturb one state at a time to measure how its derivatives change.
    # These numerical slopes form the linear model: state_dot = A state + B F.
    for state_index in range(4):
        state_perturbation = np.zeros(4)
        state_perturbation[state_index] = derivative_step
        system_state_matrix[:, state_index] = (
            derivatives(0, upright_state + state_perturbation, 0, physics)
            - derivatives(0, upright_state - state_perturbation, 0, physics)
        ) / (2 * derivative_step)

    motor_input_matrix = (
        (
            derivatives(0, upright_state, derivative_step, physics)
            - derivatives(0, upright_state, -derivative_step, physics)
        )
        / (2 * derivative_step)
    ).reshape(4, 1)
    state_penalty_matrix = np.diag(controller_settings["state_weights"])
    force_penalty_matrix = np.array([[controller_settings["force_weight"]]])
    controller_gains, _, _ = control.lqr(
        system_state_matrix,
        motor_input_matrix,
        state_penalty_matrix,
        force_penalty_matrix,
        method="scipy",
    )
    return np.asarray(controller_gains).ravel()


def calculate_force(measured_state, controller_gains, motor_force_limit):
    """
    Request cart force based on errors, capped by the controller setting.

    With the DC motor enabled, its voltage/current limits further affect the
    force actually delivered; this function only produces the request.
    """
    # The target state is all zeros, so measurement is also the state error.
    requested_motor_force = -np.dot(controller_gains, measured_state)
    return float(np.clip(requested_motor_force, -motor_force_limit, motor_force_limit))


def upright_angle(stick_angle):
    """Angle error to the nearest upright position, always in [-pi, pi)."""
    return (stick_angle + np.pi) % (2 * np.pi) - np.pi


class SwingUpController:
    """Pump pendulum energy, then hand over to LQR near the upright position."""

    def __init__(self, physics, controller_settings, swing_up_settings):
        self.physics = physics
        self.controller_settings = controller_settings
        self.swing_up_settings = swing_up_settings
        self.controller_gains = design_controller(physics, controller_settings)
        self.phase = "swing-up"
        self.cart_target_position = 0.0

    def force(self, measured_state, simulation_time):
        cart_position, cart_velocity, stick_angle, stick_angular_velocity = measured_state
        stick_angle = upright_angle(stick_angle)
        swing_up_settings = self.swing_up_settings
        # Hysteresis prevents rapid switching at the capture boundary.
        if self.phase == "balance":
            if abs(stick_angle) > np.deg2rad(swing_up_settings["release_angle_degrees"]):
                self.phase = "swing-up"
        elif (
            abs(stick_angle) < np.deg2rad(swing_up_settings["capture_angle_degrees"])
            and abs(stick_angular_velocity) < swing_up_settings["capture_angular_speed"]
        ):
            self.phase = "balance"
            # Catch where the cart is now instead of demanding an instant return
            # to the original center, which could knock the pendulum over.
            self.cart_target_position = cart_position
        if self.phase == "balance":
            state_error = np.array(
                [
                    cart_position - self.cart_target_position,
                    cart_velocity,
                    stick_angle,
                    stick_angular_velocity,
                ]
            )
            return calculate_force(
                state_error, self.controller_gains, self.controller_settings["max_force"]
            )

        physics = self.physics
        stick_length = physics["stick_length"]
        pendulum_mass_moment = (physics["stick_mass"] / 2 + physics["top_mass"]) * stick_length
        pendulum_pivot_inertia = (physics["stick_mass"] / 3 + physics["top_mass"]) * stick_length**2
        pendulum_energy = (
            pendulum_pivot_inertia * stick_angular_velocity**2 / 2
            + pendulum_mass_moment * physics["gravity"] * np.cos(stick_angle)
        )
        pendulum_upright_energy = pendulum_mass_moment * physics["gravity"]
        # The moving pivot changes pendulum energy according to
        # E_dot = -h*cos(angle)*angular_speed*cart_acceleration - friction losses.
        cart_acceleration = (
            swing_up_settings["energy_gain"]
            * (pendulum_energy - pendulum_upright_energy)
            * stick_angular_velocity
            * np.cos(stick_angle)
            - swing_up_settings["position_gain"] * cart_position
            - swing_up_settings["speed_gain"] * cart_velocity
        )
        # Exactly hanging and stationary has no preferred swing direction.
        # A brief initial cart acceleration breaks that symmetry.
        if simulation_time < swing_up_settings["kick_duration"]:
            cart_acceleration = swing_up_settings["kick_acceleration"]
        cart_acceleration = np.clip(
            cart_acceleration,
            -swing_up_settings["max_acceleration"],
            swing_up_settings["max_acceleration"],
        )
        # Convert desired cart acceleration to force using the coupled physics.
        stick_angular_acceleration = (
            pendulum_mass_moment * physics["gravity"] * np.sin(stick_angle)
            - physics["pivot_friction"] * stick_angular_velocity
            - pendulum_mass_moment * np.cos(stick_angle) * cart_acceleration
        ) / pendulum_pivot_inertia
        system_total_mass = physics["cart_mass"] + physics["stick_mass"] + physics["top_mass"]
        requested_cart_force = (
            system_total_mass * cart_acceleration
            + pendulum_mass_moment * np.cos(stick_angle) * stick_angular_acceleration
            + physics["cart_friction"] * cart_velocity
            - pendulum_mass_moment * np.sin(stick_angle) * stick_angular_velocity**2
        )
        return float(
            np.clip(
                requested_cart_force,
                -self.controller_settings["max_force"],
                self.controller_settings["max_force"],
            )
        )
