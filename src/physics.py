"""Nonlinear cart-and-stick physics, including the mass at the top."""

import numpy as np


def derivatives(simulation_time, system_state, cart_force, physics, tip_contact_force=None):
    """
    Return rates of change for [position, speed, angle, angular speed].

    Angle is in radians: zero is upright; positive leans to the right.
    The solver supplies time even though these equations do not need it.
    """
    _cart_position, cart_velocity, stick_angle, stick_angular_velocity = system_state
    stick_length = physics["stick_length"]
    stick_mass = physics["stick_mass"]
    top_weight_mass = physics["top_mass"]

    system_total_mass = physics["cart_mass"] + stick_mass + top_weight_mass
    # Mass times distance from pivot, summed over the stick and top weight.
    pendulum_mass_moment = stick_mass * stick_length / 2 + top_weight_mass * stick_length
    # A uniform rod contributes m L² / 3; a point mass contributes m L².
    pendulum_pivot_inertia = stick_mass * stick_length**2 / 3 + top_weight_mass * stick_length**2

    # Cart and stick accelerations are coupled: moving one affects the other.
    # Solve the two Newton/Lagrange equations together rather than guessing
    # the angular acceleration independently of the cart acceleration.
    cart_stick_coupling = pendulum_mass_moment * np.cos(stick_angle)
    system_mass_matrix = np.array(
        [
            [system_total_mass, cart_stick_coupling],
            [cart_stick_coupling, pendulum_pivot_inertia],
        ]
    )
    generalized_forces = np.array(
        [
            cart_force
            - physics["cart_friction"] * cart_velocity
            + pendulum_mass_moment * np.sin(stick_angle) * stick_angular_velocity**2,
            pendulum_mass_moment * physics["gravity"] * np.sin(stick_angle)
            - physics["pivot_friction"] * stick_angular_velocity,
        ]
    )
    if tip_contact_force is not None:
        tip_horizontal_force, tip_vertical_force = tip_contact_force
        # An external push on the tip affects BOTH cart motion and rotation.
        # The tip moves by [L*cos(angle), -L*sin(angle)] per radian, so its
        # force contributes this generalized torque about the moving pivot.
        generalized_forces[0] += tip_horizontal_force
        generalized_forces[1] += stick_length * (
            np.cos(stick_angle) * tip_horizontal_force - np.sin(stick_angle) * tip_vertical_force
        )
    cart_acceleration, stick_angular_acceleration = np.linalg.solve(
        system_mass_matrix, generalized_forces
    )
    return np.array(
        [cart_velocity, cart_acceleration, stick_angular_velocity, stick_angular_acceleration]
    )
