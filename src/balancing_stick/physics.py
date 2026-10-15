"""Nonlinear cart-and-stick physics, including the mass at the top."""

import numpy as np


def derivatives(time, state, force, physics):
    """Return rates of change for [position, speed, angle, angular speed].

    Angle is in radians: zero is upright; positive leans to the right.
    The solver supplies time even though these equations do not need it.
    """
    _position, speed, angle, angular_speed = state
    p = physics
    length = p["stick_length"]
    stick_mass = p["stick_mass"]
    top_mass = p["top_mass"]

    total_mass = p["cart_mass"] + stick_mass + top_mass
    # Mass times distance from pivot, summed over the stick and top weight.
    mass_distance = stick_mass * length / 2 + top_mass * length
    # A uniform rod contributes m L² / 3; a point mass contributes m L².
    pivot_inertia = stick_mass * length**2 / 3 + top_mass * length**2

    # Cart and stick accelerations are coupled: moving one affects the other.
    # Solve the two Newton/Lagrange equations together rather than guessing
    # the angular acceleration independently of the cart acceleration.
    coupling = mass_distance * np.cos(angle)
    mass_matrix = np.array([
        [total_mass, coupling],
        [coupling, pivot_inertia],
    ])
    forces = np.array([
        force - p["cart_friction"] * speed
        + mass_distance * np.sin(angle) * angular_speed**2,
        mass_distance * p["gravity"] * np.sin(angle)
        - p["pivot_friction"] * angular_speed,
    ])
    acceleration, angular_acceleration = np.linalg.solve(mass_matrix, forces)
    return np.array([speed, acceleration, angular_speed, angular_acceleration])
