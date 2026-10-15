"""Run a sampled controller against continuous nonlinear physics."""

import numpy as np
from scipy.integrate import solve_ivp

from .controller import calculate_force, design_controller
from .physics import derivatives
from .sensors import read_sensors


def validate_config(config):
    """Reject common parameter mistakes before starting the physics solver."""

    p = config["physics"]
    s = config["simulation"]
    c = config["controller"]

    for key in ("cart_mass", "stick_length", "gravity"):
        if not np.isfinite(p[key]) or p[key] <= 0:
            raise ValueError(f"physics.{key} must be finite and positive")
        
    for key in ("stick_mass", "top_mass", "cart_friction", "pivot_friction"):
        if not np.isfinite(p[key]) or p[key] < 0:
            raise ValueError(f"physics.{key} must be finite and nonnegative")
        
    if p["stick_mass"] + p["top_mass"] <= 0:
        raise ValueError("The stick or top weight must have mass")
    
    for key in ("duration", "control_interval"):
        if not np.isfinite(s[key]) or s[key] <= 0:
            raise ValueError(f"simulation.{key} must be finite and positive")
        
    if not 0 < s["fall_angle_degrees"] < 90:
        raise ValueError("fall_angle_degrees must be between 0 and 90")
    
    if not abs(s["initial_angle_degrees"]) < s["fall_angle_degrees"]:
        raise ValueError("Initial angle must be inside the fall limit")
    
    weights = np.asarray(c["state_weights"])
    noise = np.asarray(config["sensors"]["noise_std"])

    if weights.shape != (4,) or not np.all(np.isfinite(weights) & (weights > 0)):
        raise ValueError("state_weights must contain four positive finite values")
    
    if noise.shape != (4,) or not np.all(np.isfinite(noise) & (noise >= 0)):
        raise ValueError("noise_std must contain four nonnegative finite values")
    
    for key in ("max_force", "force_weight"):
        if not np.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f"controller.{key} must be finite and positive")


def run_simulation(config, mode="balance"):
    """Return times, true states, applied forces, and the stopping reason."""
    validate_config(config)
    if mode not in ("balance", "fall"):
        raise ValueError("mode must be 'balance' or 'fall'")
    
    p = config["physics"]
    s = config["simulation"]
    c = config["controller"]
    sensor = config["sensors"]
    
    gains = design_controller(p, c) if mode == "balance" else None
    rng = np.random.default_rng(sensor["seed"])
    state = np.array([0.0, 0.0, np.deg2rad(s["initial_angle_degrees"]), 0.0])
    time = 0.0
    times, states, forces = [time], [state.copy()], []
    reason = "Completed requested duration"

    # Stop before horizontal: this model has no floor collisions or rail ends.
    def fallen(t, y):
        return np.deg2rad(s["fall_angle_degrees"]) - abs(y[2])

    fallen.terminal = True
    fallen.direction = -1

    while time < s["duration"] - 1e-12:
        measurement = read_sensors(state, sensor["noise_std"], rng)
        force = (calculate_force(measurement, gains, c["max_force"])
                 if mode == "balance" else 0.0)
        forces.append(force)
        next_time = min(time + s["control_interval"], s["duration"])

        # Hold the motor force constant between sensor/control updates.
        # SciPy takes smaller internal steps as needed for accurate physics.
        solution = solve_ivp(
            lambda t, y: derivatives(t, y, force, p),
            (time, next_time), state, events=fallen,
            rtol=1e-8, atol=1e-10,
            max_step=s["control_interval"] / 2,
        )
        
        if not solution.success:
            raise RuntimeError(solution.message)
        time = float(solution.t[-1])
        state = solution.y[:, -1].copy()
        times.append(time)
        states.append(state)
        if solution.status == 1:
            reason = "Stopped: stick reached the fall angle limit"
            break

    # Force at the final sample is the force from the preceding interval.
    forces.append(forces[-1])
    return {
        "time": np.array(times), "state": np.array(states),
        "force": np.array(forces), "reason": reason, "mode": mode,
    }
