"""Run a sampled controller against continuous nonlinear physics."""

import numpy as np
from scipy.integrate import solve_ivp

from .controller import SwingUpController, calculate_force, design_controller
from .motor import DCMotor
from .physics import derivatives
from .sensors import read_sensors


def validate_config(config, mode="balance"):
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
    if mode == "swing-up":
        swing = config["swing_up"]
        for key in ("duration", "energy_gain", "max_acceleration", "kick_duration",
                    "kick_acceleration", "capture_angular_speed"):
            if not np.isfinite(swing[key]) or swing[key] <= 0:
                raise ValueError(f"swing_up.{key} must be finite and positive")
        for key in ("position_gain", "speed_gain"):
            if not np.isfinite(swing[key]) or swing[key] < 0:
                raise ValueError(f"swing_up.{key} must be finite and nonnegative")
        if not 0 < swing["capture_angle_degrees"] < swing["release_angle_degrees"] < 90:
            raise ValueError("Swing-up angles must satisfy 0 < capture < release < 90")


def run_simulation(config, mode="balance"):
    """Return times, true states, applied forces, and the stopping reason."""
    validate_config(config, mode)
    if mode not in ("balance", "fall", "swing-up"):
        raise ValueError("mode must be 'balance', 'fall', or 'swing-up'")
    
    p = config["physics"]
    s = config["simulation"]
    c = config["controller"]
    sensor = config["sensors"]
    motor_settings = config.get("motor", {})
    motor = DCMotor(motor_settings) if motor_settings.get("enabled", False) else None
    swing = SwingUpController(p, c, config["swing_up"]) if mode == "swing-up" else None
    duration = config["swing_up"]["duration"] if swing is not None else s["duration"]
    
    gains = design_controller(p, c) if mode == "balance" else None
    rng = np.random.default_rng(sensor["seed"])
    state = np.array([0.0, 0.0, np.deg2rad(s["initial_angle_degrees"]), 0.0])
    if swing is not None:
        state[2] = np.pi  # 180 degrees: hanging straight below the pivot.
    time = 0.0
    times, states, forces = [time], [state.copy()], []
    requested_forces, motor_readings = [], []
    voltage = 0.0
    phases = []
    reason = "Completed requested duration"

    # Stop near-upright experiments before horizontal. Swing-up intentionally
    # passes through horizontal, so it does not use this stopping event.
    def fallen(t, y):
        return np.deg2rad(s["fall_angle_degrees"]) - abs(y[2])

    fallen.terminal = True
    fallen.direction = -1

    def applied_force(y):
        if motor is None:
            return force
        return motor.evaluate(voltage, y[1], coast=(mode == "fall"))["force"]

    while time < duration:
        measurement = read_sensors(state, sensor["noise_std"], rng)
        force = (calculate_force(measurement, gains, c["max_force"])
                 if mode == "balance" else 0.0)
        if swing is not None:
            force = swing.force(measurement, time)
        phases.append(swing.phase if swing is not None else mode)
        requested_forces.append(force)
        if motor is not None:
            voltage = motor.voltage_for_force(force, measurement[1]) if mode != "fall" else 0.0
            motor_readings.append(motor.evaluate(voltage, state[1], coast=(mode == "fall")))
        forces.append(applied_force(state))
        next_time = min(time + s["control_interval"], duration)

        # Hold commanded VOLTAGE constant between controller updates. Actual motor
        # force changes with speed inside the solver because of back EMF.
        # With the motor disabled in YAML, hold ideal force constant instead.
        # SciPy takes smaller internal steps as needed for accurate physics.
        solution = solve_ivp(
            lambda t, y: derivatives(t, y, applied_force(y), p),
            (time, next_time), state, events=fallen if swing is None else None,
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

    # Record the final physical force using the last held voltage.
    forces.append(applied_force(state))
    requested_forces.append(requested_forces[-1])
    if motor is not None:
        motor_readings.append(motor.evaluate(voltage, state[1], coast=(mode == "fall")))
    if swing is not None:
        angle_error = (state[2] + np.pi) % (2 * np.pi) - np.pi
        settled = (swing.phase == "balance" and abs(angle_error) < np.deg2rad(2)
                   and abs(state[1]) < 0.1 and abs(state[3]) < 0.1)
        reason = ("Swing-up complete: balancing upright" if settled
                  else "Swing-up duration ended before settling upright")
    result = {
        "time": np.array(times), "state": np.array(states),
        "force": np.array(forces), "reason": reason, "mode": mode,
        "requested_force": np.array(requested_forces),
        "phase": np.array(phases + [phases[-1]]),
    }
    result["motor"] = ({key: np.array([reading[key] for reading in motor_readings])
                        for key in motor_readings[0]} if motor is not None else None)
    return result
