"""Run a sampled controller against continuous nonlinear physics."""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Any

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


def simulation_steps(config, mode="balance", contact=None, live=False):
    """Yield physical samples one control interval at a time.

    Batch runs and the interactive window use exactly the same physics. Live
    swing-up/balance runs continue beyond the configured recording duration so
    the user can keep nudging. Fall mode still stops at its fall-angle limit.
    """
    validate_config(config, mode)
    if mode not in ("balance", "fall", "swing-up"):
        raise ValueError("mode must be 'balance', 'fall', or 'swing-up'")
    p, s, c = config["physics"], config["simulation"], config["controller"]
    sensor = config["sensors"]
    motor_settings = config.get("motor", {})
    motor = DCMotor(motor_settings) if motor_settings.get("enabled", False) else None
    swing = SwingUpController(p, c, config["swing_up"]) if mode == "swing-up" else None
    duration = config["swing_up"]["duration"] if swing is not None else s["duration"]
    gains = design_controller(p, c) if mode == "balance" else None
    rng = np.random.default_rng(sensor["seed"])
    state = np.array([0.0, 0.0, np.deg2rad(s["initial_angle_degrees"]), 0.0])
    if swing is not None:
        state[2] = np.pi
    time, voltage, force = 0.0, 0.0, 0.0
    phase = mode
    reason = "Completed requested duration"

    class FallEvent:
        # SciPy reads these attributes on its callable event object.
        terminal = True
        direction = -1

        def __call__(self, t, y):
            return np.deg2rad(s["fall_angle_degrees"]) - abs(y[2])

    fallen = FallEvent()

    def applied_force(y):
        if motor is None:
            return force
        return motor.evaluate(voltage, y[1], coast=(mode == "fall"))["force"]

    def tip_force(y):
        return contact.force_on_tip(y, p) if contact is not None else np.zeros(2)

    def sample():
        return {
            "time": time, "state": state.copy(), "force": applied_force(state),
            "requested_force": force, "phase": phase, "tip_force": tip_force(state),
            "motor": motor.evaluate(voltage, state[1], coast=(mode == "fall"))
                     if motor is not None else None,
        }

    while live or time < duration:
        measurement = read_sensors(state, sensor["noise_std"], rng)
        force = (calculate_force(measurement, gains, c["max_force"])
                 if mode == "balance" else 0.0)
        if swing is not None:
            force = swing.force(measurement, time)
        phase = swing.phase if swing is not None else mode
        if motor is not None:
            voltage = motor.voltage_for_force(force, measurement[1]) if mode != "fall" else 0.0
        # Yield before integration: the UI can move the obstacle between steps.
        yield sample()
        next_time = time + s["control_interval"]
        if not live:
            next_time = min(next_time, duration)
        # Hold commanded voltage fixed while actual force changes with speed.
        # Recompute contact inside the solver so it responds to tip movement.
        solution = solve_ivp(
            lambda t, y: derivatives(t, y, applied_force(y), p, tip_force(y)),
            (time, next_time), state, events=fallen if swing is None else None,
            rtol=1e-8, atol=1e-10, max_step=s["control_interval"] / 2,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        time = float(solution.t[-1])
        state = solution.y[:, -1].copy()
        if solution.status == 1:
            reason = "Stopped: stick reached the fall angle limit"
            break

    if swing is not None:
        angle_error = (state[2] + np.pi) % (2 * np.pi) - np.pi
        settled = (swing.phase == "balance" and abs(angle_error) < np.deg2rad(2)
                   and abs(state[1]) < 0.1 and abs(state[3]) < 0.1)
        reason = ("Swing-up complete: balancing upright" if settled
                  else "Swing-up duration ended before settling upright")
    final = sample()
    final["reason"] = reason
    yield final


def samples_to_result(samples, mode):
    """Convert recorded samples into arrays used by plots and CSV export."""
    result: dict[str, Any] = {
        key: np.array([sample[key] for sample in samples])
        for key in ("time", "state", "force", "requested_force", "phase", "tip_force")
    }
    result["reason"] = samples[-1].get("reason", "Live simulation")
    result["mode"] = mode
    result["motor"] = ({key: np.array([sample["motor"][key] for sample in samples])
                        for key in samples[0]["motor"]}
                       if samples[0]["motor"] is not None else None)
    return result


def run_simulation(config, mode="balance"):
    """Run a complete undisturbed experiment for saved results or tests."""
    return samples_to_result(list(simulation_steps(config, mode)), mode)
