"""Run a sampled controller against continuous nonlinear physics."""

from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from .controller import SwingUpController, calculate_force, design_controller
from .motor import DCMotor
from .physics import derivatives
from .sensors import read_sensors


def validate_config(config, mode="balance"):
    """Reject common parameter mistakes before starting the physics solver."""

    physics = config["physics"]
    simulation_settings = config["simulation"]
    controller_settings = config["controller"]

    # Dimensions come only from YAML: require every size instead of substituting
    # hidden defaults when a setting is missing or misspelled.
    geometry_settings = config.get("geometry")
    if geometry_settings is None:
        raise ValueError("The configuration must include a geometry section")
    if not isinstance(geometry_settings, dict):
        raise TypeError("The geometry section must be a mapping of dimension names to numbers")
    for setting_name in ("top_diameter", "rod_diameter", "cart_width", "cart_height", "box_size"):
        if setting_name not in geometry_settings:
            raise ValueError(f"Missing required setting: geometry.{setting_name}")
        setting_value = geometry_settings[setting_name]
        if (
            not isinstance(setting_value, (int, float))
            or isinstance(setting_value, bool)
            or not np.isfinite(setting_value)
            or setting_value <= 0
        ):
            raise ValueError(f"geometry.{setting_name} must be finite and positive")

    for setting_name in ("cart_mass", "stick_length", "gravity"):
        if not np.isfinite(physics[setting_name]) or physics[setting_name] <= 0:
            raise ValueError(f"physics.{setting_name} must be finite and positive")

    for setting_name in ("stick_mass", "top_mass", "cart_friction", "pivot_friction"):
        if not np.isfinite(physics[setting_name]) or physics[setting_name] < 0:
            raise ValueError(f"physics.{setting_name} must be finite and nonnegative")

    if physics["stick_mass"] + physics["top_mass"] <= 0:
        raise ValueError("The stick or top weight must have mass")

    for setting_name in ("duration", "control_interval"):
        if (
            not np.isfinite(simulation_settings[setting_name])
            or simulation_settings[setting_name] <= 0
        ):
            raise ValueError(f"simulation.{setting_name} must be finite and positive")

    if not 0 < simulation_settings["fall_angle_degrees"] < 90:
        raise ValueError("fall_angle_degrees must be between 0 and 90")

    if (
        not abs(simulation_settings["initial_angle_degrees"])
        < simulation_settings["fall_angle_degrees"]
    ):
        raise ValueError("Initial angle must be inside the fall limit")

    controller_state_weights = np.asarray(controller_settings["state_weights"])
    sensor_noise_std = np.asarray(config["sensors"]["noise_std"])

    if controller_state_weights.shape != (4,) or not np.all(
        np.isfinite(controller_state_weights) & (controller_state_weights > 0)
    ):
        raise ValueError("state_weights must contain four positive finite values")

    if sensor_noise_std.shape != (4,) or not np.all(
        np.isfinite(sensor_noise_std) & (sensor_noise_std >= 0)
    ):
        raise ValueError("noise_std must contain four nonnegative finite values")

    for setting_name in ("max_force", "force_weight"):
        if (
            not np.isfinite(controller_settings[setting_name])
            or controller_settings[setting_name] <= 0
        ):
            raise ValueError(f"controller.{setting_name} must be finite and positive")
    if mode == "swing-up":
        swing_up_settings = config["swing_up"]
        for setting_name in (
            "duration",
            "energy_gain",
            "max_acceleration",
            "kick_duration",
            "kick_acceleration",
            "capture_angular_speed",
        ):
            if (
                not np.isfinite(swing_up_settings[setting_name])
                or swing_up_settings[setting_name] <= 0
            ):
                raise ValueError(f"swing_up.{setting_name} must be finite and positive")
        for setting_name in ("position_gain", "speed_gain"):
            if (
                not np.isfinite(swing_up_settings[setting_name])
                or swing_up_settings[setting_name] < 0
            ):
                raise ValueError(f"swing_up.{setting_name} must be finite and nonnegative")
        if (
            not 0
            < swing_up_settings["capture_angle_degrees"]
            < swing_up_settings["release_angle_degrees"]
            < 90
        ):
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
    physics, simulation_settings, controller_settings = (
        config["physics"],
        config["simulation"],
        config["controller"],
    )
    sensor_settings = config["sensors"]
    motor_settings = config.get("motor", {})
    motor = DCMotor(motor_settings) if motor_settings.get("enabled", False) else None
    swing_up_controller = (
        SwingUpController(physics, controller_settings, config["swing_up"])
        if mode == "swing-up"
        else None
    )
    simulation_duration = (
        config["swing_up"]["duration"]
        if swing_up_controller is not None
        else simulation_settings["duration"]
    )
    controller_gains = (
        design_controller(physics, controller_settings) if mode == "balance" else None
    )
    sensor_random_generator = np.random.default_rng(sensor_settings["seed"])
    system_state = np.array(
        [0.0, 0.0, np.deg2rad(simulation_settings["initial_angle_degrees"]), 0.0]
    )
    if swing_up_controller is not None:
        system_state[2] = np.pi
    simulation_time, motor_command_voltage, requested_motor_force = 0.0, 0.0, 0.0
    controller_phase = mode
    stop_reason = "Completed requested duration"

    class FallEvent:
        # SciPy reads these attributes on its callable event object.
        terminal = True
        direction = -1

        def __call__(self, solver_time, solver_state):
            return np.deg2rad(simulation_settings["fall_angle_degrees"]) - abs(solver_state[2])

    fall_event = FallEvent()

    def applied_force(solver_state):
        if motor is None:
            return requested_motor_force
        return motor.evaluate(motor_command_voltage, solver_state[1], coast=(mode == "fall"))[
            "force"
        ]

    def tip_force(solver_state):
        return contact.force_on_tip(solver_state, physics) if contact is not None else np.zeros(2)

    def sample():
        return {
            "time": simulation_time,
            "state": system_state.copy(),
            "force": applied_force(system_state),
            "requested_force": requested_motor_force,
            "phase": controller_phase,
            "tip_force": tip_force(system_state),
            "motor": (
                motor.evaluate(motor_command_voltage, system_state[1], coast=(mode == "fall"))
                if motor is not None
                else None
            ),
        }

    while live or simulation_time < simulation_duration:
        measured_state = read_sensors(
            system_state, sensor_settings["noise_std"], sensor_random_generator
        )
        requested_motor_force = (
            calculate_force(measured_state, controller_gains, controller_settings["max_force"])
            if mode == "balance"
            else 0.0
        )
        if swing_up_controller is not None:
            requested_motor_force = swing_up_controller.force(measured_state, simulation_time)
        controller_phase = swing_up_controller.phase if swing_up_controller is not None else mode
        if motor is not None:
            motor_command_voltage = (
                motor.voltage_for_force(requested_motor_force, measured_state[1])
                if mode != "fall"
                else 0.0
            )
        # Yield before integration: the UI can move the obstacle between steps.
        yield sample()
        next_simulation_time = simulation_time + simulation_settings["control_interval"]
        if not live:
            next_simulation_time = min(next_simulation_time, simulation_duration)
        # Hold commanded voltage fixed while actual force changes with speed.
        # Recompute contact inside the solver so it responds to tip movement.
        integration_result = solve_ivp(
            lambda solver_time, solver_state: derivatives(
                solver_time,
                solver_state,
                applied_force(solver_state),
                physics,
                tip_force(solver_state),
            ),
            (simulation_time, next_simulation_time),
            system_state,
            events=fall_event if swing_up_controller is None else None,
            rtol=1e-8,
            atol=1e-10,
            max_step=simulation_settings["control_interval"] / 2,
        )
        if not integration_result.success:
            raise RuntimeError(integration_result.message)
        simulation_time = float(integration_result.t[-1])
        system_state = integration_result.y[:, -1].copy()
        if integration_result.status == 1:
            stop_reason = "Stopped: stick reached the fall angle limit"
            break

    if swing_up_controller is not None:
        stick_angle_error = (system_state[2] + np.pi) % (2 * np.pi) - np.pi
        stick_is_balanced = (
            swing_up_controller.phase == "balance"
            and abs(stick_angle_error) < np.deg2rad(2)
            and abs(system_state[1]) < 0.1
            and abs(system_state[3]) < 0.1
        )
        stop_reason = (
            "Swing-up complete: balancing upright"
            if stick_is_balanced
            else "Swing-up duration ended before settling upright"
        )
    final_sample = sample()
    final_sample["reason"] = stop_reason
    yield final_sample


def samples_to_result(samples, mode):
    """Convert recorded samples into arrays used by plots and CSV export."""
    result: dict[str, Any] = {
        result_field: np.array([sample[result_field] for sample in samples])
        for result_field in ("time", "state", "force", "requested_force", "phase", "tip_force")
    }
    result["reason"] = samples[-1].get("reason", "Live simulation")
    result["mode"] = mode
    result["motor"] = (
        {
            result_field: np.array([sample["motor"][result_field] for sample in samples])
            for result_field in samples[0]["motor"]
        }
        if samples[0]["motor"] is not None
        else None
    )
    return result


def run_simulation(config, mode="balance"):
    """Run a complete undisturbed experiment for saved results or tests."""
    return samples_to_result(list(simulation_steps(config, mode)), mode)
