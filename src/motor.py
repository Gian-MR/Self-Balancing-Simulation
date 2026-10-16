"""A simple, steady-current model of the Pololu #4698 DC gearmotor.

Voltage -> current -> torque -> pulley force. Motor inductance, rotating inertia,
backlash, and temperature are omitted in this first approximation.
"""

import numpy as np


class DCMotor:
    def __init__(self, motor_settings):
        # Published values and our mechanism/driver choices live in YAML.
        for setting_name in (
            "nominal_voltage",
            "supply_voltage",
            "stall_current",
            "no_load_current",
            "no_load_output_rpm",
            "stall_output_torque",
            "gear_ratio",
            "pulley_radius",
            "current_limit",
        ):
            setting_value = motor_settings[setting_name]
            if not np.isfinite(setting_value) or setting_value <= 0:
                raise ValueError(f"motor.{setting_name} must be finite and positive")
        if motor_settings["no_load_current"] >= motor_settings["stall_current"]:
            raise ValueError("Motor no-load current must be below stall current")

        self.motor_supply_voltage = motor_settings["supply_voltage"]
        self.motor_current_limit = motor_settings["current_limit"]
        self.gearbox_ratio = motor_settings["gear_ratio"]
        self.pulley_radius = motor_settings["pulley_radius"]

        # At stall the shaft isn't moving: V = R I, so R is approximately V/I.
        self.motor_resistance = motor_settings["nominal_voltage"] / motor_settings["stall_current"]
        output_angular_velocity = motor_settings["no_load_output_rpm"] * 2 * np.pi / 60
        rotor_angular_velocity = output_angular_velocity * self.gearbox_ratio
        # Spinning generates an opposing voltage (back EMF): E = Ke * omega.
        self.motor_back_emf_constant = (
            motor_settings["nominal_voltage"]
            - self.motor_resistance * motor_settings["no_load_current"]
        ) / rotor_angular_velocity
        # In SI units a DC motor's torque constant equals its back-EMF constant.
        self.motor_torque_constant = self.motor_back_emf_constant
        # Approximate no-load losses as viscous friction: torque_loss = B omega.
        self.motor_viscous_friction = (
            self.motor_torque_constant * motor_settings["no_load_current"] / rotor_angular_velocity
        )
        # Fit an effective gearbox efficiency to the published output stall torque.
        # This is an estimate, not a manufacturer-specified efficiency.
        self.gearbox_efficiency = motor_settings["stall_output_torque"] / (
            self.gearbox_ratio * self.motor_torque_constant * motor_settings["stall_current"]
        )
        if self.motor_back_emf_constant <= 0 or not 0 < self.gearbox_efficiency <= 1:
            raise ValueError("Motor reference specifications are inconsistent")

    def voltage_for_force(self, requested_cart_force, measured_cart_velocity):
        """
        Translate LQR force into voltage using estimated speed compensation.

        This feedforward conversion is not an actual current-feedback controller.
        Voltage/current limits can prevent the requested force from being reached.
        """
        output_angular_velocity = measured_cart_velocity / self.pulley_radius
        rotor_angular_velocity = output_angular_velocity * self.gearbox_ratio
        output_shaft_torque = requested_cart_force * self.pulley_radius
        # Reverse the gearbox relation used below, including braking operation.
        if output_shaft_torque * output_angular_velocity >= 0:
            rotor_shaft_torque = output_shaft_torque / (
                self.gearbox_ratio * self.gearbox_efficiency
            )
        else:
            rotor_shaft_torque = output_shaft_torque * self.gearbox_efficiency / self.gearbox_ratio
        motor_current = (
            rotor_shaft_torque + self.motor_viscous_friction * rotor_angular_velocity
        ) / self.motor_torque_constant
        motor_current = np.clip(motor_current, -self.motor_current_limit, self.motor_current_limit)
        motor_command_voltage = (
            self.motor_resistance * motor_current
            + self.motor_back_emf_constant * rotor_angular_velocity
        )
        return float(
            np.clip(motor_command_voltage, -self.motor_supply_voltage, self.motor_supply_voltage)
        )

    def evaluate(self, motor_command_voltage, cart_velocity, coast=False):
        """
        Calculate instantaneous motor behavior at the actual cart speed.

        Assume a rigid belt with no slip: output angular speed = cart speed/radius.
        The ideal driver limits current and can handle regenerative braking.
        In coast mode the motor circuit is open, rather than shorted at zero volts.
        """
        output_angular_velocity = cart_velocity / self.pulley_radius
        rotor_angular_velocity = output_angular_velocity * self.gearbox_ratio
        motor_command_voltage = float(
            np.clip(motor_command_voltage, -self.motor_supply_voltage, self.motor_supply_voltage)
        )
        motor_back_emf_voltage = self.motor_back_emf_constant * rotor_angular_velocity
        if coast:
            motor_current = 0.0
            motor_terminal_voltage = motor_back_emf_voltage  # Open-circuit generated voltage.
        else:
            motor_current = float(
                np.clip(
                    (motor_command_voltage - motor_back_emf_voltage) / self.motor_resistance,
                    -self.motor_current_limit,
                    self.motor_current_limit,
                )
            )
            # When limiting current, the ideal driver adjusts terminal voltage.
            motor_terminal_voltage = self.motor_resistance * motor_current + motor_back_emf_voltage
            # Outside this envelope the ideal driver assumption is insufficient.
            if abs(motor_terminal_voltage) > self.motor_supply_voltage + 1e-8:
                raise ValueError("Motor overspeed exceeds the current-limited driver model")
        rotor_shaft_torque = (
            self.motor_torque_constant * motor_current
            - self.motor_viscous_friction * rotor_angular_velocity
        )
        # Losses reduce delivered torque when driving. When back-driven, they
        # increase the resisting torque needed to generate the same rotor torque.
        if rotor_shaft_torque * rotor_angular_velocity >= 0:
            output_shaft_torque = rotor_shaft_torque * self.gearbox_ratio * self.gearbox_efficiency
        else:
            output_shaft_torque = rotor_shaft_torque * self.gearbox_ratio / self.gearbox_efficiency
        return {
            "force": output_shaft_torque / self.pulley_radius,
            "command_voltage": motor_command_voltage,
            "terminal_voltage": motor_terminal_voltage,
            "current": motor_current,
            "output_torque": output_shaft_torque,
            "output_rpm": output_angular_velocity * 60 / (2 * np.pi),
            "rotor_rpm": rotor_angular_velocity * 60 / (2 * np.pi),
        }
