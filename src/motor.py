"""A simple, steady-current model of the Pololu #4698 DC gearmotor.

Voltage -> current -> torque -> pulley force. Motor inductance, rotating inertia,
backlash, and temperature are omitted in this first approximation.
"""

import numpy as np


class DCMotor:
    def __init__(self, settings):
        # Published values and our mechanism/driver choices live in YAML.
        for key in ("nominal_voltage", "supply_voltage", "stall_current",
                    "no_load_current", "no_load_output_rpm", "stall_output_torque",
                    "gear_ratio", "pulley_radius", "current_limit"):
            value = settings[key]
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"motor.{key} must be finite and positive")
        if settings["no_load_current"] >= settings["stall_current"]:
            raise ValueError("Motor no-load current must be below stall current")

        self.supply_voltage = settings["supply_voltage"]
        self.current_limit = settings["current_limit"]
        self.gear_ratio = settings["gear_ratio"]
        self.radius = settings["pulley_radius"]

        # At stall the shaft isn't moving: V = R I, so R is approximately V/I.
        self.resistance = settings["nominal_voltage"] / settings["stall_current"]
        output_speed = settings["no_load_output_rpm"] * 2 * np.pi / 60
        rotor_speed = output_speed * self.gear_ratio
        # Spinning generates an opposing voltage (back EMF): E = Ke * omega.
        self.back_emf_constant = (settings["nominal_voltage"] - self.resistance * settings["no_load_current"]) / rotor_speed
        # In SI units a DC motor's torque constant equals its back-EMF constant.
        self.torque_constant = self.back_emf_constant
        # Approximate no-load losses as viscous friction: torque_loss = B omega.
        self.viscous_friction = (self.torque_constant * settings["no_load_current"] / rotor_speed)
        # Fit an effective gearbox efficiency to the published output stall torque.
        # This is an estimate, not a manufacturer-specified efficiency.
        self.efficiency = settings["stall_output_torque"] / (self.gear_ratio * self.torque_constant * settings["stall_current"])
        if self.back_emf_constant <= 0 or not 0 < self.efficiency <= 1:
            raise ValueError("Motor reference specifications are inconsistent")

    def voltage_for_force(self, requested_force, measured_cart_speed):
        """
        Translate LQR force into voltage using estimated speed compensation.

        This feedforward conversion is not an actual current-feedback controller.
        Voltage/current limits can prevent the requested force from being reached.
        """
        output_speed = measured_cart_speed / self.radius
        rotor_speed = output_speed * self.gear_ratio
        output_torque = requested_force * self.radius
        # Reverse the gearbox relation used below, including braking operation.
        if output_torque * output_speed >= 0:
            rotor_torque = output_torque / (self.gear_ratio * self.efficiency)
        else:
            rotor_torque = output_torque * self.efficiency / self.gear_ratio
        current = (rotor_torque + self.viscous_friction * rotor_speed) / self.torque_constant
        current = np.clip(current, -self.current_limit, self.current_limit)
        voltage = self.resistance * current + self.back_emf_constant * rotor_speed
        return float(np.clip(voltage, -self.supply_voltage, self.supply_voltage))

    def evaluate(self, command_voltage, cart_speed, coast=False):
        """
        Calculate instantaneous motor behavior at the actual cart speed.

        Assume a rigid belt with no slip: output angular speed = cart speed/radius.
        The ideal driver limits current and can handle regenerative braking.
        In coast mode the motor circuit is open, rather than shorted at zero volts.
        """
        output_speed = cart_speed / self.radius
        rotor_speed = output_speed * self.gear_ratio
        voltage = float(np.clip(command_voltage, -self.supply_voltage, self.supply_voltage))
        back_emf = self.back_emf_constant * rotor_speed
        if coast:
            current = 0.0
            terminal_voltage = back_emf  # Open-circuit generated voltage.
        else:
            current = float(np.clip((voltage - back_emf) / self.resistance, -self.current_limit, self.current_limit))
            # When limiting current, the ideal driver adjusts terminal voltage.
            terminal_voltage = self.resistance * current + back_emf
            # Outside this envelope the ideal driver assumption is insufficient.
            if abs(terminal_voltage) > self.supply_voltage + 1e-8:
                raise ValueError("Motor overspeed exceeds the current-limited driver model")
        rotor_torque = self.torque_constant * current - self.viscous_friction * rotor_speed
        # Losses reduce delivered torque when driving. When back-driven, they
        # increase the resisting torque needed to generate the same rotor torque.
        if rotor_torque * rotor_speed >= 0:
            output_torque = rotor_torque * self.gear_ratio * self.efficiency
        else:
            output_torque = rotor_torque * self.gear_ratio / self.efficiency
        return {
            "force": output_torque / self.radius,
            "command_voltage": voltage, "terminal_voltage": terminal_voltage,
            "current": current, "output_torque": output_torque,
            "output_rpm": output_speed * 60 / (2 * np.pi),
            "rotor_rpm": rotor_speed * 60 / (2 * np.pi),
        }
