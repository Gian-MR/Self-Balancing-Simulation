"""Check motor reference points, limits, and interaction with the cart."""

import copy
from pathlib import Path
import unittest

import numpy as np
import yaml

from src.motor import DCMotor
from src.simulation import run_simulation


class MotorTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "config/parameters.yaml"
        self.config = yaml.safe_load(path.read_text())
        self.settings = self.config["motor"].copy()

    def test_published_stall_point_without_conservative_current_limit(self):
        # Test the short-duration reference point, not a safe continuous load.
        self.settings["current_limit"] = self.settings["stall_current"]
        motor = DCMotor(self.settings)
        reading = motor.evaluate(self.settings["nominal_voltage"], 0)
        self.assertAlmostEqual(reading["current"], self.settings["stall_current"])
        self.assertAlmostEqual(reading["output_torque"], self.settings["stall_output_torque"])

    def test_published_no_load_point(self):
        motor = DCMotor(self.settings)
        speed = self.settings["no_load_output_rpm"] * 2 * np.pi / 60 * motor.pulley_radius
        reading = motor.evaluate(self.settings["nominal_voltage"], speed)
        self.assertAlmostEqual(reading["current"], self.settings["no_load_current"])
        self.assertAlmostEqual(reading["output_torque"], 0)
        self.assertAlmostEqual(reading["rotor_rpm"], reading["output_rpm"] * motor.gearbox_ratio)

    def test_back_emf_reduces_current_at_fixed_voltage(self):
        self.settings["current_limit"] = self.settings["stall_current"]
        motor = DCMotor(self.settings)
        self.assertGreater(motor.evaluate(12, 0)["current"],
                           motor.evaluate(12, 0.3)["current"])

    def test_force_conversion_driving_and_braking_both_directions(self):
        motor = DCMotor(self.settings)
        for speed in (-0.1, 0, 0.1):
            for force in (-2, 2):
                voltage = motor.voltage_for_force(force, speed)
                self.assertAlmostEqual(motor.evaluate(voltage, speed)["force"], force)

    def test_current_limit_and_open_circuit_coast(self):
        motor = DCMotor(self.settings)
        reading = motor.evaluate(100, 0)
        self.assertEqual(reading["command_voltage"], motor.motor_supply_voltage)
        self.assertAlmostEqual(reading["current"], motor.motor_current_limit)
        coast = motor.evaluate(0, 0.1, coast=True)
        self.assertEqual(coast["current"], 0)
        self.assertLess(coast["force"], 0)  # Mechanical drag remains.
        self.assertGreater(coast["terminal_voltage"], 0)
        self.assertLess(motor.evaluate(0, 0.1)["force"], coast["force"])

    def test_simulation_balances_small_tilt_and_logs_actual_force(self):
        config = copy.deepcopy(self.config)
        config["motor"]["enabled"] = True
        config["physics"]["cart_mass"] = 1.0
        config["simulation"]["initial_angle_degrees"] = 8.0
        config["simulation"]["duration"] = 8.0
        result = run_simulation(config)
        self.assertAlmostEqual(result["time"][-1], 8)
        self.assertLess(abs(result["state"][-1, 2]), np.deg2rad(0.1))
        motor = DCMotor(config["motor"])
        self.assertLessEqual(abs(result["motor"]["current"]).max(), motor.motor_current_limit)
        self.assertLessEqual(abs(result["motor"]["terminal_voltage"]).max(), motor.motor_supply_voltage)
        for i in range(len(result["time"])):
            reading = motor.evaluate(result["motor"]["command_voltage"][i], result["state"][i, 1])
            self.assertAlmostEqual(reading["force"], result["force"][i])

    def test_heavy_cart_large_tilt_can_exceed_motor_capability(self):
        config = copy.deepcopy(self.config)
        config["motor"]["enabled"] = True
        config["physics"]["cart_mass"] = 5.0
        config["simulation"]["initial_angle_degrees"] = 12.0
        config["simulation"]["duration"] = 8.0
        result = run_simulation(config)
        self.assertIn("fall angle", result["reason"])
        self.assertLess(result["time"][-1], 8)
        self.assertGreater(np.max(abs(result["requested_force"] - result["force"])), 1)

    def test_invalid_motor_settings(self):
        for key, value in [("pulley_radius", 0), ("current_limit", -1),
                           ("gear_ratio", np.nan), ("no_load_current", 4)]:
            settings = self.settings.copy()
            settings[key] = value
            with self.assertRaises(ValueError):
                DCMotor(settings)


if __name__ == "__main__":
    unittest.main()
