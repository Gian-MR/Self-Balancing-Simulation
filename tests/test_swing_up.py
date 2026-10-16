"""Swing-up must start below, add energy, capture upright, and stay there."""

import copy
from pathlib import Path
import unittest

import numpy as np
import yaml

from src.controller import SwingUpController, upright_angle
from src.simulation import run_simulation


class SwingUpTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "config/parameters.yaml"
        self.config = yaml.safe_load(path.read_text())

    def test_starts_below_and_stays_balanced_after_capture(self):
        result = run_simulation(self.config, "swing-up")
        angle = upright_angle(result["state"][:, 2])
        self.assertAlmostEqual(abs(angle[0]), np.pi)
        np.testing.assert_array_equal(result["state"][0, [0, 1, 3]], 0)
        self.assertAlmostEqual(result["time"][-1], self.config["swing_up"]["duration"])
        self.assertTrue(np.any(result["phase"] == "balance"))
        # A brief upright crossing isn't sufficient: check the final five seconds.
        tail = result["time"] > result["time"][-1] - 5
        self.assertTrue(np.all(result["phase"][tail] == "balance"))
        self.assertLess(abs(angle[tail]).max(), np.deg2rad(1))
        self.assertLess(abs(result["state"][tail, 1]).max(), 0.05)
        self.assertLessEqual(abs(result["requested_force"]).max(), self.config["controller"]["max_force"])
        self.assertLessEqual(abs(result["motor"]["current"]).max(), self.config["motor"]["current_limit"])

    def test_capture_also_works_with_twice_the_control_frequency(self):
        config = copy.deepcopy(self.config)
        config["simulation"]["control_interval"] /= 2
        result = run_simulation(config, "swing-up")
        self.assertEqual(result["phase"][-1], "balance")
        self.assertLess(abs(upright_angle(result["state"][-1, 2])), np.deg2rad(1))

    def test_angle_wrapping_capture_and_release(self):
        controller = SwingUpController(self.config["physics"], self.config["controller"],
                                       self.config["swing_up"])
        # Passing upright too quickly should not trigger capture.
        controller.force(np.array([0., 0., 2 * np.pi, 5.]), 1)
        self.assertEqual(controller.phase, "swing-up")
        controller.force(np.array([0.4, 0., 2 * np.pi, 0.]), 2)
        self.assertEqual(controller.phase, "balance")
        self.assertAlmostEqual(controller.cart_target_position, 0.4)
        controller.force(np.array([0.4, 0., np.deg2rad(25), 0.]), 3)
        self.assertEqual(controller.phase, "swing-up")

    def test_rejects_invalid_swing_settings(self):
        config = copy.deepcopy(self.config)
        config["swing_up"]["duration"] = 0
        with self.assertRaises(ValueError):
            run_simulation(config, "swing-up")


if __name__ == "__main__":
    unittest.main()
