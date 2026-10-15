"""Behavior checks: run with python -m unittest discover -s tests -v."""

import copy
from pathlib import Path
import unittest

import numpy as np
from scipy.integrate import solve_ivp
import yaml

from src.physics import derivatives
from src.simulation import run_simulation


class SimulationTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "config/parameters.yaml"
        self.config = yaml.safe_load(path.read_text())
        # These original tests isolate cart/stick physics with an ideal actuator.
        # Motor behavior is tested separately in test_motor.py.
        self.config["motor"]["enabled"] = False

    def test_uncontrolled_stick_falls(self):
        result = run_simulation(self.config, "fall")
        self.assertLess(result["time"][-1], self.config["simulation"]["duration"])
        self.assertAlmostEqual(np.rad2deg(result["state"][-1, 2]), 80, places=5)
        np.testing.assert_array_equal(result["force"], 0)

    def test_controller_recovers_and_respects_force_limit(self):
        result = run_simulation(self.config)
        self.assertAlmostEqual(result["time"][-1], self.config["simulation"]["duration"])
        self.assertLess(abs(result["state"][-1, 2]), np.deg2rad(0.1))
        self.assertLess(abs(result["state"][-1, 0]), 0.02)
        self.assertLessEqual(abs(result["force"]).max(), 15)

    def test_smaller_control_interval_gives_similar_trajectory(self):
        original = run_simulation(self.config)
        finer_config = copy.deepcopy(self.config)
        finer_config["simulation"]["control_interval"] /= 2
        finer = run_simulation(finer_config)
        for column in (0, 2):
            interpolated = np.interp(original["time"], finer["time"],
                                     finer["state"][:, column])
            self.assertLess(abs(original["state"][:, column] - interpolated).max(), 0.02)

    def test_no_friction_or_motor_conserves_energy(self):
        p = self.config["physics"].copy()
        p["cart_friction"] = p["pivot_friction"] = 0
        solution = solve_ivp(lambda t, y: derivatives(t, y, 0, p),
                             (0, 2), [0, 0, 0.15, 0],
                             rtol=1e-10, atol=1e-12, max_step=0.01)
        self.assertTrue(solution.success)
        _, v, theta, omega = solution.y
        length = p["stick_length"]
        total = p["cart_mass"] + p["stick_mass"] + p["top_mass"]
        h = (p["stick_mass"] / 2 + p["top_mass"]) * length
        inertia = (p["stick_mass"] / 3 + p["top_mass"]) * length**2
        energy = (total * v**2 / 2 + h * np.cos(theta) * v * omega
                  + inertia * omega**2 / 2 + h * p["gravity"] * np.cos(theta))
        self.assertLess(np.ptp(energy), 1e-7)


if __name__ == "__main__":
    unittest.main()
