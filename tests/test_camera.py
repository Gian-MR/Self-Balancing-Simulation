"""Camera motion must be gradual and independent of the physics state."""

from pathlib import Path
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.interaction import NudgeBox
from src.simulation import samples_to_result, simulation_steps
from src.visualization import create_figure


class CameraTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "config/parameters.yaml"
        self.config = yaml.safe_load(path.read_text())

    def tearDown(self):
        plt.close("all")

    def test_reset_box_beside_distant_cart_without_fake_velocity(self):
        box = NudgeBox(self.config)
        for position in (-20.0, 30.0):
            box.velocity[:] = [1, 1]
            box.reset(cart_position=position)
            np.testing.assert_allclose(box.center, box.home + [position, 0])
            box.update_velocity(1 / 30)
            np.testing.assert_array_equal(box.velocity, [0, 0])
            state = np.array([position, 0, 0, 0])
            np.testing.assert_array_equal(box.force_on_tip(state, self.config["physics"]), [0, 0])

    def test_camera_eases_both_directions_and_keeps_scale(self):
        for direction in (-1, 1):
            first = next(simulation_steps(self.config, "balance"))
            result = samples_to_result([first], "balance")
            result["live"] = True
            fig, draw = create_figure(result, self.config)
            scene = fig.axes[0]
            width = np.ptp(scene.get_xlim())
            result["state"][0, 0] = direction
            result["time"][0] = 1 / 30
            draw(0)
            center = np.mean(scene.get_xlim())
            self.assertGreater(direction * center, 0)
            self.assertLess(direction * center, 0.2)
            # Redrawing the same sample must not advance the camera again.
            draw(0)
            self.assertAlmostEqual(np.mean(scene.get_xlim()), center)
            for frame in range(2, 91):
                result["time"][0] = frame / 30
                draw(0)
            self.assertAlmostEqual(np.mean(scene.get_xlim()), direction, places=4)
            self.assertAlmostEqual(np.ptp(scene.get_xlim()), width)
            self.assertEqual(result["state"][0, 0], direction)
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
