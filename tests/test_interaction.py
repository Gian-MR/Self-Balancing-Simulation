"""Contact mechanics and live mouse interaction must change the actual physics."""

from pathlib import Path
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backend_bases import MouseEvent
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, Rectangle
import numpy as np
import yaml

from src.interaction import NudgeBox
from src.physics import derivatives
from src.simulation import run_simulation, simulation_steps
from src.visualization import show_animation


class InteractionTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "config/parameters.yaml"
        self.config = yaml.safe_load(path.read_text())
        self.config["simulation"]["initial_angle_degrees"] = 0

    def tearDown(self):
        plt.close("all")

    def test_contact_direction_no_contact_and_force_limit(self):
        box = NudgeBox(self.config)
        state = np.zeros(4)
        physics = self.config["physics"]
        np.testing.assert_array_equal(box.force_on_tip(state, physics), [0, 0])
        touching_distance = box.width / 2 + box.tip_radius / 2
        for offset in (-touching_distance, touching_distance):
            box.center[:] = [offset, physics["stick_length"]]
            force = box.force_on_tip(state, physics)
            self.assertGreater(force[0] * -offset, 0)
            self.assertEqual(force[1], 0)
            self.assertLessEqual(np.linalg.norm(force), box.max_force)
        box.center[:] = [0, physics["stick_length"]]
        self.assertTrue(np.all(np.isfinite(box.force_on_tip(state, physics))))

    def test_tip_force_changes_rotation_and_matches_work(self):
        state = np.array([0., .2, .3, .4])
        physics = self.config["physics"]
        tip_force = np.array([.5, -.2])
        without = derivatives(0, state, 0, physics)
        with_push = derivatives(0, state, 0, physics, tip_force)
        self.assertGreater(np.linalg.norm(with_push - without), 0)
        length = physics["stick_length"]
        tip_velocity = np.array([state[1] + length * np.cos(state[2]) * state[3],
                                 -length * np.sin(state[2]) * state[3]])
        torque = length * (np.cos(state[2]) * tip_force[0] - np.sin(state[2]) * tip_force[1])
        self.assertAlmostEqual(np.dot(tip_force, tip_velocity),
                               tip_force[0] * state[1] + torque * state[3])

    def test_small_push_then_release_and_recovery(self):
        box = NudgeBox(self.config)
        box.max_force = .05
        box.center[:] = [-(box.width / 2 + box.tip_radius / 2),
                         self.config["physics"]["stick_length"]]
        steps = simulation_steps(self.config, "balance", box, live=True)
        samples = [next(steps) for _ in range(11)]
        # A small real ball can leave contact before the final sample.
        self.assertGreater(max(s["state"][2] for s in samples), 1e-4)
        self.assertGreater(max(s["tip_force"][0] for s in samples), 0)
        box.center[:] = [-10, 10]
        for _ in range(500):
            final = next(steps)
        self.assertLess(abs(final["state"][2]), np.deg2rad(.1))
        np.testing.assert_array_equal(final["tip_force"], [0, 0])

    def test_live_continues_after_duration_and_buttons_drag_and_reset(self):
        self.config["simulation"]["duration"] = .05
        created = []

        def capture_animation(*args, **kwargs):
            animation = FuncAnimation(*args, **kwargs)
            created.append(animation)
            return animation

        def send(fig, name, xy):
            x, y = fig.axes[0].transData.transform(xy)
            event = MouseEvent(name, fig.canvas, x, y, button=1)
            fig.canvas.callbacks.process(name, event)

        def click_button(fig, text):
            ax = next(ax for ax in fig.axes if any(t.get_text() == text for t in ax.texts))
            x, y = ax.transAxes.transform((.5, .5))
            for name in ("button_press_event", "button_release_event"):
                fig.canvas.callbacks.process(name, MouseEvent(name, fig.canvas, x, y, button=1))

        def interact():
            fig = plt.gcf()
            scene = fig.axes[0]
            box = next(p for p in scene.patches
                       if isinstance(p, Rectangle) and p.get_facecolor() == to_rgba("mediumpurple"))
            before = np.array(box.get_xy())
            center = before + [box.get_width() / 2, box.get_height() / 2]
            send(fig, "button_press_event", center)
            contact_x = -(box.get_width() / 2 + self.config["geometry"]["top_diameter"] / 4)
            send(fig, "motion_notify_event", [contact_x, 1.0])
            send(fig, "button_release_event", [contact_x, 1.0])
            self.assertGreater(np.linalg.norm(np.array(box.get_xy()) - before), .5)
            for i in range(5):
                created[-1]._func(i)
            # The steel ball's position must move in response to contact.
            ball = next(p for p in scene.patches if isinstance(p, Circle)
                        and np.isclose(p.radius, self.config["geometry"]["top_diameter"] / 2))
            self.assertGreater(abs(ball.center[0]), .001)
            text = scene.texts[0].get_text()
            self.assertIn("t = 0.17", text)  # Continues beyond configured 0.05 s.
            cart = next(p for p in scene.patches if isinstance(p, Rectangle)
                        and p is not box)
            cart_position = cart.get_x() + cart.get_width() / 2
            click_button(fig, "Reset box")
            np.testing.assert_allclose(box.get_xy(), before + [cart_position, 0])
            for mode, label in [("swing-up", "Restart: Swing up"),
                                ("fall", "Restart: Fall (motor off)"),
                                ("balance", "Restart: Balance")]:
                click_button(fig, label)
                self.assertTrue(fig._suptitle.get_text().endswith(f"{mode} mode"))
                created[-1]._func(0)
            for animation in created:
                animation.pause()

        with patch("src.visualization.FuncAnimation", side_effect=capture_animation), \
             patch("matplotlib.pyplot.show", side_effect=interact):
            show_animation(run_simulation(self.config), self.config)


if __name__ == "__main__":
    unittest.main()
