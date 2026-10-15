"""Soft contact between a movable box and the orange weight at the stick tip."""

import numpy as np

from .geometry import dimensions


class NudgeBox:
    """A mouse-positioned spring/damper obstacle with a limited pushing force."""

    stiffness = 80.0  # N/m of overlap.
    damping = 1.5     # N per m/s of relative closing speed.
    max_force = 3.0   # N; dragging never directly changes the stick's state.

    def __init__(self, config=None):
        sizes = dimensions(config)
        # The visible ball and its contact boundary use the SAME dimensions.
        self.width = self.height = sizes["box_size"]
        self.tip_radius = sizes["top_diameter"] / 2
        length = config["physics"]["stick_length"] if config is not None else 1.0
        self.home = np.array([-0.35 * length, 0.25 * length])
        self.reset()

    def reset(self, cart_position=0.0):
        """Park beside the current cart, clearing any previous drag velocity."""
        self.center = self.home + np.array([cart_position, 0.0])
        self.velocity = np.zeros(2)
        self._previous_center = self.center.copy()

    @property
    def lower_left(self) -> tuple[float, float]:
        """Matplotlib expects an (x, y) tuple, rather than a NumPy array."""
        return (float(self.center[0] - self.width / 2),
                float(self.center[1] - self.height / 2))

    def update_velocity(self, simulation_interval):
        """Estimate box velocity once per displayed frame, then hold it fixed."""
        velocity = (self.center - self._previous_center) / simulation_interval
        speed = np.linalg.norm(velocity)
        self.velocity = velocity * min(1.0, 2.0 / max(speed, 1e-12))
        self._previous_center = self.center.copy()

    def force_on_tip(self, state, physics):
        """Return the current contact force [horizontal, vertical] in newtons.

        Unlike an impulse diagnostic, this is recalculated from overlap and
        relative velocity whenever the physics solver asks for an acceleration.
        """
        position, speed, angle, angular_speed = state
        length = physics["stick_length"]
        tip = np.array([position + length * np.sin(angle), length * np.cos(angle)])
        tip_velocity = np.array([speed + length * np.cos(angle) * angular_speed,
                                 -length * np.sin(angle) * angular_speed])
        half_size = np.array([self.width, self.height]) / 2
        relative_tip = tip - self.center
        nearest_point = np.clip(relative_tip, -half_size, half_size)
        delta = relative_tip - nearest_point
        distance = np.linalg.norm(delta)
        if distance > self.tip_radius:
            return np.zeros(2)
        if distance > 1e-12:
            normal = delta / distance
            overlap = self.tip_radius - distance
        else:
            # If dragged over the tip, push toward the nearest box face.
            distances = half_size - abs(relative_tip)
            axis = int(np.argmin(distances))
            normal = np.zeros(2)
            normal[axis] = 1.0 if relative_tip[axis] >= 0 else -1.0
            overlap = self.tip_radius + distances[axis]
        separation_speed = np.dot(tip_velocity - self.velocity, normal)
        magnitude = np.clip(self.stiffness * overlap - self.damping * separation_speed,
                            0.0, self.max_force)
        return magnitude * normal
