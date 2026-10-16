"""Soft contact between a movable box and the orange weight at the stick tip."""

import numpy as np


class NudgeBox:
    """A mouse-positioned spring/damper obstacle with a limited pushing force."""

    stiffness = 80.0  # N/m of overlap.
    damping = 1.5  # N per m/s of relative closing speed.
    max_force = 3.0  # N; dragging never directly changes the stick's state.

    def __init__(self, config):
        geometry_settings = config["geometry"]
        # The visible ball and its contact boundary use the SAME dimensions.
        self.width = self.height = geometry_settings["box_size"]
        self.tip_radius = geometry_settings["top_diameter"] / 2
        stick_length = config["physics"]["stick_length"]
        self.home = np.array([-0.35 * stick_length, 0.25 * stick_length])
        self.reset()

    def reset(self, cart_position=0.0):
        """Park beside the current cart, clearing any previous drag velocity."""
        self.center = self.home + np.array([cart_position, 0.0])
        self.velocity = np.zeros(2)
        self._previous_center = self.center.copy()

    @property
    def lower_left(self) -> tuple[float, float]:
        """Matplotlib expects an (x, y) tuple, rather than a NumPy array."""
        return (float(self.center[0] - self.width / 2), float(self.center[1] - self.height / 2))

    def update_velocity(self, frame_interval):
        """Estimate box velocity once per displayed frame, then hold it fixed."""
        box_velocity = (self.center - self._previous_center) / frame_interval
        box_speed = np.linalg.norm(box_velocity)
        self.velocity = box_velocity * min(1.0, 2.0 / max(box_speed, 1e-12))
        self._previous_center = self.center.copy()

    def force_on_tip(self, system_state, physics):
        """Return the current contact force [horizontal, vertical] in newtons.

        Unlike an impulse diagnostic, this is recalculated from overlap and
        relative velocity whenever the physics solver asks for an acceleration.
        """
        cart_position, cart_velocity, stick_angle, stick_angular_velocity = system_state
        stick_length = physics["stick_length"]
        tip_position = np.array(
            [cart_position + stick_length * np.sin(stick_angle), stick_length * np.cos(stick_angle)]
        )
        tip_velocity = np.array(
            [
                cart_velocity + stick_length * np.cos(stick_angle) * stick_angular_velocity,
                -stick_length * np.sin(stick_angle) * stick_angular_velocity,
            ]
        )
        box_half_size = np.array([self.width, self.height]) / 2
        tip_relative_position = tip_position - self.center
        box_nearest_point = np.clip(tip_relative_position, -box_half_size, box_half_size)
        tip_separation = tip_relative_position - box_nearest_point
        tip_separation_distance = np.linalg.norm(tip_separation)
        if tip_separation_distance > self.tip_radius:
            return np.zeros(2)
        if tip_separation_distance > 1e-12:
            contact_normal = tip_separation / tip_separation_distance
            contact_overlap = self.tip_radius - tip_separation_distance
        else:
            # If dragged over the tip, push toward the nearest box face.
            box_face_distances = box_half_size - abs(tip_relative_position)
            contact_axis = int(np.argmin(box_face_distances))
            contact_normal = np.zeros(2)
            contact_normal[contact_axis] = 1.0 if tip_relative_position[contact_axis] >= 0 else -1.0
            contact_overlap = self.tip_radius + box_face_distances[contact_axis]
        contact_separation_velocity = np.dot(tip_velocity - self.velocity, contact_normal)
        contact_force_magnitude = np.clip(
            self.stiffness * contact_overlap - self.damping * contact_separation_velocity,
            0.0,
            self.max_force,
        )
        return contact_force_magnitude * contact_normal
