"""Shared drawing/contact dimensions in meters, separate from mass settings."""

import math


def dimensions(config=None):
    """Use measured sizes where known and editable assumptions otherwise."""
    sizes = {
        "top_diameter": 0.067,   # Representative tennis-ball diameter.
        "rod_diameter": 0.006,   # Assumed; replace with a measured diameter.
        "cart_width": 0.10,     # Assumed carriage envelope, not motor specs.
        "cart_height": 0.045,
        "box_size": 0.04,       # Size of the draggable square obstacle.
    }
    if config is not None:
        sizes.update(config.get("geometry", {}))
    for name, value in sizes.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"geometry.{name} must be finite and positive")
    return sizes
