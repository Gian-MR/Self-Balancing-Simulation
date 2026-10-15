"""Animate recorded physics and plot the quantities used by the controller."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import Button

from .simulation import run_simulation


def create_figure(result, config, fig=None, show_controls=False):
    """Build a figure and a function that draws a chosen recorded sample."""
    time, state = result["time"], result["state"]
    length = config["physics"]["stick_length"]
    if fig is None:
        fig = plt.figure(figsize=(11, 7), constrained_layout=True)
    else:
        # Reuse the same window when a mode button starts a new run.
        fig.clear()
    rows = 4 if show_controls else 3
    heights = [1, 1, 1, 0.28] if show_controls else [1, 1, 1]
    grid = fig.add_gridspec(rows, 2, width_ratios=[1.3, 1], height_ratios=heights)
    scene = fig.add_subplot(grid[:3, 0])
    axes = [fig.add_subplot(grid[i, 1]) for i in range(3)]
    fig.suptitle(f"Self-balancing stick — {result['mode']} mode")

    # Include the entire recorded cart path in the viewport.
    scene.set_xlim(state[:, 0].min() - length - 0.4,
                   state[:, 0].max() + length + 0.4)
    scene.set_ylim(-length - 0.35 if result["mode"] == "swing-up" else -0.35,
                   length + 0.35)
    scene.set_aspect("equal")
    scene.set_xlabel("Horizontal position (m)")
    scene.set_ylabel("Height above pivot (m)")
    # This line represents the elevated guide rail, not a collision floor.
    scene.axhline(-0.2, color="gray")
    cart = Rectangle((0, -0.2), 0.35, 0.2, color="steelblue")
    scene.add_patch(cart)
    rod, = scene.plot([], [], color="black", linewidth=3)
    weight = Circle((0, length), radius=0.07, color="darkorange")
    scene.add_patch(weight)
    label = scene.text(0.03, 0.97, "", transform=scene.transAxes, va="top")
    scene.set_title("Blue: cart | Orange: top mass")

    angle_errors = (state[:, 2] + np.pi) % (2 * np.pi) - np.pi
    plotted_angles = np.rad2deg(angle_errors)
    # Do not connect +180 to -180 with a misleading vertical line on the plot.
    wrap_points = np.flatnonzero(abs(np.diff(angle_errors)) > np.pi) + 1
    plotted_angles[wrap_points] = np.nan
    signals = [plotted_angles, state[:, 0], result["force"]]
    titles = ["Stick angle (degrees)", "Cart position (m)", "Motor force (N)"]
    cursors = []
    for ax, signal, title in zip(axes, signals, titles):
        ax.plot(time, signal)
        ax.axhline(0, color="gray", linewidth=0.6)
        ax.set_ylabel(title)
        ax.grid(alpha=0.25)
        cursors.append(ax.axvline(0, color="darkorange", linewidth=1))
    if result.get("motor") is not None:
        axes[-1].plot(time, result["requested_force"], linestyle="--", alpha=0.7,
                      label="Requested")
        axes[-1].lines[0].set_label("Actual")
        axes[-1].legend(loc="best")
    axes[-1].set_xlabel("Simulation time (s)")

    def draw(index) -> list[Artist]:
        position, _, angle, _ = state[index]
        # Angle is measured from vertical, so horizontal uses sin(angle).
        tip_x = position + length * np.sin(angle)
        tip_y = length * np.cos(angle)
        cart.set_x(position - 0.35 / 2)
        rod.set_data([position, tip_x], [0, tip_y])
        weight.center = (tip_x, tip_y)
        label.set_text(f"t = {time[index]:.2f} s\n"
                       f"angle = {np.rad2deg(angle_errors[index]):.2f}°")
        if result["mode"] == "swing-up":
            label.set_text(label.get_text() + f"\nController: {result['phase'][index]}")
        if result.get("motor") is not None:
            motor = result["motor"]
            label.set_text(label.get_text() +
                           f"\nOutput shaft: {motor['output_rpm'][index]:.0f} RPM"
                           f"\nCurrent: {motor['current'][index]:.2f} A"
                           f"\nTerminal voltage: {motor['terminal_voltage'][index]:.1f} V")
        for cursor in cursors:
            cursor.set_xdata([time[index], time[index]])

        # Matplotlib's callback type expects the objects updated in this frame.
        return [cart, rod, weight, label, *cursors]

    draw(0)
    return fig, draw


def save_summary(result, config, path):
    """Save a static final scene and complete time-history plots."""
    fig, draw = create_figure(result, config)
    draw(len(result["time"]) - 1)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def show_animation(result, config):
    """Play a run, with buttons for swing-up, balance, and fall experiments."""
    fig = plt.figure(figsize=(11, 7), constrained_layout=True)
    # Keep animations and buttons alive while the window is open.
    active = {"animation": None, "buttons": []}
    recordings = {result["mode"]: result}

    def start_mode(mode):
        previous = active["animation"]
        if previous is not None and previous.event_source is not None:
            previous.pause()

        # Settings stay the same; every button restarts from the initial tilt.
        if mode not in recordings:
            recordings[mode] = run_simulation(config, mode)
        recording = recordings[mode]
        _, draw = create_figure(recording, config, fig=fig, show_controls=True)
        subplot = fig.axes[0].get_subplotspec()
        # Some Matplotlib axes have no subplot layout. Check before using it.
        
        if subplot is None:
            raise RuntimeError("The simulation axes need a subplot layout.")
        grid = subplot.get_gridspec()
        button_grid = grid[3, :].subgridspec(1, 3)
        buttons = []
        for column, (button_mode, label) in enumerate([
            ("swing-up", "Restart: Swing up"),
            ("balance", "Restart: Balance"),
            ("fall", "Restart: Fall (motor off)"),
        ]):
            button = Button(fig.add_subplot(button_grid[0, column]), label)
            # Capture this iteration's mode so each button keeps its own action.
            button.on_clicked(lambda event, selected_mode=button_mode: start_mode(selected_mode))
            buttons.append(button)
        active["buttons"] = buttons

        # Rendering at 30 fps is separate from the 100 Hz control updates.
        frame_times = np.arange(0, recording["time"][-1], 1 / 30)
        frames = np.searchsorted(recording["time"], frame_times)
        frames = np.unique(np.append(frames, len(recording["time"]) - 1))
        active["animation"] = FuncAnimation(
            fig, draw, frames=frames, interval=1000 / 30,
            repeat=False, cache_frame_data=False,
        )
        # Draw immediately so each new animation starts after a button click.
        fig.canvas.draw()

    start_mode(result["mode"])
    plt.show()
    return active["animation"]
