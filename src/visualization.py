"""Plot saved experiments and run an interactive simulation with a draggable box."""

from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import Button

from .interaction import NudgeBox
from .simulation import samples_to_result, simulation_steps


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
    requested_line = None
    if result.get("motor") is not None:
        requested_line, = axes[-1].plot(time, result["requested_force"], linestyle="--", alpha=0.7,
                                      label="Requested")
        axes[-1].lines[0].set_label("Actual")
        axes[-1].legend(loc="best")
    axes[-1].set_xlabel("Simulation time (s)")

    def draw(index) -> list[Artist]:
        # Live mode replaces these arrays as new physical samples arrive.
        time, state = result["time"], result["state"]
        angle_errors = (state[:, 2] + np.pi) % (2 * np.pi) - np.pi
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
        if result.get("live"):
            label.set_text(label.get_text() +
                           f"\nBox push: {np.linalg.norm(result['tip_force'][index]):.2f} N")
            plotted = np.rad2deg(angle_errors)
            plotted[np.flatnonzero(abs(np.diff(angle_errors)) > np.pi) + 1] = np.nan
            for ax, values in zip(axes, [plotted, state[:, 0], result["force"]]):
                ax.lines[0].set_data(time, values)
                ax.relim()
                ax.autoscale_view(scalex=False)
                ax.set_xlim(time[0], max(time[0] + 10, time[-1]))
            if requested_line is not None:
                requested_line.set_data(time, result["requested_force"])
                axes[-1].relim()
                axes[-1].autoscale_view(scalex=False)
            left, right = scene.get_xlim()
            scene.set_xlim(min(left, position - length - 0.4),
                           max(right, position + length + 0.4))
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
    """Run live physics so a draggable obstacle can push the orange weight."""
    fig = plt.figure(figsize=(11, 7), constrained_layout=True)
    # Retain widgets, callbacks, and the physics generator until the window closes.
    active = {"animation": None, "buttons": [], "drag_offset": None}

    def start_mode(mode):
        previous = active["animation"]
        if previous is not None and previous.event_source is not None:
            previous.pause()

        box = NudgeBox()
        steps = simulation_steps(config, mode, contact=box, live=True)
        first_sample = next(steps)
        # Keep at most 30 simulated seconds of history to bound UI memory use.
        history = deque([first_sample], maxlen=max(2, int(30 / config["simulation"]["control_interval"])))
        recording = samples_to_result(history, mode)
        recording["live"] = True
        _, draw = create_figure(recording, config, fig=fig, show_controls=True)
        scene = fig.axes[0]
        scene.set_title("Drag the purple box into the orange weight to nudge it", fontsize=10)
        scene.set_xlim(-2 * config["physics"]["stick_length"],
                       3 * config["physics"]["stick_length"])
        patch = Rectangle(box.lower_left,
                          box.width, box.height, facecolor="mediumpurple", edgecolor="indigo",
                          zorder=5)
        scene.add_patch(patch)
        active.update(box=box, patch=patch, scene=scene, drag_offset=None,
                      history=history, steps=steps, target_time=0.0)
        status = scene.text(0.03, 0.02, "Live physics — drag the purple box; move it away to release.",
                            transform=scene.transAxes, fontsize=8, va="bottom", wrap=True)
        subplot = fig.axes[0].get_subplotspec()
        # Some Matplotlib axes have no subplot layout. Check before using it.
        
        if subplot is None:
            raise RuntimeError("The simulation axes need a subplot layout.")
        grid = subplot.get_gridspec()
        button_grid = grid[3, :].subgridspec(1, 4)
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
        reset = Button(fig.add_subplot(button_grid[0, 3]), "Reset box")

        def reset_box(event):
            box.reset()
            active["drag_offset"] = None
            patch.set_xy(box.lower_left)
            patch.set_facecolor("mediumpurple")
            fig.canvas.draw_idle()

        reset.on_clicked(reset_box)
        active["buttons"].append(reset)

        def tick(frame) -> list[Artist]:
            # Advance 1/30 simulated second per frame, using the configured
            # control interval. Slow rendering slows playback, not the physics.
            box.update_velocity(1 / 30)
            active["target_time"] += 1 / 30
            try:
                while history[-1]["time"] < active["target_time"]:
                    sample = next(steps)
                    history.append(sample)
                    if "reason" in sample:
                        status.set_text(sample["reason"] + " — use a Restart button.")
                        current_animation = active["animation"]
                        if current_animation is not None:
                            current_animation.pause()
                        break
            except StopIteration:
                current_animation = active["animation"]
                if current_animation is not None:
                    current_animation.pause()
            except (ValueError, RuntimeError) as error:
                # A very strong nudge can exceed the motor model's envelope.
                # Keep the window and restart controls usable if this happens.
                status.set_text(f"Paused: {error}. Use a Restart button.")
                current_animation = active["animation"]
                if current_animation is not None:
                    current_animation.pause()
            recording.update(samples_to_result(history, mode))
            artists = draw(len(recording["time"]) - 1)
            patch.set_xy(box.lower_left)
            touching = np.linalg.norm(box.force_on_tip(history[-1]["state"], config["physics"])) > 0
            patch.set_facecolor("tomato" if touching else "mediumpurple")
            return [*artists, patch, status]

        active["animation"] = FuncAnimation(
            fig, tick, init_func=lambda: draw(len(recording["time"]) - 1), interval=1000 / 30,
            repeat=False, cache_frame_data=False,
        )
        # Draw immediately so each new animation starts after a button click.
        fig.canvas.draw()

    def press(event):
        if event.button != 1 or event.inaxes is not active.get("scene"):
            return
        toolbar = getattr(fig.canvas, "toolbar", None)
        if toolbar is not None and toolbar.mode:
            return  # Let the standard pan/zoom tools work normally.
        patch = active["patch"]
        contains, _ = patch.contains(event)
        if contains and event.xdata is not None and event.ydata is not None:
            active["drag_offset"] = active["box"].center - [event.xdata, event.ydata]

    def motion(event):
        if (active["drag_offset"] is None or event.inaxes is not active.get("scene")
                or event.xdata is None or event.ydata is None):
            return
        box = active["box"]
        box.center = np.array([event.xdata, event.ydata]) + active["drag_offset"]
        active["patch"].set_xy(box.lower_left)
        fig.canvas.draw_idle()

    def release(event):
        # Also release when the mouse leaves the scene or window before mouse-up.
        active["drag_offset"] = None

    fig.canvas.mpl_connect("button_press_event", press)
    fig.canvas.mpl_connect("motion_notify_event", motion)
    fig.canvas.mpl_connect("button_release_event", release)
    fig.canvas.mpl_connect("figure_leave_event", release)
    start_mode(result["mode"])
    plt.show()
    return active["animation"]
