"""Plot saved experiments and run an interactive simulation with a draggable box."""

from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from matplotlib.patches import Circle, PathPatch, Polygon, Rectangle
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
from matplotlib.widgets import Button

from .interaction import NudgeBox
from .simulation import samples_to_result, simulation_steps


def tennis_ball(ax, center, radius, seam_width):
    """Draw a round tennis ball with two curved seams in physical coordinates."""
    ball = Circle(center, radius, facecolor="#c6df36", edgecolor="#657b20",
                  linewidth=0.6, zorder=7)
    ax.add_patch(ball)
    seams = []
    for sign in (-1, 1):
        # Cubic Bezier curves bend the seams inward across the ball's face.
        vertices = radius * np.array([
            [sign * 0.5, 0.866], [sign * 0.05, 0.35],
            [sign * 0.05, -0.35], [sign * 0.5, -0.866],
        ])
        path = Path(vertices, [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
        seam = PathPatch(path, facecolor="none", edgecolor="#fffde6",
                         linewidth=seam_width, zorder=8)
        seam.set_transform(Affine2D().translate(*center) + ax.transData)
        ax.add_patch(seam)
        seams.append(seam)
    return ball, seams


def create_figure(result, config, fig=None, show_controls=False):
    """Build a figure and a function that draws a chosen recorded sample."""
    time, state = result["time"], result["state"]
    length = config["physics"]["stick_length"]
    sizes = config["geometry"]
    ball_radius = sizes["top_diameter"] / 2
    rod_radius = sizes["rod_diameter"] / 2
    cart_width, cart_height = sizes["cart_width"], sizes["cart_height"]
    if fig is None:
        fig = plt.figure(figsize=(13, 8), constrained_layout=True)
    else:
        # Reuse the same window when a mode button starts a new run.
        fig.clear()
    rows = 4 if show_controls else 3
    heights = [1, 1, 1, 0.28] if show_controls else [1, 1, 1]
    grid = fig.add_gridspec(rows, 2, width_ratios=[1.3, 1], height_ratios=heights)
    scene = fig.add_subplot(grid[:3, 0])
    axes = [fig.add_subplot(grid[i, 1]) for i in range(3)]
    fig.suptitle(f"Self-balancing stick — {result['mode']} mode")

    # Equal axis scaling keeps one meter the same length in both directions.
    margin = 0.15 * length
    half_width = 1.2 * length if result["mode"] == "swing-up" else 0.65 * length
    center = (state[:, 0].min() + state[:, 0].max()) / 2
    half_width += np.ptp(state[:, 0]) / 2
    scene.set_xlim(center - half_width, center + half_width)
    scene.set_ylim(-length - margin if result["mode"] == "swing-up" else -0.25 * length,
                   length + margin)
    scene.set_aspect("equal")
    scene.set_xlabel("Horizontal position (m)")
    scene.set_ylabel("Height above pivot (m)")
    scene.set_facecolor("#f4f7fa")
    scene.grid(alpha=0.15)
    # Schematic rail: the pendulum can rotate in a plane in front of it.
    scene.axhline(-cart_height, color="#7b8b99", linewidth=2, zorder=1)
    cart = Rectangle((0, -cart_height), cart_width, cart_height,
                     facecolor="#426c8a", edgecolor="#203748", linewidth=0.7, zorder=4)
    scene.add_patch(cart)
    # A polygon gives the rod its actual width in meters, rather than pixels.
    rod = Polygon(np.zeros((4, 2)), closed=True, facecolor="#30383e",
                  edgecolor="none", zorder=5)
    scene.add_patch(rod)
    weight, ball_seams = tennis_ball(scene, (0, length), ball_radius, seam_width=0.6)
    pivot = Circle((0, 0), radius=min(rod_radius, cart_height / 5),
                   facecolor="#d2a758", edgecolor="none", zorder=8)
    scene.add_patch(pivot)
    label = scene.text(0.03, 0.97, "", transform=scene.transAxes, va="top", fontsize=8,
                      bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})
    scene.set_title(f"To scale: rod {length:g} m | ball Ø {sizes['top_diameter'] * 1000:g} mm",
                    fontsize=10)

    # A fixed physical scale reference helps judge the small parts.
    scene.plot([0.04, 0.04 + 0.1 / (2 * half_width)], [0.065, 0.065],
               transform=scene.transAxes, color="#203748", linewidth=2)
    scale_label = scene.text(0.04, 0.08, "100 mm", transform=scene.transAxes, fontsize=8)
    scale_bar = scene.lines[-1]

    # The ball is genuinely small in the main view. Magnify it separately;
    # never enlarge its collision boundary just to make it easier to see.
    detail = scene.inset_axes((0.68, 0.05, 0.29, 0.23))
    extent = max(2 * sizes["top_diameter"], 2 * sizes["rod_diameter"])
    detail.set_xlim(-extent, extent)
    detail.set_ylim(-extent, extent)
    detail.set_aspect("equal")
    detail.set_xticks([])
    detail.set_yticks([])
    detail.set_title("Tip detail (magnified)", fontsize=7)
    detail_rod = Polygon(np.zeros((4, 2)), facecolor="#30383e", edgecolor="none")
    detail.add_patch(detail_rod)
    tennis_ball(detail, (0, 0), ball_radius, seam_width=1.5)
    detail.text(0.5, 0.02, f"Ø {sizes['top_diameter'] * 1000:g} mm",
                transform=detail.transAxes, ha="center", fontsize=7)

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

    last_camera_time = float(time[0])

    def draw(index) -> list[Artist]:
        nonlocal last_camera_time
        # Live mode replaces these arrays as new physical samples arrive.
        time, state = result["time"], result["state"]
        angle_errors = (state[:, 2] + np.pi) % (2 * np.pi) - np.pi
        position, _, angle, _ = state[index]
        # Angle is measured from vertical, so horizontal uses sin(angle).
        tip_x = position + length * np.sin(angle)
        tip_y = length * np.cos(angle)
        cart.set_x(position - cart_width / 2)
        # Unit vector along the rod, and its perpendicular, in world coordinates.
        direction = np.array([np.sin(angle), np.cos(angle)])
        side = rod_radius * np.array([np.cos(angle), -np.sin(angle)])
        base = np.array([position, 0])
        tip = np.array([tip_x, tip_y])
        rod.set_xy([base + side, tip + side, tip - side, base - side])
        detail_base = -2 * extent * direction
        detail_rod.set_xy([detail_base + side, side, -side, detail_base - side])
        weight.center = (tip_x, tip_y)
        for seam in ball_seams:
            seam.set_transform(Affine2D().translate(tip_x, tip_y) + scene.transData)
        pivot.center = (position, 0)
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
            for ax, values, minimum_span in zip(
                    axes, [plotted, state[:, 0], result["force"]], [10.0, 0.1, 2.0]):
                ax.lines[0].set_data(time, values)
                ax.relim()
                ax.autoscale_view(scalex=False)
                # Avoid microscopic axis ranges when a live trace starts at zero.
                low, high = ax.get_ylim()
                if high - low < minimum_span:
                    middle = (low + high) / 2
                    ax.set_ylim(middle - minimum_span / 2, middle + minimum_span / 2, auto=True)
                ax.set_xlim(time[0], max(time[0] + 10, time[-1]))
            if requested_line is not None:
                requested_line.set_data(time, result["requested_force"])
                axes[-1].relim()
                axes[-1].autoscale_view(scalex=False)
                low, high = axes[-1].get_ylim()
                if high - low < 2:
                    middle = (low + high) / 2
                    axes[-1].set_ylim(middle - 1, middle + 1, auto=True)
            # Ease toward the cart rather than jumping when it crosses a boundary.
            # Simulation time makes smoothing independent of rendering frame rate.
            left, right = scene.get_xlim()
            elapsed = max(0.0, float(time[index]) - last_camera_time)
            follow_fraction = -np.expm1(-elapsed / 0.25)
            camera_center = (left + right) / 2
            camera_center += follow_fraction * (position - camera_center)
            view_half_width = (right - left) / 2
            scene.set_xlim(camera_center - view_half_width, camera_center + view_half_width)
            last_camera_time = float(time[index])
        left, right = scene.get_xlim()
        scale_bar.set_xdata([0.04, 0.04 + 0.1 / (right - left)])
        for cursor in cursors:
            cursor.set_xdata([time[index], time[index]])

        # Matplotlib's callback type expects the objects updated in this frame.
        return [cart, rod, weight, *ball_seams, pivot, detail_rod, scale_bar, scale_label, label, *cursors]

    draw(0)
    return fig, draw


def save_summary(result, config, path):
    """Save a static final scene and complete time-history plots."""
    fig, draw = create_figure(result, config)
    draw(len(result["time"]) - 1)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def show_animation(result, config):
    """Run live physics so a draggable obstacle can push the tennis ball."""
    fig = plt.figure(figsize=(13, 8), constrained_layout=True)
    # Retain widgets, callbacks, and the physics generator until the window closes.
    active = {"animation": None, "buttons": [], "drag_offset": None}

    def start_mode(mode):
        previous = active["animation"]
        if previous is not None and previous.event_source is not None:
            previous.pause()

        box = NudgeBox(config)
        steps = simulation_steps(config, mode, contact=box, live=True)
        first_sample = next(steps)
        # Keep at most 30 simulated seconds of history to bound UI memory use.
        history = deque([first_sample], maxlen=max(2, int(30 / config["simulation"]["control_interval"])))
        recording = samples_to_result(history, mode)
        recording["live"] = True
        _, draw = create_figure(recording, config, fig=fig, show_controls=True)
        scene = fig.axes[0]
        patch = Rectangle(box.lower_left,
                          box.width, box.height, facecolor="mediumpurple", edgecolor="indigo",
                          zorder=5)
        scene.add_patch(patch)
        active.update(box=box, patch=patch, scene=scene, drag_offset=None,
                      history=history, steps=steps, target_time=0.0)
        status = scene.text(0.03, 0.02, "Drag the purple square into the tennis ball; move away to release.",
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
            box.reset(cart_position=float(history[-1]["state"][0]))
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
