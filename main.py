"""Start here: load settings, run physics, save results, and show animation."""

import argparse
from pathlib import Path

import numpy as np
import yaml

from src.balancing_stick.simulation import run_simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["balance", "fall"], default="balance")
    parser.add_argument("--no-gui", action="store_true",
                        help="Save the plots without opening an animation window")
    args = parser.parse_args()

    # Resolve paths relative to this file so running from VS Code also works.
    root = Path(__file__).resolve().parent
    with (root / "config/parameters.yaml").open() as file:
        config = yaml.safe_load(file)
    try:
        result = run_simulation(config, args.mode)
    except (ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Simulation error: {error}\n")

    output = root / "results"
    output.mkdir(exist_ok=True)
    csv_path = output / f"{args.mode}.csv"
    png_path = output / f"{args.mode}.png"
    # CSV angles use radians, as the physics does. The plots use degrees.
    np.savetxt(csv_path, np.column_stack([
        result["time"], result["state"], result["force"],
    ]), delimiter=",", comments="",
        header="time_s,position_m,velocity_m_s,angle_rad,angular_velocity_rad_s,force_N")

    # Select a non-window backend before importing visualization in no-GUI mode.
    if args.no_gui:
        import matplotlib
        matplotlib.use("Agg")
    from src.balancing_stick.visualization import save_summary, show_animation
    save_summary(result, config, png_path)
    print(result["reason"])
    print(f"Final time: {result['time'][-1]:.2f} s")
    print(f"Final angle: {np.rad2deg(result['state'][-1, 2]):.3f} degrees")
    print(f"Saved {csv_path}\nSaved {png_path}")
    
    if not args.no_gui:
        show_animation(result, config)


if __name__ == "__main__":
    main()
