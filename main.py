"""Start here: load settings, run physics, save results, and show animation."""

import argparse
from pathlib import Path

import numpy as np
import yaml
from src.simulation import run_simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["swing-up", "balance", "fall"], default="swing-up")
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
    columns = [result["time"], result["state"], result["force"], result["requested_force"]]
    header = "time_s,position_m,velocity_m_s,angle_rad,angular_velocity_rad_s,force_N,requested_force_N"
    if result["motor"] is not None:
        for key, unit in [("command_voltage", "V"), ("terminal_voltage", "V"),
                          ("current", "A"), ("output_torque", "Nm"),
                          ("output_rpm", "rpm"), ("rotor_rpm", "rpm")]:
            columns.append(result["motor"][key])
            header += f",{key}_{unit}"
    columns.append((result["phase"] == "balance").astype(int))
    header += ",balance_controller_active"
    np.savetxt(csv_path, np.column_stack(columns), delimiter=",", comments="", header=header)

    # Select a non-window backend before importing visualization in no-GUI mode.
    if args.no_gui:
        import matplotlib
        matplotlib.use("Agg")
    from src.visualization import save_summary, show_animation
    save_summary(result, config, png_path)
    print(result["reason"])
    print(f"Final time: {result['time'][-1]:.2f} s")
    angle_error = (result["state"][-1, 2] + np.pi) % (2 * np.pi) - np.pi
    print(f"Final angle: {np.rad2deg(angle_error):.3f} degrees from upright")
    print(f"Saved {csv_path}\nSaved {png_path}")
    
    if not args.no_gui:
        show_animation(result, config)


if __name__ == "__main__":
    main()
