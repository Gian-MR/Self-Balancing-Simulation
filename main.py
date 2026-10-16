"""Start here: load settings, run physics, save results, and show animation."""

import argparse
from pathlib import Path

import numpy as np
import yaml

from src.simulation import run_simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["swing-up", "balance", "fall"], default="swing-up")
    parser.add_argument(
        "--no-gui", action="store_true", help="Save the plots without opening an animation window"
    )
    arguments = parser.parse_args()

    # Resolve paths relative to this file so running from VS Code also works.
    project_directory = Path(__file__).resolve().parent
    with (project_directory / "config/parameters.yaml").open() as config_file:
        config = yaml.safe_load(config_file)
    try:
        simulation_result = run_simulation(config, arguments.mode)
    except (TypeError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Simulation error: {error}\n")

    results_directory = project_directory / "results"
    results_directory.mkdir(exist_ok=True)
    csv_path = results_directory / f"{arguments.mode}.csv"
    png_path = results_directory / f"{arguments.mode}.png"
    # CSV angles use radians, as the physics does. The plots use degrees.
    csv_columns = [
        simulation_result["time"],
        simulation_result["state"],
        simulation_result["force"],
        simulation_result["requested_force"],
    ]
    csv_header = (
        "time_s,position_m,velocity_m_s,angle_rad,angular_velocity_rad_s,force_N,requested_force_N"
    )
    if simulation_result["motor"] is not None:
        for motor_quantity, motor_unit in [
            ("command_voltage", "V"),
            ("terminal_voltage", "V"),
            ("current", "A"),
            ("output_torque", "Nm"),
            ("output_rpm", "rpm"),
            ("rotor_rpm", "rpm"),
        ]:
            csv_columns.append(simulation_result["motor"][motor_quantity])
            csv_header += f",{motor_quantity}_{motor_unit}"
    csv_columns.append((simulation_result["phase"] == "balance").astype(int))
    csv_header += ",balance_controller_active"
    np.savetxt(
        csv_path, np.column_stack(csv_columns), delimiter=",", comments="", header=csv_header
    )

    # Select a non-window backend before importing visualization in no-GUI mode.
    if arguments.no_gui:
        import matplotlib

        matplotlib.use("Agg")
    from src.visualization import save_summary, show_animation

    save_summary(simulation_result, config, png_path)
    print(simulation_result["reason"])
    print(f"Final time: {simulation_result['time'][-1]:.2f} s")
    stick_angle_error = (simulation_result["state"][-1, 2] + np.pi) % (2 * np.pi) - np.pi
    print(f"Final angle: {np.rad2deg(stick_angle_error):.3f} degrees from upright")
    print(f"Saved {csv_path}\nSaved {png_path}")

    if not arguments.no_gui:
        show_animation(simulation_result, config)


if __name__ == "__main__":
    main()
