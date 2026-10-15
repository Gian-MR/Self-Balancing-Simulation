# Self-Balancing Simulation

A beginner-friendly Python simulation of a cart balancing a stick with a weight
fixed to its tip. Start with `main.py`, then follow the imports into `src/`.

## Set up in VS Code

Open this folder in VS Code. Install Microsoft's Python extension, then open a
terminal in this folder. A local `.venv` may already exist from project setup.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run **Python: Select Interpreter** in the VS Code command palette and choose
`.venv/bin/python`. Python 3.9 or newer is supported by the source; pip selects
dependency releases compatible with your Python version.

## Run

```sh
python main.py                  # Balance from a small initial tilt.
python main.py --mode fall      # Turn off the motor and watch it fall.
python main.py --no-gui         # Save results without an animation window.
```

Each run saves a CSV and a PNG under `results/`, overwriting the previous files
for that mode. Close the animation window to finish the interactive run. Physics
is computed first; the animation then plays those recorded results. Playback
speed may vary with your computer.

The animation window has **Restart: Balance** and **Restart: Fall (motor off)**
buttons. Click either to replay that mode from the configured starting tilt in
the same window. You can switch while playback is running or after it finishes.
Button playback does not overwrite the CSV/PNG saved at program startup.

## Learn one part at a time

1. Read `docs/model.md` for units, assumptions, and equations.
2. Change `initial_angle_degrees` in `config/parameters.yaml` and run fall mode.
3. Open `physics.py`: it calculates the derivatives, not the animation.
4. Open `simulation.py`: SciPy advances time between controller updates.
5. Open `controller.py`: LQR calculates a corrective motor force.
6. Open `visualization.py`: it draws the recorded states.
7. Try a small sensor noise value in the configuration after ideal balancing works.

Examples: change top mass from 0.2 to 0.3 kg, compare a 4° and 8° starting tilt,
or reduce the force limit. The LQR gains are recalculated when the model changes;
changing mass therefore compares redesigned controllers, not a fixed controller's
robustness to an unknown payload.

The model has no floor collisions or rail ends. It stops at an 80° tilt by
default. LQR balances near upright; it does not swing the stick up from below.
The orange circle is a visual marker for a point mass, not a modeled solid ball.

## Check the model

```sh
python -m unittest discover -s tests -v
```

References: [SciPy integration](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html),
[python-control LQR](https://python-control.readthedocs.io/en/latest/generated/control.lqr.html),
and [Matplotlib animation](https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html).
