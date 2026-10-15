# Self-Balancing Simulation

A beginner-friendly Python simulation of a cart balancing a stick with a weight
fixed to its tip. Start with `main.py`, then follow the imports into `src/`.

The actuator now models a Pololu #4698 DC gearmotor with voltage, current, back
EMF, gearbox losses, and a pulley. See [the motor walkthrough](docs/motor.md).
It is a first approximation fitted to published reference data, not a validated
hardware replica. In near-upright balance mode, a 5 kg cart starting at 12 degrees
exceeds the default motor capability and falls, while a 2 degree starting tilt
recovers. Set `motor.enabled: false` for ideal force.

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
python main.py                  # Start hanging below, swing up, then balance.
python main.py --mode balance   # Balance from a small initial tilt.
python main.py --mode fall      # Turn off the motor and watch it fall.
python main.py --no-gui         # Save results without an animation window.
```

Each run saves a CSV and a PNG under `results/`, overwriting the previous files
for that mode. Close the animation window to finish the interactive run. Physics
is computed first; the animation then plays those recorded results. Playback
speed may vary with your computer.

The animation window has **Restart: Swing up**, **Restart: Balance**, and
**Restart: Fall (motor off)** buttons. Swing-up starts exactly downward; balance
and fall use the configured initial tilt. You can switch while playback is
running or after it finishes.
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
default in balance/fall modes. Swing-up allows full rotation and adds an energy
controller that switches to LQR near upright. The rail is elevated with clearance
for the rod below it; ground/rail collisions are not modeled.
The orange circle is a visual marker for a point mass, not a modeled solid ball.

## How swing-up works

The stick starts at 180 degrees (straight down) with zero speed. A brief cart
nudge begins the swing. An energy-based controller then moves the cart to build
the pendulum's energy. When the stick is within 6 degrees of upright and rotating
slower than 1 rad/s, LQR takes over. If it leaves 20 degrees, swing-up resumes.
Angles wrap around so a complete rotation is handled correctly.

With the current 5 kg cart and motor limits, capture occurs at about 11.7 seconds
in the 30-second swing-up run. Settings are under `swing_up` in YAML. Capture holds
the cart near its position at handoff; it does not demand a return to the original
center. The model still has no finite track, so this is not a guarantee that it
would fit a real rail. Changing the plant or motor may require new tuning.

The controller is in `controller.py` and its phase is displayed in the scene.
CSV `balance_controller_active` is 1 when LQR is active, otherwise 0. The CSV angle
remains unwrapped radians; displayed angles use the nearest upright position.
The approach follows [MIT's energy-shaping and LQR swing-up discussion](https://underactuated.csail.mit.edu/acrobot.html).

## Check the model

```sh
python -m unittest discover -s tests -v
```

References: [SciPy integration](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html),
[python-control LQR](https://python-control.readthedocs.io/en/latest/generated/control.lqr.html),
and [Matplotlib animation](https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html).
