# Self-Balancing Simulation

A beginner-friendly Python simulation of a cart balancing a stick with a weight
fixed to its tip. Start with `main.py`, then follow the imports into `src/`.

The actuator now models a Pololu #4698 DC gearmotor with voltage, current, back
EMF, gearbox losses, and a pulley. See [the hardware overview](docs/hardware.md).
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

Each run first saves an undisturbed reference experiment as CSV and PNG under
`results/`, overwriting the previous files for that mode. The window then starts
a fresh, live simulation that responds to mouse interaction. Close the window to
finish. Slow rendering slows playback rather than increasing the physics steps.
Mouse nudges are not included in the automatically saved reference CSV/PNG.

The animation window has **Restart: Swing up**, **Restart: Balance**, and
**Restart: Fall (motor off)** buttons. Swing-up starts exactly downward; balance
and fall use the configured initial tilt. You can switch while playback is
running or after it finishes. Live swing-up and balance keep running beyond the
configured reference duration so you can keep experimenting. Fall mode still
stops at its fall limit. Restart buttons do not overwrite the saved reference.

## Nudge the weight

The main scene now uses physical dimensions with equal horizontal/vertical
scaling: a 1 m rod and a 67 mm tennis ball. The ball's contact radius matches
its drawing. A magnified tip inset makes it easier to inspect without enlarging
the ball in the main scene. The 100 mm scale bar provides a size reference.
The camera smoothly follows the cart at a fixed scale during live runs.

Edit `geometry` in `config/parameters.yaml` for dimensions. All dimensions
are read only from this YAML section; missing or invalid
dimensions produce a configuration error rather than using fallback sizes.
Rod diameter (6 mm) and cart size (100 × 45 mm) are assumptions until measured; mass alone cannot
determine them. The purple obstacle is a 40 mm square. Rod/cart dimensions are
visual only; ball and obstacle sizes also determine contact. The dynamics still
approximate the ball as a point mass at `stick_length` from the pivot; they do
not include a finite sphere's rotational inertia or rod/cart collisions.

Click and hold the **purple box**, then drag it into the **tennis ball**.
It turns red while pushing. Move it away to release contact, or click **Reset
box**. Dragging only affects the tip weight, not the rod or blue cart. Disable
Matplotlib's pan/zoom toolbar modes before dragging.

The box is a mouse-controlled obstacle with a soft spring/damper contact model,
capped at 3 N. It applies physical forces to both cart motion and stick rotation;
it never directly sets the stick angle or cart position. It stays where you put
it after release, so move it clear if you want the push to end. **Reset box**
brings it back beside the cart's current position, even after the cart has moved
far away. Resetting clears the box's velocity to avoid an artificial kick.
Small nudges may
be corrected; larger ones can knock the stick over. Swing-up tries to recover.

The scene displays contact force as **Box push**. The last 30 simulated seconds
are kept in the live plots. This is a simplified interaction model: the box is
positioned by the mouse and has no simulated mass or recoil. Strong pushes can
exceed the motor model's operating envelope; the window pauses with a message
and the Restart buttons remain available.

## Learn one part at a time

1. Read `docs/model.md` for units, assumptions, and equations.
2. Change `initial_angle_degrees` in `config/parameters.yaml` and run fall mode.
3. Open `physics.py`: it calculates the derivatives, not the animation.
4. Open `simulation.py`: SciPy advances time between controller updates.
5. Open `controller.py`: LQR calculates a corrective motor force.
6. Open `visualization.py`: it draws the states and handles mouse interaction.
7. Try a small sensor noise value in the configuration after ideal balancing works.
8. Open `interaction.py`: it computes the box's force on the tip weight.

Examples: change top mass from 0.2 to 0.3 kg, compare a 4° and 8° starting tilt,
or reduce the force limit. The LQR gains are recalculated when the model changes;
changing mass therefore compares redesigned controllers, not a fixed controller's
robustness to an unknown payload.

The model has no floor collisions or rail ends. It stops at an 80° tilt by
default in balance/fall modes. Swing-up allows full rotation and adds an energy
controller that switches to LQR near upright. The rail is elevated with clearance
for the rod below it; ground/rail collisions are not modeled.
The tennis ball is modeled dynamically as a point mass, without its own spin.

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
