# Physical model

A cart translates horizontally. A uniform rigid stick pivots freely on the cart,
and an additional point mass is fixed at the tip. Motion stays in one vertical
plane. The motor applies a horizontal force to the cart.

State order throughout the code:

| Symbol | Meaning | Unit |
|---|---|---|
| x | Cart position; positive right | m |
| v | Cart velocity | m/s |
| theta | Stick angle from upright; positive right | rad |
| omega | Stick angular velocity | rad/s |

For near-upright balance mode, the desired state is [0, 0, 0, 0]. The initial cart
is stationary and centered, with the configured stick tilt. Degrees in
configuration are converted to radians. Swing-up mode instead starts exactly
downward (180 degrees), pumps energy into the stick, and switches to LQR near
upright. The cart target for capture is its position at the handoff.

Let M be cart mass, m be stick mass, p be top mass, L be stick length, g be
gravity, b be cart friction, c be pivot friction, and F be motor force. Define:

```
T = M + m + p              total moving mass
h = m L / 2 + p L          first moment of mass about pivot
J = m L² / 3 + p L²        rotational inertia about pivot

T x_ddot + h cos(theta) theta_ddot = F - b v + h sin(theta) omega²
h cos(theta) x_ddot + J theta_ddot = h g sin(theta) - c omega
```

These coupled equations follow from the kinetic and potential energies:

```
kinetic = T v² / 2 + h cos(theta) v omega + J omega² / 2
potential = h g cos(theta)
```

Upright has maximum potential energy, so a small uncontrolled tilt grows.
`physics.py` solves the two equations for the accelerations using a 2×2 matrix.
The state derivative is [v, x_ddot, omega, theta_ddot].

SciPy integrates the nonlinear equations between 100 Hz control updates. With
the DC motor enabled, commanded voltage is held within each interval and actual
force changes with speed. With the ideal actuator, force is held constant.
The LQR controller uses a linear approximation at upright, but the simulated
physics remains nonlinear. See [the motor model](motor.md) for the actuator.

Sensors initially report the exact four states. Optional Gaussian measurement
noise is independent in each channel, with a fixed seed for repeatability.
Velocity channels assume readings/estimates already exist; no estimator is
implemented. Noise changes the controller input, never the true physical state.

Simplifications: point top mass, rigid rod, viscous friction, no delays, no ground
contact, and no finite rail. The DC motor uses a steady-current electrical model
with voltage/current limits; its inductance and rotating inertia are omitted.
Balance/fall modes end at the configured fall angle. Swing-up allows full
rotation and runs for its separately configured duration; the rail is assumed
elevated with clearance for the stick below it.
