# What this simulation uses

Current setup as of October 16, 2026. Change values in
`config/parameters.yaml`; this document is a snapshot of those settings.

| Part | Current specification | Role |
|---|---|---|
| Carbon-fiber tube | 1 m long, 34 g; assumed 6 mm outside diameter | Rigid stick that pivots on the cart |
| Tennis ball | 56 g, 67 mm diameter | Weight at the top of the tube |
| Moving cart | 190 g; assumed drawing size 100 × 45 mm | Moves horizontally to catch and balance the stick |
| Motor | [Pololu #4698](https://www.pololu.com/product/4698), 24 V, 6.25:1 gearbox | Drives the cart through a belt and pulley |
| Pulley and belt | Assumed 20-tooth, 2 mm-pitch pulley; 6.366 mm pitch radius | Converts shaft rotation into cart movement; no belt slip modeled |
| Motor driver | Assumed 24 V supply with a 0.75 A current limit | Controls motor voltage and limits current; no specific driver selected |
| Purple square | 100 × 100 mm; contact force capped at 3 N | Draggable test obstacle for nudging the ball; no simulated mass |

## Motor and control

The motor's published reference values are **1600 RPM and 0.1 A unloaded**,
with extrapolated stall values of **3 A and 0.3432 N m**. These describe the
motor reference, not continuous operating limits. The simulation uses the
lower **0.75 A** driver limit and caps the controller's force request at **13 N**.

`src/motor.py` converts requested force into voltage and calculates actual force
using motor speed, current, back EMF, and estimated gearbox losses. With the
chosen pulley, 1600 RPM corresponds to about **1.07 m/s** unloaded cart speed;
loaded performance depends on the force required.

An energy-based controller swings the stick up from below, then **LQR** keeps
it upright. Commands update at **100 Hz**. Balance mode starts at a **2° tilt**;
swing-up starts hanging straight down.

## Sensors

No physical sensor hardware has been selected. `src/sensors.py` simulates
readings of cart position/velocity and stick angle/angular velocity. Current
noise standard deviations are **20 mm, 0.1 m/s, 0.02 rad (about 1.15°), and
0.1 rad/s**, respectively. Seed `7` makes the noise sequence repeatable.

## Assumptions to remember

- The tube is a uniform rigid rod; the ball is a point mass at a modeled
  pivot-to-center distance of 1 m. Flexing, ball deformation, and attachment
  mass are not included.
- The 6 mm tube diameter and cart dimensions are drawing assumptions. Tube
  inner diameter, wall thickness, supplier, and mounting hardware are unspecified.
- The code uses 190 g as the **whole translating cart mass**, although its YAML
  comment calls it the motor mass. A real build needs the actual moving mass;
  a motor fixed to the rail would not be part of that mass.
- Cart and pivot friction are currently zero; motor losses remain enabled.
  The rail has no travel limits, and motor heating, backlash, and rotating
  inertia are not modeled.
- The square is a test tool. Its size and the ball's size determine contact;
  rod/cart drawing dimensions do not automatically change mass or dynamics.
