# First DC motor model: Pololu #4698

Read `src/balancing_stick/motor.py` first. `DCMotor` connects the LQR force request
to motor voltage, then calculates the actual force delivered to the cart.

## Reference data and choices

Manufacturer reference: [Pololu #4698](https://www.pololu.com/product/4698), with
[37D performance curves](https://www.pololu.com/file/0J1736/pololu-37d-metal-gearmotors-rev-1-2.pdf).

| Reference | Value |
|---|---|
| Nominal voltage | 24 V |
| Exact gearbox ratio | 6.25:1 |
| Unloaded output speed | 1600 RPM |
| Unloaded current | 0.1 A |
| Extrapolated stall current | 3 A |
| Extrapolated output stall torque | 3.5 kgf cm = 0.34323275 N m |

Only these reference values come from the manufacturer. We choose a 20-tooth,
2 mm-pitch pulley (pitch radius about 6.366 mm), a 24 V supply, and an ideal
four-quadrant current-limiting driver. The default 0.75 A limit follows the
manufacturer's general recommendation of 25% or less of stall current. This is
not a detailed thermal rating or a reproduction of a particular motor driver.

Resistance, back-EMF constant, viscous loss, and effective gearbox efficiency are
estimated to fit the reference endpoints. Matching those endpoints is a check
on the equations, not independent validation against measured hardware.

## Equations, one step at a time

Let G be gearbox ratio, r be pulley pitch radius, v be cart speed, and omega be
motor rotor speed. The belt is rigid and does not slip:

```
output angular speed = v / r
rotor angular speed = G * v / r
output RPM = output angular speed * 60 / (2*pi)
```

We neglect inductance, so current reaches its electrical steady value instantly:

```
R = nominal voltage / stall current
Ke = (nominal voltage - R * no-load current) / no-load rotor speed
Kt = Ke                          # SI motor constants: N m/A and V s/rad
B = Kt * no-load current / no-load rotor speed

back EMF = Ke * rotor angular speed
current = (terminal voltage - back EMF) / R
net rotor torque = Kt * current - B * rotor angular speed
```

The ideal driver caps current at the configured limit by adjusting terminal
voltage. It is assumed to support braking/regeneration. An explicit error stops
the simulation if overspeed would require voltage outside the supply rails to
maintain that limit; real driver protection beyond this envelope is not modeled.

Estimated gearbox efficiency is fitted from the stall point:

```
eta = output stall torque / (G * Kt * stall current)
output torque = eta * G * net rotor torque     # Driving
output torque = G * net rotor torque / eta     # Back-driven / braking
cart force = output torque / r
```

Using the reciprocal efficiency while braking prevents the gearbox from creating
energy. This is still a coarse constant-efficiency approximation of real losses.

## How it fits into the simulation

1. LQR requests a cart force, still capped by `controller.max_force`.
2. `voltage_for_force()` converts that request into voltage using measured speed.
3. Voltage is held during each control interval.
4. `evaluate()` recalculates current and actual force at every physics solver step.
5. Speed, current, voltage, output torque, and actual/requested force are recorded.

The conversion compensates for estimated back EMF and drag; it does not implement
a measured-current feedback loop. Limits and changing speed mean requested and
actual force can differ. Current limiting is an ideal driver abstraction.

Fall mode opens the motor circuit: zero current, no electrical braking, but some
mechanical drag. This differs from commanding zero terminal voltage, which
would electrically brake the motor. The gearbox remains mechanically attached.

Set `motor.enabled: false` to compare with the original ideal-force model.
The graph compares requested and actual force, and the scene displays output
shaft RPM, current, and terminal voltage. CSV files contain motor telemetry.

## First experiments

In near-upright balance mode, a 5 kg cart at 12 degrees falls with the default 0.75 A motor limit.
A 5 kg cart at 2 degrees and a 1 kg cart at 8 degrees recover with the current
other settings. Swing-up mode uses a separate controller and downward start;
see the README for that experiment.

Change one setting at a time, then restart the Python program. Smaller pulleys
give more force for a given output torque, but less cart speed for a given RPM.
Increasing supply voltage is not the same as increasing current capacity.

No rotor/gearbox inertia, inductance, temperature, backlash, belt elasticity,
driver switching, or encoder quantization is modeled yet. RPM is kinematically
linked to cart speed, not a separate state. Sensor readings retain the previous
ideal/noisy-state model; the encoder specification is not simulated here.
