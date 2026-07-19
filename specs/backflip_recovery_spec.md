# Go2 Backflip with Recovery — Task Spec

Task ID: `Template-Rob6323-Go2-Backflip-Direct-v0`

Goal: the Go2 starts in its default quadruped stance, performs **one backflip**
(full backward rotation about the body pitch axis, airborne), lands, and recovers
to quadruped standing without falling, holding the stand until episode timeout.

Design decisions (agreed 2026-07-19):
- Stock actuator model: DC motor, 23.5 N·m effort/saturation limit, 30 rad/s
  velocity limit. No boosting.
- Timed phase schedule observed by the policy (reference-free reward shaping).
- Single flip per episode, then stand.
- Replaces the previous (unregistered) backflip implementation entirely.

## Episode structure

Control at 50 Hz (sim dt 1/200 s, decimation 4). Episode length 5.0 s.
Phase is a function of episode time `t`:

| Phase | Window | Active rewards |
|---|---|---|
| P0 prepare | [0, 0.5) s | standing terms, ungated |
| P1 flip | [0.5, 2.0) s | rotation (airborne-gated), jump height, completion |
| P2 recover + stand | [2.0, 5.0) s | standing terms × flip-completion gate, completion |

Regularization terms (joint limits, torque, action rate/jerk, joint velocity)
are always on.

## Rotation bookkeeping

- `cum_pitch = Σ root_ang_vel_b[:, 1] · step_dt`, reset to 0 each episode.
  In the FLU body frame a nose-up/backward rotation is negative about +y, so a
  full backflip reaches `cum_pitch ≈ −2π`.
- `flip_progress = clamp(−cum_pitch / 2π, 0, 1)`
- `aerial_pitch`: backward pitch accumulated only while NO body is in contact
  (design iter3: without it, training converged to a full-rotation ground tumble
  over the legs at zero altitude).
- `flip_done = clamp(aerial_pitch / π, 0, 1) × clamp((flip_progress − 0.7) / 0.25, 0, 1)`
  — rotation must be complete AND at least half of it ballistic.

## Actions / control

Policy outputs 12 joint-position offsets: `q_des = 0.5 · a + q_default`.
PD torque `τ = 30.0 (q_des − q) − 0.7 q̇`, clipped to ±23.5 N·m (the DC-motor
model clips again, velocity-dependent). Implicit actuator stiffness/damping are
zeroed as in the other tasks.

## Observations (47)

`root_lin_vel_b(3) · root_ang_vel_b(3) · projected_gravity_b(3) ·
(joint_pos − default)(12) · joint_vel(12) · prev actions(12) ·
[t / 5.0, cum_pitch / 2π](2)`

## Rewards

Flip window:
- `flip_rotation` (+20.0, P1+P2): monotonic progress reward
  `(flip_progress − running_max_progress).clamp(min=0)` — pays only for new net
  backward rotation, bounded at 1.0×scale per episode. (Design iter2: the original
  airborne-gated pitch-*rate* reward was farmed by rocking in place with zero net
  rotation.)
- `flip_height` (+3.0, P1 only): `clamp(base_height − 0.30, 0, 0.5) ×` fully-airborne
  gate (design iter4: statically rearing up on the hind legs farmed the ungated
  version — only ballistic altitude pays; iter5: threshold lowered from 0.35 to
  standing height so any real hop pays from the first centimeter)
- `aerial_rotation` (+5.0/rad, P1+P2, iter5): increments of backward pitch performed
  fully airborne, credit capped at 2π — continuous bridge from "hop" to "aerial flip"
  (iter4 stalled at "stand + lean": the compound completion gate was too far from
  any paying behavior)

From P1 onward:
- `flip_completion` (+1.0/step): `flip_done`

Standing (P0 ungated; P2 × `flip_done`):
- `stand_upright` (+1.0): `clamp(−projected_gravity_b[:,2], 0, 1)`
- `stand_height` (+1.0): `exp(−(base_height − 0.30)² / 0.01)`
- `stand_pose` (+0.5): `exp(−Σ(q − q_default)² / 1.0)`
- `stand_feet_contact` (+0.5): fraction of the 4 feet in contact (> 5 N)
- `stand_still` (−0.1): `‖v_lin_b‖² + 0.2 ‖ω_b‖²`

Always on:
- `undesired_contact_penalty` (−1.0): thigh/calf/hip ground contact (> 5 N) —
  kneeling/sitting must never fake "standing" or unload the feet for free
  (design iter2: observed exploit)
- `joint_limit_penalty` (−10.0): overshoot beyond soft joint limits
- `torque_magnitude_penalty` (−1e-4): `mean(τ²) / τ_lim²`
- `action_rate_penalty` (−1e-3), `action_jerk_penalty` (−5e-4): weaker than the
  locomotion tasks — the flip is inherently violent
- `dof_vel_penalty` (−5e-5)

## Termination

- Timeout at 5 s.
- Base contact force > 1 N at any time (a failed flip lands on the back).
- No upside-down and no min-height termination: inverted mid-flip and a deep
  crouch are both nominal states.

## Reset

Default quadruped pose at the terrain origin, zero velocities, zero `cum_pitch`
and action history. `episode_length_buf` is **not** randomized at startup (phase
is tied to episode time), so resets stay synchronized.

## Training

`BackflipPPORunnerCfg`: same PPO/network settings as the biped task,
`entropy_coef = 0.01`, `max_iterations = 3000`, experiment
`go2_backflip_direct`, 4096 envs.

## Implementation gotcha (found in design iter2)

The contact sensor MUST be registered with the scene
(`self.scene.sensors["contact_sensor"] = self._contact_sensor` in `_setup_scene`),
otherwise `sensor.update()` is never called and all contact forces stay frozen at
zero. The older quadruped/biped envs in this repo lack this registration, so their
contact-based reward terms and the quadruped base-contact termination are inert.

## Known risks / tuning knobs

- At stock torque, liftoff is the hard part: `flip_height` pays before rotation
  is discovered; raise its scale if the policy never leaves the ground.
- If the policy back-rolls on the ground instead of jumping, the airborne gate
  on `flip_rotation` is doing its job — check `flip_height` progress instead.
- `Kp/Kd` (30/0.7) and `action_scale` (0.5) are cfg fields if the crouch or
  extension look too soft.
