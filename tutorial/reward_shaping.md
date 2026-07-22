# Reward Shaping for Advanced Go2 Skills

Dense, carefully gated reward terms are what make the biped and backflip tasks learnable. Both skills have hard local optima (crouch-walking, tip-toe standing, ground tumbling, skipping the flip) that a naive reward would happily farm. This document walks through the reward design that closed those loopholes.

Source implementations:

- Biped: `biped_rob6323_go2_env.py` / `biped_rob6323_go2_env_cfg.py`
- Backflip: `backflip_rob6323_go2_env.py` / `backflip_rob6323_go2_env_cfg.py`

---

## Shared design principles

1. **Ungated shaping first, gated skill later.** Early terms create a gradient toward the hard transition (rear-up / launch). Skill terms only pay once that transition has started, so the policy cannot cash in on a wrong posture.
2. **Gates multiply rewards, they do not replace them.** Soft clamps in `[0, 1]` keep gradients continuous near the threshold instead of creating cliff edges.
3. **Monotonic / bounded progress rewards.** Terms that integrate rates can be farmed by oscillation. Paying only for *new* progress (or using a linear ramp from a known start) bounds the episodic return.
4. **Regularization is always on.** Torque, action rate/jerk, joint velocity, and soft joint limits discourage thrashing without deciding *what* skill to learn.

---

## Part 1: Bipedal (Hind-Leg) Locomotion

### Task goal

The Go2 starts in its normal quadruped pose, rears up onto its hind legs, then stands and walks while tracking `(vx, yaw-rate)` commands. Front legs stay actuated but should tuck against the body and stay off the ground once upright.

### Key signals

| Signal | Definition | Role |
|---|---|---|
| `up_proj` | `-projected_gravity_b[:, 0]`, clamped ≥ 0 | 0 lying flat, 1 fully nose-up (body +x = world up) |
| `gate` | soft ramp of `up_proj` from 0.80 → 0.95 | Opens locomotion / tuck / contact terms only when nearly upright |
| `height_gate` | soft ramp of base height from 0.35 → 0.45 m | Blocks crouch-walking from collecting tracking reward |
| `track_gate` | `gate * height_gate` | Combined uprightness × height gate for velocity / gait terms |
| Heading velocities | planar forward/lateral from `(body_x − body_z)` | Valid across the whole pitch range (body-x alone degenerates at 90°) |

### Ungated shaping (drives the rear-up)

These terms are active from the first step so the policy has a continuous gradient out of quadruped stance.

#### `upright` (+2.0)

```text
rew = 0.5 * (1 + up_proj)
```

Smoothly increases as the trunk pitches toward vertical. At quadruped rest (`up_proj ≈ 0`) the raw value is 0.5; fully upright it is 1.0. This is the primary exploration signal for the rear-up.

#### `height_ramp` (+3.0)

```text
height_pitch_gate = clamp((up_proj − 0.5) / 0.4, 0, 1)
rew = height_pitch_gate * clamp((h − 0.30) / (0.52 − 0.30), 0, 1)
```

A **linear** ramp from the quadruped start height (0.30 m) to the biped target (0.52 m). An exponential kernel centered at 0.52 m would be nearly flat at the start height and give no gradient. The soft pitch gate (`up_proj` 0.5 → 0.9) kills the **tip-toe quadruped** exploit: standing tall on all four legs raised base height without pitching, and previously farmed this term with zero balance risk.

#### `roll_penalty` (−2.0)

```text
pen = projected_gravity_b[:, 1]²
```

Keeps the rear-up in the sagittal plane. Sideways lean during the transition is the fastest way to fall.

#### `joint_limit_penalty` (−10.0)

Sum of soft-limit overshoot across all joints. Hind hips/thighs run near their limits when upright; this term keeps the policy from hanging on the hard stops.

#### `hind_calf_contact_penalty` (−0.2)

Ungated count of hind-calf contacts above 5 N. Sitting back on the haunches is a tempting intermediate posture during rear-up; making it never free pushes the policy toward a true biped stance instead of a seated crouch.

### Gated skill terms (pay only once upright)

#### `height_fine` (+1.0)

```text
rew = gate * exp(−(h − 0.52)² / 0.02)
```

Once the trunk is upright, this exponential kernel pins the base near the biped target. The wider kernel (`0.02`) reaches further down into a pitched crouch so extending the legs remains attractive.

#### `track_lin_vel` (+1.5) and `track_yaw_rate` (+0.75)

```text
rew_lin = track_gate * exp(−((vx_cmd − v_fwd)² + v_lat²) / 0.25)
rew_yaw = track_gate * exp(−(yaw_cmd − ω_z_w)² / 0.25)
```

Standard exponential velocity tracking, but **doubly gated**. Without `height_gate`, training converged to a crouch-walk local optimum: vertical trunk (`up_proj ≈ 0.85`), base stuck at ~0.30 m, tracking saturated, zero falls — and no incentive to stand up. Tracking must only pay when the robot is both upright *and* tall. Total tracking payoff is intentionally smaller than standing payoff so dropping back down is never profitable.

Yaw is tracked in the **world** frame (`root_ang_vel_w[:, 2]`): body-frame yaw becomes roll once the robot is pitched 90°.

#### `front_tuck_penalty` (−0.25)

Squared error of the six front-leg joints from a tucked pose `(hip=0, thigh=1.3, calf=−2.4)`. Gated so the front legs can still push off during rear-up.

#### `front_contact_penalty` (−0.5)

Count of front feet in contact (> 5 N), gated. Once upright, resting on the front feet defeats the biped objective; during rear-up those contacts remain legal.

#### `tracking_contacts_shaped_force` (+1.0)

Rewards hind-foot contact forces that match a 2-foot gait clock (RL/RR, 180° phase offset, von-Mises smoothing). Also multiplied by `track_gate`, so gait only pays in the standing regime. Without this, the policy tends toward a shuffle or hop rather than an alternating biped walk.

#### `undesired_contact_penalty` (−1.0)

Gated count of contacts on hind thighs, hind calves, and the base. Brief base taps during overshoot are recoverable (no hard termination), but sustained thigh/calf contact while upright is sitting — and must cost reward. Gating is intentional: those contacts are legal during the rear-up transition.

#### `vertical_velocity_penalty` (−0.5)

```text
pen = gate * clamp(|v_z| − 0.3, min=0)²
```

Damps bobbing once upright. Excess vertical motion wastes energy and destabilizes the biped balance.

### Regularization (always on)

| Term | Scale | What it penalizes |
|---|---|---|
| `torque_magnitude_penalty` | −1e-4 | Mean squared PD torque / τ_lim² |
| `action_rate_penalty` | −2e-3 | ‖a_t − a_{t−1}‖² (first difference) |
| `action_jerk_penalty` | −1e-3 | ‖a_t − 2a_{t−1} + a_{t−2}‖² (second difference) |
| `dof_vel_penalty` | −1e-4 | Σ q̇² |

### How the terms compose

```text
Episode timeline (successful policy)
─────────────────────────────────────
t≈0     quadruped rest
        upright ≈ 1.0 raw, height_ramp ≈ 0, all gated terms = 0

t early rear-up: pitch ↑, height ↑ under pitch gate
        upright + height_ramp dominate; front contacts allowed

t mid   upright gate opens (up_proj ≳ 0.8), height extends past 0.35 m
        height_fine, tuck, undesired-contact turn on

t late  track_gate opens → velocity + gait rewards take over
        standing payoff stays larger than tracking so falling is never worth it
```

Local optima this design closed:

1. **Crouch-walk** → tracking also height-gated.
2. **Tip-toe quadruped** → height ramp pitch-gated.
3. **Sitting on haunches** → ungated hind-calf contact penalty + gated undesired contact + height_fine at 0.52 m.

---

## Part 2: Backflip with Recovery

### Task goal

The Go2 starts in quadruped stance, performs **one** full backward rotation while airborne, lands, and recovers to a quiet quadruped stand until episode timeout (5 s). Skipping the flip, tumbling on the ground, or rearing statically must never pay.

### Timed phase schedule

Phase is a function of episode time. The policy observes normalized time and rotation progress so it can plan within the schedule.

| Phase | Window | What pays |
|---|---|---|
| P0 prepare | `[0, 0.5)` s | Standing terms, ungated |
| P1 flip | `[0.5, 2.0)` s | Rotation / height / aerial / completion |
| P2 recover + stand | `[2.0, 5.0)` s | Standing × flip-completion gate; completion still active |

Regularization and undesired-contact penalties are always on.

### Key signals

| Signal | Definition | Role |
|---|---|---|
| `cum_pitch` | ∫ `root_ang_vel_b[:, 1] dt` | Integrated body pitch; full backflip ≈ −2π |
| `flip_progress` | `clamp(−cum_pitch / 2π, 0, 1)` | Fraction of a full backward turn |
| `aerial_pitch` | backward pitch accumulated only while fully airborne | Distinguishes ballistic flips from ground tumbles |
| `fully_airborne` | max contact force over all bodies < 5 N | Gate for jump height and aerial rotation credit |
| `flip_done` | `aerial_gate × soft(flip_progress)` | Opens P2 standing rewards only after a real flip |
| `stand_gate` | `in_prepare + in_recovery * flip_done` | P0 free; P2 locked until flip completes |

### Flip-window terms

#### `flip_rotation` (+20.0)

```text
progress_gain = (flip_progress − max_flip_progress).clamp(min=0)
rew = (in_flip + in_recovery) * progress_gain
```

**Monotonic progress reward.** Pays only for *new* net backward rotation (tracked by a running max). Total raw credit per episode is bounded by 1.0, so the scaled return is bounded by 20. An earlier airborne-gated *rate* reward was farmed by rocking in place with zero net rotation; the running-max formulation kills that exploit.

#### `flip_height` (+3.0)

```text
rew = in_flip * fully_airborne * clamp(h − 0.30, 0, 0.5)
```

Rewards ballistic altitude during the flip window only. Without the airborne gate, statically rearing up on the hind legs (base > 0.35 m) farmed this term with no rotation. The floor is standing height (0.30 m) so any real hop pays from the first centimeter.

#### `aerial_rotation` (+5.0)

```text
aerial_gain = Δ aerial_pitch  (clamped total ≤ 2π)
rew = (in_flip + in_recovery) * aerial_gain
```

Per-radian credit for backward pitch performed while fully airborne. This is the continuous bridge from "hop" to "aerial flip." Relying only on the compound `flip_done` gate left too large a gap: early training stalled at stand-and-lean because nothing paid until nearly a full turn was already complete.

#### `flip_completion` (+1.0)

```text
aerial_gate = clamp(aerial_pitch / π, 0, 1)
flip_done   = aerial_gate * clamp((flip_progress − 0.7) / 0.25, 0, 1)
rew = (in_flip + in_recovery) * flip_done
```

Dense "you finished it" signal once ≥ 70% of a turn is done *and* at least half a turn (π rad) was ballistic. Ground-tumbling over the legs can accumulate `cum_pitch ≈ −2π` at zero altitude; requiring aerial rotation kills that optimum. `flip_done` also unlocks P2 standing rewards.

### Standing terms (P0 ungated; P2 gated by flip completion)

```text
stand_gate = in_prepare + in_recovery * flip_done
```

#### `stand_upright` (+1.0)

```text
rew = stand_gate * clamp(−projected_gravity_b[:, 2], 0, 1)
```

Rewards belly-down orientation (quadruped upright). In P0 this encourages a clean crouch/launch setup; in P2 it only pays after a completed flip, so standing still never substitutes for flipping.

#### `stand_height` (+1.0)

```text
rew = stand_gate * exp(−(h − 0.30)² / 0.01)
```

Pins the base near the default quadruped stance height after recovery.

#### `stand_pose` (+0.5)

```text
rew = stand_gate * exp(−Σ(q − q_default)² / 1.0)
```

Pulls joints back to the default standing pose after the violent tuck/extension of the flip.

#### `stand_feet_contact` (+0.5)

Mean fraction of the four feet in contact (> 5 N). Encourages a plant on all fours rather than a crouch on the calves or a one-sided land.

#### `stand_still` (−0.1)

```text
pen = stand_gate * (‖v_lin_b‖² + 0.2 ‖ω_b‖²)
```

Penalizes residual motion once standing is gated on. Keeps the recovery phase from bouncing forever.

### Always-on contact and regularization

#### `undesired_contact_penalty` (−1.0)

Count of thigh / calf / hip ground contacts (> 5 N). Kneeling or sitting can fake both "standing" (base upright, feet unloaded) and "airborne feet." This term is always on so those postures are never free — including during the flip window.

#### Soft limits and smoothness

| Term | Scale | Note |
|---|---|---|
| `joint_limit_penalty` | −10.0 | Soft-limit overshoot |
| `torque_magnitude_penalty` | −1e-4 | Mean τ² / τ_lim² |
| `action_rate_penalty` | −1e-3 | Weaker than locomotion — the flip is inherently violent |
| `action_jerk_penalty` | −5e-4 | Same rationale |
| `dof_vel_penalty` | −5e-5 | Light joint-velocity damping |

### How the terms compose

```text
Episode timeline (successful policy)
─────────────────────────────────────
P0  [0, 0.5)   stand quietly; prepare for crouch
               stand_* terms pay; flip terms off

P1  [0.5, 2.0) crouch → launch → rotate backward in air
               flip_height pays while airborne
               flip_rotation / aerial_rotation accumulate progress
               flip_done ramps up near a full ballistic turn

P2  [2.0, 5.0) land and hold quadruped stand
               stand_* unlock only if flip_done > 0
               skipping the flip leaves P2 with near-zero standing reward
```

Local optima this design closed:

1. **Rocking / rate farming** → monotonic running-max rotation reward.
2. **Ground tumble** → aerial-pitch requirement inside `flip_done` + `aerial_rotation` shaping.
3. **Static rear-up for height** → `flip_height` gated by fully airborne.
4. **Skip the flip and stand** → P2 standing multiplied by `flip_done`.
5. **Kneel / sit to fake stand** → always-on undesired-contact penalty.

---

## Takeaways

- **Gates encode curriculum.** Ungated terms create the transition; gated terms define the skill. Putting both in one scalar reward works when the gates are soft and the relative scales keep standing more valuable than cheating.
- **Shape what you can measure early.** Linear height ramps and per-radian aerial rotation bridge large gaps that a single completion bonus cannot.
- **Bound episodic credit.** Progress rewards with running maxima (backflip) and pitch-gated linear ramps (biped) prevent infinite farming of dense terms.
- **Penalties close the last loopholes.** Contact on the wrong bodies, crouch tracking, and tip-toe height hacks all required explicit counters discovered during training iterations — not just stronger positive rewards.
