# Bipedal (Hind-Leg) Locomotion Task for Unitree Go2

## Context

The repo currently has one registered Direct RL task (`Template-Rob6323-Go2-Direct-v0`) for quadruped velocity-tracking locomotion, implemented in
[rob6323_go2_env.py](source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/rob6323_go2_env.py) / [rob6323_go2_env_cfg.py](source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/rob6323_go2_env_cfg.py).
Goal: a **new, separately registered task** where the Go2 starts in its normal quadruped pose, rears up onto its hind legs, then stands/walks bipedally tracking **vx + yaw-rate** commands (vy ≡ 0). Front legs stay actuated but are rewarded to hold a tucked pose and penalized for ground contact (no termination on front contact). Training runs **locally** via the Apptainer sandbox (`/home/daniel/Dev/rl_prj/isaac-lab-base.sif` + IsaacLab checkout at `/home/daniel/Dev/rl_prj/IsaacLab`), NOT on the HPC. Deliverable now: implementation + smoke tests only — **no full training run yet**.

User-decided design choices (fixed): new registered task id; quadruped start + learned stand-up; vx+yaw commands; front legs penalized (tuck + contact), not terminated.

## Approach

**Copy-and-modify** (not subclass): ~90% of the reward/termination/gait logic changes, so a new independent env class is cleaner than a fragile subclass. New files sit beside the existing ones.

### Files

1. **NEW** `source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/biped_rob6323_go2_env_cfg.py` — `BipedRob6323Go2EnvCfg(DirectRLEnvCfg)`
2. **NEW** `source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/biped_rob6323_go2_env.py` — `BipedRob6323Go2Env(DirectRLEnv)`
3. **EDIT** `source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/__init__.py` — register `Template-Rob6323-Go2-Biped-Direct-v0` (entry points to the new env/cfg, `rsl_rl_cfg_entry_point` → `BipedPPORunnerCfg`)
4. **EDIT** `source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/agents/rsl_rl_ppo_cfg.py` — add `BipedPPORunnerCfg`: same net `[256,128,64]`/elu/obs-norm, `experiment_name="go2_biped_direct"`, `max_iterations=3000`, `save_interval=100`, `entropy_coef=0.005` (helps rear-up exploration)
5. **NEW** `scripts/train_local.sh` — Apptainer wrapper for local runs (see below)
6. **NEW** `specs/biped_locomotion_spec.md` — save this plan/spec into the repo per user request

### Cfg (`BipedRob6323Go2EnvCfg`) — key values

Keep from quadruped cfg: sim (dt=1/200, decimation=4), plane terrain, `UNITREE_GO2_CFG` with implicit stiffness/damping zeroed, contact sensor, debug-vis markers, Kp=20/Kd=0.5.

Changed/new:
- `observation_space = 50` (see obs table), `action_space = 12`
- `action_scale = 0.5` (0.25 can't reach the biped hind-thigh pose: needs ~±1.5 rad offsets)
- `torque_limits = 23.5` (match DCMotor effort limit; 100 was fiction — penalties/logging should see real torques)
- `episode_length_s = 20.0`, `command_resample_time = 5.0`
- `lin_vel_x_range = (-0.3, 0.5)`, `ang_vel_yaw_range = (-0.5, 0.5)`, vy forced 0 in `_sample_commands`
- Biped stance: `biped_height_target = 0.52`, `quad_start_height = 0.30`, gate thresholds `gate_lower = 0.80`, `gate_upper = 0.95`
- Front tuck targets: `F[LR]_hip = 0.0`, `F[LR]_thigh = 1.3`, `F[LR]_calf = -2.4` (inside soft limits)
- `gait_frequency = 1.5` Hz (2-foot clock)
- scene `num_envs = 2048` default (12 GB RTX 4080; smoke tests override with `--num_envs 32-64`)
- No `base_height_min` field (termination removed)

### Env class — structural deltas from `Rob6323Go2Env`

- `__init__`: keep actions/commands (keep 3-dim command vector, index 1 ≡ 0 — avoids reshuffling obs/debug-vis code), last-action history, applied torques, PD tensors. Find joint/body indices via `find_joints`/`find_bodies` regex, never hardcoded: `_front_joint_ids` (`F[LR]_{hip,thigh,calf}_joint`) + matching tuck-target tensor; sensor ids split into `_hind_feet_ids_sensor` (RL/RR foot) and `_front_feet_ids_sensor` (FL/FR foot); undesired-contact ids = hind thighs/calves + base. Gait buffers shrink to 2 feet (`clock_inputs`, `foot_indices`, `desired_contact_states` all `(N,2)`); drop duty-factor machinery. New `_episode_sums` keys for the new reward set.
- Keep verbatim: `_setup_scene`, `_pre_physics_step`, `_apply_action` (PD + clip), `_penalty_action_rate/_jerk/_torque_magnitude`, `_update_commands`, `_reset_idx` pattern (incl. random `episode_length_buf` spreading), debug-vis.
- `_step_contact_targets`: 2-foot version, 180° offset (RL/RR), same duration-warp + von-Mises smoothing as existing code.

### Core frame math (correctness-critical)

Rear-up = pitch θ about body −y. `projected_gravity_b = (−sinθ, 0, −cosθ)`: quadruped `(0,0,−1)`, fully upright `(−1,0,0)` — world-up is **body +x** when standing.

```python
up_proj = -projected_gravity_b[:, 0]                      # 1.0 when upright
gate = ((up_proj - gate_lower) / (gate_upper - gate_lower)).clamp(0, 1)   # 0 below ~53° pitch, 1 above ~72°
```

Heading frame (degenerate-free for θ ∈ [0°, 135°], i.e. the whole alive set — body x alone degenerates at 90°, body −z at 0°):
```python
fwd_w = quat_apply(root_quat_w, unit_x);  up_w = quat_apply(root_quat_w, unit_z)
heading = (fwd_w - up_w)[:, :2];  heading /= heading.norm(dim=1, keepdim=True).clamp(min=1e-6)
v_fwd = (root_lin_vel_w[:, :2] * heading).sum(1);  v_lat = cross-z component
```
Yaw rate tracked as **world-frame** `root_ang_vel_w[:, 2]` (body-frame yaw becomes roll when pitched 90°).

### Rewards (per-step, same convention as existing env)

Ungated shaping (drives rear-up from frame 1):
- `rew_upright = 0.5 * (1 + up_proj)`, scale **+2.0**
- `rew_height`: **linear ramp** `((h − 0.30)/(0.52 − 0.30)).clamp(0,1)`, scale **+1.5** (exp kernel would be flat at start height — no gradient)
- `pen_roll = projected_gravity_b[:,1]²`, scale **−2.0**
- joint-limit penalty: overshoot beyond `soft_joint_pos_limits`, summed, scale **−10.0**
- keep: action rate **−2e-3**, jerk **−1e-3**, torque magnitude **−1e-4**, dof_vel **−1e-4**

Gated by `gate` (pay only once upright):
- `rew_height_fine = exp(−(h − 0.52)²/0.01)`, scale **+1.0**
- lin tracking `exp(−((vx_cmd − v_fwd)² + v_lat²)/0.25)`, scale **+1.5**; yaw tracking `exp(−(yaw_cmd − ω_z_w)²/0.25)`, scale **+0.75** (total tracking < standing payoff → never profitable to drop back down)
- front tuck: `Σ(q_front − q_tuck)²`, scale **−0.25**; front-foot contact (force > 5 N), scale **−0.5** (gated so front legs can push off during rear-up)
- hind gait contact-tracking (2-foot analog of `_reward_tracking_contacts_shaped_force`), scale **+1.0**
- undesired contact (hind thigh/calf + base, force > 5 N), scale **−1.0**
- vertical-velocity damping `clamp(|v_z_w| − 0.3, min=0)²`, scale **−0.5**

Dropped entirely: Raibert, diagonal-phase, duty-factor, symmetry, pacing, hopping, feet-clearance, orient, lin_vel_z, ang_vel_xy, roll_pitch, base-height-error (quadruped versions).

### Terminations (all three quadruped criteria replaced)

```python
fell_backward = projected_gravity_b[:, 2] > 0.7    # past ~135° — NOT the old `> 0`, which fires at 90.01°!
fell_forward  = projected_gravity_b[:, 0] > 0.7
fell_sideways = projected_gravity_b[:, 1].abs() > 0.8
```
Removed: `base_height_min` (crouch before push-off must be legal), old upside-down check, base-contact-force termination (brief base taps during overshoot are recoverable; handled by gated penalty instead). Update `Episode_Termination/*` log keys.

### Observations — total **50**

| block | dim |
|---|---|
| root_lin_vel_b / root_ang_vel_b / projected_gravity_b | 3+3+3 |
| commands (vx, 0, yaw) | 3 |
| joint_pos − default / joint_vel / actions | 12+12+12 |
| clock_inputs (2 hind feet) | 2 |

### `scripts/train_local.sh`

Model on `train.slurm` lines 76–115 with local paths; generic pass-through:
- `SIF_IMAGE=/home/daniel/Dev/rl_prj/isaac-lab-base.sif`, `ISAACLAB_DIR=/home/daniel/Dev/rl_prj/IsaacLab`, `RUN_DIR=/home/daniel/Dev/rob6323_go2_project`, `CACHE_ROOT=$HOME/.cache/isaac-local`, logs bind → `$RUN_DIR/logs/rsl_rl_local`
- `mkdir -p` all cache dirs (same set train.slurm creates), then `apptainer exec --nv --containall` with the same `-B` binds, `pip install -q -e /workspace/run/source/rob6323_go2`, then `/isaac-sim/python.sh ${SCRIPT} "$@"`
- `SCRIPT` env var (default `scripts/rsl_rl/train.py`) lets the same wrapper run `list_envs.py` / `zero_agent.py` / `random_agent.py` / `play.py`
- `chmod +x`

## Verification (this session — no full run)

```bash
# (a) registration
SCRIPT=/workspace/run/scripts/list_envs.py ./scripts/train_local.sh
# (b) smoke: shapes/indices/NaNs; zero-action robot should stand quadruped WITHOUT terminating
SCRIPT=/workspace/run/scripts/zero_agent.py   ./scripts/train_local.sh --task Template-Rob6323-Go2-Biped-Direct-v0 --num_envs 32 --headless
SCRIPT=/workspace/run/scripts/random_agent.py ./scripts/train_local.sh --task Template-Rob6323-Go2-Biped-Direct-v0 --num_envs 32 --headless
# (c) tiny end-to-end train: rsl-rl wiring + Episode_Reward/* logging
./scripts/train_local.sh --task Template-Rob6323-Go2-Biped-Direct-v0 --num_envs 64 --max_iterations 5 --headless
```
Sanity during (b): tracking rewards must be exactly 0 (gate closed at quadruped rest), `rew_upright` ≈ 1.0 raw. Full training later: `--num_envs 2048 --max_iterations 3000 --headless`.

## Risks (documented in spec, monitored during training)

1. **Rear-up exploration hardness** (main risk) — mitigations baked in: linear height ramp, ungated upright term, free front legs during transition, no base-contact termination, entropy 0.005, desynced resets. Fallback: raise upright scale to 4.0.
2. **Sitting-on-haunches reward hack** — countered by gated undesired-contact penalty + height-fine at 0.52 m; fallback: small ungated base-contact penalty (−0.2).
3. Torque authority (23.5 Nm knees) is marginal but realistic; solver iteration bump 4→8 available if contact jitter appears.
4. VRAM: fall back to 1024 envs + `num_steps_per_env=48` if 2048 OOMs.

## Verification results (2026-07-18, local WSL2 machine)

All smoke tests passed via `scripts/train_local.sh` (Apptainer sandbox; a native install is
impossible on this Ubuntu 20.04 WSL distro — Isaac Sim 5.1 binaries require glibc >= 2.34):

- `list_envs.py`: `Template-Rob6323-Go2-Biped-Direct-v0` registered alongside the quadruped task.
- `zero_agent.py` / `random_agent.py` (32 envs, headless): obs space (N, 50), action space (N, 12),
  no shape/index errors or NaNs over multi-minute runs.
- Tiny train (`--num_envs 64 --max_iterations 5 --headless`): completed end to end, mean reward
  increasing (25 -> 54), all `Episode_Reward/*` terms logged. Gate verified: every gated term
  (tracking, height_fine, tuck, contacts) is exactly 0 while the robot is not upright; ungated
  shaping (upright, height_ramp, roll, joint limits) active. Random-policy falls terminate via
  `fell_sideways` as expected. Logs: `logs/rsl_rl_local/rsl_rl/go2_biped_direct/<timestamp>/`.

WSL2 notes baked into `scripts/train_local.sh`: bind `/usr/lib/wsl` + `/dev/dxg` for CUDA
(the driver store 9p mount is required, `/usr/lib/wsl/lib` alone is not enough); the Vulkan
`ERROR_INCOMPATIBLE_DRIVER` messages at startup are non-fatal for headless physics-only runs;
`PYTHONUNBUFFERED=1` is required because kit can abort at shutdown before flushing stdout.

## Iteration 2 (2026-07-18, after run 1)

Run 1 (4096 envs) got stuck in a **crouch-walk local optimum** by ~iter 500: up_proj ~0.85
(vertical trunk), zero falls, tracking reward saturated — but base height pinned at the
quadruped ~0.30 m for 300+ iterations. The robot walked on deeply bent hind legs and had
no incentive to extend: tracking paid in full while crouched.

Changes:
1. **Tracking is now height-gated too**: `track_gate = upright_gate * clamp((h-0.35)/0.10, 0, 1)`
   applied to lin/yaw tracking and the gait contact reward. Crouch-walking no longer pays.
2. `height_ramp_reward_scale` 1.5 -> 3.0.
3. `height_fine` kernel widened (width 0.01 -> 0.02, cfg `height_fine_width`) so its gradient
   reaches further down the crouch.
4. New **ungated** `hind_calf_contact_penalty` (-0.2 per calf in contact): sitting back on the
   hind calves is never free, even below the uprightness gate.

## Iteration 3 (2026-07-18, after run 2)

Run 2 fixed the crouch-walk hack, and height began climbing (~iter 700-1100, pitched-up crouch
slowly extending). But at ~iter 1300 the policy found reward hack #2: **tip-toe quadruped** —
h_ramp was gated only by height, so standing tall on all four extended legs (up_proj ~0,
base ~0.39 m) collected ~1.2/step of height reward with no balance risk, and the policy
abandoned the rear-up entirely (upright per-step fell 1.79 -> 1.0; fell_forward face-plants
appeared).

Change: `height_ramp` now has its own **soft pitch gate** `clamp((up_proj-0.5)/0.4, 0, 1)`
(cfg `height_ramp_gate_lower/upper` = 0.5/0.9) — opens from ~30 deg pitch, full at ~64 deg.
Tall-quadruped earns nothing; the pitched-up crouch (up_proj ~0.8+) keeps ~75-100% of the
height gradient. The main uprightness gate (0.80/0.95) was NOT used here on purpose: the
crouch hovers at up_proj ~0.78-0.85 and would get near-zero height gradient under it.
