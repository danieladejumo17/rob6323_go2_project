# ROB6323 Go2 Project — Isaac Lab

This repository is the starter code for the NYU Reinforcement Learning and Optimal Control project in which students train a Unitree Go2 walking policy in Isaac Lab starting from a minimal baseline and improve it via reward shaping and robustness strategies. Please read this README fully before starting and follow the exact workflow and naming rules below to ensure your runs integrate correctly with the cluster scripts and grading pipeline.

## Repository policy

- Fork this repository and do not change the repository name in your fork.  
- Your fork must be named rob6323_go2_project so cluster scripts and paths work without modification.

### Prerequisites

- **GitHub Account:** You must have a GitHub account to fork this repository and manage your code. If you do not have one, [sign up here](https://github.com/join).

### Links
1.  **Project Webpage:** [https://machines-in-motion.github.io/RL_class_go2_project/](https://machines-in-motion.github.io/RL_class_go2_project/)
2.  **Project Tutorial:** [https://github.com/machines-in-motion/rob6323_go2_project/blob/master/tutorial/tutorial.md](https://github.com/machines-in-motion/rob6323_go2_project/blob/master/tutorial/tutorial.md)

## Connect to Greene

- Connect to the NYU Greene HPC via SSH; if you are off-campus or not on NYU Wi‑Fi, you must connect through the NYU VPN before SSHing to Greene.  
- The official instructions include example SSH config snippets and commands for greene.hpc.nyu.edu and dtn.hpc.nyu.edu as well as VPN and gateway options: https://sites.google.com/nyu.edu/nyu-hpc/accessing-hpc?authuser=0#h.7t97br4zzvip.

## Clone in $HOME

After logging into Greene, `cd` into your home directory (`cd $HOME`). You must clone your fork into `$HOME` only (not scratch or archive). This ensures subsequent scripts and paths resolve correctly on the cluster. Since this is a private repository, you need to authenticate with GitHub. You have two options:

### Option A: Via VS Code (Recommended)
The easiest way to avoid managing keys manually is to configure **VS Code Remote SSH**. If set up correctly, VS Code forwards your local credentials to the cluster.
- Follow the [NYU HPC VS Code guide](https://sites.google.com/nyu.edu/nyu-hpc/training-support/general-hpc-topics/vs-code) to set up the connection.

> **Tip:** Once connected to Greene in VS Code, you can clone directly without using the terminal:
> 1. **Sign in to GitHub:** Click the "Accounts" icon (user profile picture) in the bottom-left sidebar. If you aren't signed in, click **"Sign in with GitHub"** and follow the browser prompts to authorize VS Code.
> 2. **Clone the Repo:** Open the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`), type **Git: Clone**, and select it.
> 3. **Select Destination:** When prompted, select your home directory (`/home/<netid>/`) as the clone location.
>
> For more details, see the [VS Code Version Control Documentation](https://code.visualstudio.com/docs/sourcecontrol/intro-to-git#_clone-a-repository-locally).

### Option B: Manual SSH Key Setup
If you prefer using a standard terminal, you must generate a unique SSH key on the Greene cluster and add it to your GitHub account:
1. **Generate a key:** Run the `ssh-keygen` command on Greene (follow the official [GitHub documentation on generating a new SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent#generating-a-new-ssh-key)).
2. **Add the key to GitHub:** Copy the output of your public key (e.g., `cat ~/.ssh/id_ed25519.pub`) and add it to your account settings (follow the [GitHub documentation on adding a new SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)).

### Execute the Clone
Once authenticated, run the following commands. Replace `<your-git-ssh-url>` with the SSH URL of your fork (e.g., `git@github.com:YOUR_USERNAME/rob6323_go2_project.git`).
```
cd $HOME
git clone <your-git-ssh-url> rob6323_go2_project
```
*Note: You must ensure the target directory is named exactly `rob6323_go2_project`. This ensures subsequent scripts and paths resolve correctly on the cluster.*
## Install environment

- Enter the project directory and run the installer to set up required dependencies and cluster-side tooling.  
```
cd $HOME/rob6323_go2_project
./install.sh
```
Do not skip this step, as it configures the environment expected by the training and evaluation scripts. It will launch a job in burst to set up things and clone the IsaacLab repo inside your greene storage. You must wait until the job in burst is complete before launching your first training. To check the progress of the job, you can run `ssh burst "squeue -u $USER"`, and the job should disappear from there once it's completed. It takes around **30 minutes** to complete. 
You should see something similar to the screenshot below (captured from Greene):

![Example burst squeue output](docs/img/burst_squeue_example.png)

In this output, the **ST** (state) column indicates the job status:
- `PD` = pending in the queue (waiting for resources).
- `CF` = instance is being configured.
- `R`  = job is running.

On burst, it is common for an instance to fail to configure; in that case, the provided scripts automatically relaunch the job when this happens, so you usually only need to wait until the job finishes successfully and no longer appears in `squeue`.

## What to edit

- In this project you'll only have to modify the two files below, which define the Isaac Lab task and its configuration (including PPO hyperparameters).  
  - source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/rob6323_go2_env.py  
  - source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/rob6323_go2_env_cfg.py
PPO hyperparameters are defined in source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/agents/rsl_rl_ppo_cfg.py, but you shouldn't need to modify them.

## How to edit

- Option A (recommended): Use VS Code Remote SSH from your laptop to edit files on Greene; follow the NYU HPC VS Code guide and connect to a compute node as instructed (VPN required off‑campus) (https://sites.google.com/nyu.edu/nyu-hpc/training-support/general-hpc-topics/vs-code). If you set it correctly, it makes the login process easier, among other things, e.g., cloning a private repo.
- Option B: Edit directly on Greene using a terminal editor such as nano.  
```
nano source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/rob6323_go2_env.py
```
- Option C: Develop locally on your machine, push to your fork, then pull changes on Greene within your $HOME/rob6323_go2_project clone.

> **Tip:** Don't forget to regularly push your work to github

## Launch training

- From $HOME/rob6323_go2_project on Greene, submit a training job via the provided script.  
```
cd "$HOME/rob6323_go2_project"
./train.sh
```
- Check job status with SLURM using squeue on the burst head node as shown below.  
```
ssh burst "squeue -u $USER"
```
Be aware that jobs can be canceled and requeued by the scheduler or underlying provider policies when higher-priority work preempts your resources, which is normal behavior on shared clusters using preemptible partitions.

## Where to find results

- When a job completes, logs are written under logs in your project clone on Greene in the structure logs/[job_id]/rsl_rl/go2_flat_direct/[date_time]/.  
- Inside each run directory you will find a TensorBoard events file (events.out.tfevents...), neural network checkpoints (model_[epoch].pt), YAML files with the exact PPO and environment parameters, and a rollout video under videos/play/ that showcases the trained policy.  

## Download logs to your computer

Use `rsync` to copy results from the cluster to your local machine. It is faster and can resume interrupted transfers. Run this on your machine (NOT on Greene):

```
rsync -avzP -e 'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null' <netid>@dtn.hpc.nyu.edu:/home/<netid>/rob6323_go2_project/logs ./
```

*Explanation of flags:*
- `-a`: Archive mode (preserves permissions, times, and recursive).
- `-v`: Verbose output.
- `-z`: Compresses data during transfer (faster over network).
- `-P`: Shows progress bar and allows resuming partial transfers.

## Visualize with TensorBoard

You can inspect training metrics (reward curves, loss values, episode lengths) using TensorBoard. This requires installing it on your local machine.

1.  **Install TensorBoard:**
    On your local computer (do NOT run this on Greene), install the package:
    ```
    pip install tensorboard
    ```

2.  **Launch the Server:**
    Navigate to the folder where you downloaded your logs and start the server:
    ```
    # Assuming you are in the directory containing the 'logs' folder
    tensorboard --logdir ./logs
    ```

3.  **View Metrics:**
    Open your browser to the URL shown (usually `http://localhost:6006/`).

## Debugging on Burst

Burst storage is accessible only from a job running on burst, not from the burst login node. The provided scripts do not automatically synchronize error logs back to your home directory on Greene. However, you will need access to these logs to debug failed jobs. These error logs differ from the logs in the previous section.

The suggested way to inspect these logs is via the Open OnDemand web interface:

1.  Navigate to [https://ood-burst-001.hpc.nyu.edu](https://ood-burst-001.hpc.nyu.edu).
2.  Select **Files** > **Home Directory** from the top menu.
3.  You will see a list of files, including your `.err` log files.
4.  Click on any `.err` file to view its content directly in the browser.

> **Important:** Do not modify anything inside the `rob6323_go2_project` folder on burst storage. This directory is managed by the job scripts, and manual changes may cause synchronization issues or job failures.

## Project scope reminder

- The assignment expects you to go beyond velocity tracking by adding principled reward terms (posture stabilization, foot clearance, slip minimization, smooth actions, contact and collision penalties), robustness via domain randomization, and clear benchmarking metrics for evaluation as described in the course guidelines.  
- Keep your repository organized, document your changes in the README, and ensure your scripts are reproducible, as these factors are part of grading alongside policy quality and the short demo video deliverable.

## Resources

- [Isaac Lab documentation](https://isaac-sim.github.io/IsaacLab/main/source/setup/ecosystem.html) — Everything you need to know about IsaacLab, and more!
- [Isaac Lab ANYmal C environment](https://github.com/isaac-sim/IsaacLab/tree/main/source/isaaclab_tasks/isaaclab_tasks/direct/anymal_c) — This targets ANYmal C (not Unitree Go2), so use it as a reference and adapt robot config, assets, and reward to Go2.
- [DMO (IsaacGym) Go2 walking project page](https://machines-in-motion.github.io/DMO/) • [Go2 walking environment used by the authors](https://github.com/Jogima-cyber/IsaacGymEnvs/blob/e351da69e05e0433e746cef0537b50924fd9fdbf/isaacgymenvs/tasks/go2_terrain.py) • [Config file used by the authors](https://github.com/Jogima-cyber/IsaacGymEnvs/blob/e351da69e05e0433e746cef0537b50924fd9fdbf/isaacgymenvs/cfg/task/Go2Terrain.yaml) — Look at the function `compute_reward_CaT` (beware that some reward terms have a weight of 0 and thus are deactivated, check weights in the config file); this implementation includes strong reward shaping, domain randomization, and training disturbances for robust sim‑to‑real, but it is written for legacy IsaacGym and the challenge is to re-implement it in Isaac Lab.
- **API References**:
    - [ArticulationData (`robot.data`)](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html#isaaclab.assets.ArticulationData) — Contains `root_pos_w`, `joint_pos`, `projected_gravity_b`, etc.
    - [ContactSensorData (`_contact_sensor.data`)](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sensors.html#isaaclab.sensors.ContactSensorData) — Contains `net_forces_w` (contact forces).

---
# Changes Made for Gait Robustness
## I. Domain Randomization
### *Actuator Friction Model with Randomization*
> To run the environment with Actuator Friction model the files `friction_rob6323_go2_env.py` and `friction_rob6323_go2_env_cfg.py` should be renamed to `rob6323_go2_env.py` and `rob6323_go2_env_cfg.py`. The original `rob6323_go2_env.py` and `rob6323_go2_env_cfg.py` should be given a temporary name.

An actuator friction model was created by subtracting the friction torque, $\tau_{\text{friction}}$, from the output torque from the PD controller, $\tau_{\text{PD}}$.

#### Actuator Friction Model

The friction torque is defined as:

$$
\tau_{\text{friction}} = \tau_{\text{stiction}} + \tau_{\text{viscous}}
$$

$$
\tau_{\text{stiction}} = F_s \cdot \tanh\!\left(\frac{\dot{q}}{0.1}\right)
$$

$$
\tau_{\text{viscous}} = \mu_v \cdot \dot{q}
$$

The PD controller torque is modified as:

$$
\tau_{PD} \leftarrow \tau_{PD} - \tau_{\text{friction}}
$$

During environment reset, the randomized parameters are sampled as:

$$
\mu_v^{\epsilon} \sim \mathcal{U}(0.0, 0.3), \quad
F_s^{\epsilon} \sim \mathcal{U}(0.0, 2.5)
$$

where $\dot{q}$ is the joint velocity, $F_s$ is the stiction coefficient, and $\mu_v$ is the viscous coefficient. This forces the policy to overcome internal joint resistance, significantly narrowing the sim-to-real gap.

## II. Reward Shaping
To achieve gait robustness, the following princinpled reward terms were added in addition to the velocity tracking terms. All reward terms are implemented in `rob6323_go2_env.py` and are scaled by reward scales defined in the `rob6323_go2_env_cfg.py`.


### **Base Stability Rewards:**
### 1. Base Orientation Penalty
<!-- Describe this penalty and why it was added -->
<!-- Reference the function where this term is implemented and how -->
To prevent the robot from tipping over or walking with a tilt, this penalty minimizes the projection of gravity onto the robot's horizontal plane. It encourages the robot's vertical axis to align with the global gravity vector.

This reward is implemented in the `_penalty_roll_pitch` function. It calculates the sum of squares of the projected gravity vector's x and y components. We penalize only significant deviations to allow for small angles for locomotion.

### 2. Base Roll and Pitch (Angular Velocity) Penalty
To reduce wobbling and ensure a stable upper body, this term penalizes high angular velocities in the roll and pitch axes. This dampens oscillations in the robot's base.

It is mplemented in the `_penalty_roll_pitch` function. We also penalize only significant angular X and Y velocities of the robot's base.

### 3. Base Vertical (Z) Velocity Penalty
To prevent the robot from hopping unnecessarily or jittering vertically while walking on flat ground, this penalty discourages vertical movement of the base.

Implemented in the `_penalty_vertical_velocity` function. It penalizes the square of the base's linear velocity in the z-direction.

### 4. Base Height Error Penalty
This term ensures the robot maintains a specific, optimal standing height defined in the configuration. It prevents the robot from crawling too low or standing too high on tiptoes.

Implemented in the `_penalty_base_height_error` function. It calculates the squared difference between the current root height and the target_height.

### 5. Undesired Contact Penalty
To prevent the robot from falling or bumping its knees, this term penalizes collisions on any part of the robot body except the feet (e.g., thighs, calves, or base).

Implemented in the `_penalty_undesired_contacts` function. It checks the net external contact forces on all bodies not listed in feet_indices and applies a penalty if forces are detected.

### **Action Regularization and Smoothness Rewards:**
### 6. Torque Regularization
To encourage energy efficiency and prevent motor overheating, this term penalizes high torque output from the actuators.

Implementation: Implemented in the `_penalty_torque_magnitude` function. It computes the mean of squared torques applied to all joints, which is then normalized by torque-limits-squared to make it scale-invariant.

### 7. Action Rate Penalty
This penalizes the rate of change in actions (the difference between the current action and the previous action). It encourages smooth control signals and prevents high-frequency oscillation in the actuators.

Implemented in the `_penalty_action_rate` function. It computes the mean of the squared difference `self._actions - self._previous_actions`.

### 8. Action Jerk Penalty
The action jerk penalty (second derivative) was separated from action rate (first derivative) penalty so that each can have different reward scales. This penalizes the acceleration of the action signal, further smoothing the motion.

Implementation: Implemented in the `_penalty_action_jerk` function. It uses a finite difference approximation of the second derivative: `self._actions - 2*self._previous_actions + self._pre_previous_actions`.

### 9. Raibert Heuristic Reward
This reward encourages the robot to place its feet in locations that stabilize the base velocity, following the Raibert heuristic (foot placement based on velocity). It guides the policy toward physically realistic locomotion strategies.

Implementation: Implemented in the `_reward_raibert_heuristic` function.

### 10. Feet Clearance
To prevent tripping, this term encourages the feet to lift to a specific height during the swing phase. It penalizes swing feet that are moving fast but remain close to the ground.

Implementation: Implemented in the `_reward_feet_clearance` function. It identifies feet in the swing phase and penalizes them if their height is below `foot_clearance_target` as defined in `rob6323_go2_env_cfg`.

### 11. Feet Contact Forces
This term encourages the robot to generate contact forces that match the desired contact schedule (swing vs. stance). It rewards applying force when in stance and zero force when in swing, ensuring the physical contacts align with the gait timing.

Implementation: Implemented in the `_reward_tracking_contacts_shaped_force` function. It computes the absolute error between the normalized contact force magnitude and the desired contact state (0 or 1), rewarding minimizing this error.

### 12. Joint Velocity Penalty
This acts as a damping term for the joints, penalizing high rotational velocities. It helps reduce vibrations and ensures the robot operates within safe mechanical limits.

Implementation: Implemented in the `_get_rewards` function. It penalizes the sum of squared joint velocities.

### **Gait Shaping Rewards:**
### 13. Diagonal Phase Consistency Reward
This enforces a trot gait by rewarding synchronization between diagonal leg pairs (e.g., Front-Left and Rear-Right should move together).

Implementation: Implemented in the `_reward_diagonal_phase_consistency` function. It computes the error between the clock inputs of diagonal pairs.

### 14. Duty Factor Reward
This ensures the feet spend the correct proportion of the gait cycle on the ground (stance phase) versus in the air (swing phase), matching the desired gait parameters.

Implementation: Implemented in the `_reward_duty_factor` function. It calculates the mean error between the actual contact timing and the desired duty factor provided by the clock.

### 15. Symmetry Reward
This encourages time-symmetric gait patterns, ensuring that the phase offset between legs matches the desired configuration (e.g., legs shouldn't be out of phase in a way that creates a limp).

Implementation: Implemented in the `_reward_symmetry function`. It penalizes deviations from the expected phase differences between specific leg pairs.

### 16. Pacing Penalty
To prevent the robot from entering a "pace" gait (where lateral pairs like Front-Left and Rear-Left move together), this term penalizes synchronization of lateral legs.

Implementation: Implemented in the `_penalty_pacing function`. It applies a penalty if the clock inputs of lateral pairs are too similar.

### 17. Hopping Penalty
To prevent the robot from bounding or pronking (hopping with front or back legs together) when a trot is desired, this term penalizes synchronization of the front pair or the rear pair.

Implementation: Implemented in the `_penalty_hopping function`. It applies a penalty if the clock inputs of the front legs or rear legs are too similar.

## III. Early Stopping
To speed up learning, an episode terminates early if specific failure conditions are met.

### 1. Base Height Limits

The episode terminates if the robot's base height falls below a critical threshold (fall) or exceeds a maximum.

Implementation: Implemented in `_get_dones`. Checks `self.robot.data.root_pos_w[:, 2]` to get the base Z position in the world frame.

### 2. Base Orientation (Upside Down)

The episode terminates if the robot flips over. This is detected by checking the projection of the gravity vector onto the robot's body Z-axis. If the component is positive, it implies the robot is upside down.

Implementation: Implemented in `_get_dones`. It checks self.robot.data.projected_gravity_b[:, 2] > 0.

### 3. Undesired Base Contact (Crash)

The episode terminates if the robot's base (torso) makes contact with the ground. This prevents the robot from learning strategies that involve dragging the body.

Implementation: Implemented in `_get_dones`. It uses the contact sensor to check for contacts on the base.

---

# Controlled Backflip with Recovery (Bonus)
> To run the environment for the controlled backflip with recovery task, the files `backflip_rob6323_go2_env.py` and `backflip_rob6323_go2_env_cfg.py` should be renamed to `rob6323_go2_env.py` and `rob6323_go2_env_cfg.py`. The original `rob6323_go2_env.py` and `rob6323_go2_env_cfg.py` should be given a temporary name.

For the controlled backflip with recovery task, the following changes were made to the environment configuration and rewards. However, we could not achieve a full backflip with a recovery landing on the four legs yet. The robot manages to make a turn (around the axis  for pitch), but lands on its back.

- Stage-Based State Machine: We introduced a 5-stage scheduler (stand, sit, jump, air, land) managed by base height and contact sensors to orchestrate the backflip maneuver.

- Observation Space Expansion: We expanded observations to include backflip context: _a normalized stage index, stage timer, and integrated turn progress._ 

- Action Scaffolding: We replaced static default joint positions with dynamic, stage-dependent nominal poses (e.g., crouching during sit, tucking legs during jump/air).

- Dynamic Action Scaling: Applied a multiplier (backflip_action_scale_mult) to boost control authority specifically during the high-power jump and air stages.


- Removal of Locomotion Rewards: We eliminated gait-specific terms like _duty_factor, raibert_heuristic, phase_consistency, and feet_clearance._

- Stage-Wise Reward Shaping: We implemented distinct reward weights and objectives for each stage (e.g., height reward decrease in sit vs. height reward increase in jump).


- Rotation and Balance Objectives

  -  Velocity: We rewarded high-pitch angular velocity during jump and air (previously penalized in walking) while penalizing it during land.


  - Turn Progress: We added a specific reward for accumulating pitch rotation.

  - Orientation: We changed orientation targets to allow/encourage rotation during flight while enforcing upright stability during landing.


- Inverted Posture Allowance: We modified termination criteria to allow the robot to be upside-down during jump and air stages (previously a crash condition).

### Reference:

> **Stage-Wise Reward Shaping for Acrobatic Robots: A Constrained Multi-Objective Reinforcement Learning Approach**  
> *arXiv:2409.15755v1, 2024*




