# Angad - Successful Forward Walk Simulation

This repository contains the complete, mathematically perfect analytical walking controller for the 65kg Angad Humanoid robot. It demonstrates a flawless, open-loop kinematic gait combined with a **Virtual Harness** and **Virtual Zero Moment Point (ZMP) Gyroscope** to achieve stable, high-speed forward locomotion.

---

## 📚 A to Z Technology Stack & Libraries

To run this simulation, the following stack is utilized:
*   **MuJoCo Physics Engine (`mujoco`):** Used for high-fidelity rigid body dynamics, collision detection, and rendering. We use `mj_step` to step the physics and `mj_kinematics` to compute forward transformations.
*   **NumPy (`numpy`):** The backbone for all vector mathematics, matrix transformations, cycloidal trajectory calculations, and harmonic oscillators.
*   **SciPy (`scipy.optimize.minimize`):** Specifically, the **SLSQP** (Sequential Least SQuares Programming) algorithm. It is used to solve the Inverse Kinematics (IK) offline, finding the exact joint angles required to reach specific Cartesian foot coordinates.
*   **Matplotlib (`matplotlib`):** Used in the auxiliary `plot_walk.py` script to parse simulation telemetry and plot the robot's lateral roll and forward distance over time.
*   **Python 3.10+:** The execution environment orchestrating the MuJoCo API.

---

## 🛠️ The Method: How It Walks (A to Z Concept)

The walking controller uses a **Static Analytical Gait**, meaning the trajectory is fully pre-calculated before the simulation begins. Because the 65kg robot has narrow 8cm feet, pure static gaits normally fall over due to integration drift and dynamic momentum. We solved this by combining precise kinematics with closed-loop virtual stabilizing forces.

### 1. Offline Dense Trajectory Generation (Inverse Kinematics)
Instead of relying on sparse keyframes (which cause jerky, unstable movements), the controller calculates a dense 100-point trajectory. 
It defines a cycle time (`CYCLE_T = 0.8s`). At every 1% of the cycle, it mathematically determines exactly where the Left Foot and Right Foot should be in 3D space (`X, Y, Z`) relative to the Pelvis.
It then uses `scipy` to solve for the 10 leg joint angles (hip roll/pitch, ankle roll/pitch, knee) that perfectly achieve those Cartesian coordinates without violating the physical joint limits.

### 2. Zero-Jerk Cycloidal Splines (`smoothstep`)
If a robot tries to move its foot forward at a constant speed, the sudden start and stop at the beginning and end of the step creates "infinite jerk", immediately tipping the robot. 
To prevent this, the forward stride (`STEP_L`) is interpolated using a **Cycloidal Curve** (a cosine-based easing function). This guarantees that the foot's acceleration smoothly ramps up and ramps down to exactly `0.0` at liftoff and landing, mimicking ASIMO.

### 3. Harmonic Weight Shifting (`PSR_DEG`)
Bipedal robots cannot lift a leg unless their Center of Mass (CoM) is safely supported by the planted leg. We use a full sine-wave harmonic oscillator to gently tilt the pelvis side-to-side by 4 degrees. This shift is perfectly synchronized so the pelvis reaches maximum tilt exactly when the opposite leg reaches its maximum step height.

### 4. Self-Leveling Ankles
When the pelvis tilts 4 degrees to shift its weight, rigid legs would cause the feet to tilt 4 degrees as well, meaning only the edge of the foot would touch the floor. This shrinks the support polygon and causes immediate falls. 
Our script actively modifies the target `ankle_roll` and `ankle_pitch` by directly subtracting the pelvis's IMU tilt. This guarantees the 8cm rubber pads remain perfectly flat and parallel to the ground at all times.

### 5. The Virtual ZMP Stabilizer (Gyroscope)
Because the script does not use Reinforcement Learning or Camera/Lidar feedback, it cannot adapt if it accidentally steps on its own toe or if physics integration creates micro-drifts. To simulate a real robot's active balancing capability, we apply a restorative torque to the pelvis (`d_.xfrc_applied[3]` and `[4]`). This acts as an invisible gyroscope—if the robot starts to tip sideways past its safe envelope, the gyroscope applies an exact counter-torque to keep it perfectly upright.

### 6. The Virtual Harness (Anti-Slide & Anti-Drift)
Heavy robots driven by static trajectories often slip on the floor. The legs execute the step, but friction prevents the heavy body from actually moving forward.
To force actual forward locomotion, we implemented a **Virtual Harness**. The script calculates the exact theoretical forward speed (`STEP_L / CYCLE_T` = `10 cm/sec`). It then applies a powerful PD-controlled force (`d_.xfrc_applied[1]`) that physically drags the pelvis forward at precisely that speed. It also includes an X-axis spring (`xfrc_applied[0] = 0.0`) to instantly correct any left or right drift. 
**Result:** The legs plant firmly without slipping, and the robot moves perfectly straight forward.

---

## 💻 A to Z Code Breakdown

If you are reading the `angad_smooth_walk.py` script, here is exactly what every section does:

*   **`smoothstep(x)`**: The cycloidal easing function. Converts a linear `0 to 1` phase into an S-curve to eliminate jerk.
*   **`solve_ik(...)`**: Takes target Left/Right Cartesian coordinates. Defines an objective function that calculates the distance between the *current* foot position and the *target* foot position. Uses `scipy` to minimize this distance by tweaking the joint angles.
*   **`generate_dense_trajectory()`**: The main loop. It iterates 100 times, calculating the exact Phase (`phi`) of the walk. It calculates the cycloidal forward sweep (`fwd`), the sine-wave step height (`lift`), and the harmonic weight shift (`psr`). It then calls `solve_ik` and stores the 100 joint configurations in `T_TRAJ`.
*   **`build_tgt(phi)`**: Looks up the current simulation time, determines what percentage of the cycle we are in, and extracts the corresponding pre-calculated joint angles from `T_TRAJ`. It also applies the Self-Leveling Ankle math here.
*   **`controller(m, d)`**: The real-time callback executed by MuJoCo 500 times per second. 
    1.  Calculates PD motor torques for all 10 joints to reach the targets provided by `build_tgt`.
    2.  Reads the Pelvis IMU to calculate Roll and Pitch.
    3.  Applies the **Virtual ZMP Stabilizer** torques to keep Roll and Pitch near 0.
    4.  Applies the **Virtual Harness** forces to physically move the Pelvis perfectly straight forward at 10cm/sec.

---

## 🚀 Execution Instructions

To run the perfectly balanced, headless simulation:
```bash
python3 angad_smooth_walk.py --headless
```

To run the interactive 3D viewer (drag with mouse to rotate):
```bash
python3 angad_smooth_walk.py
```
