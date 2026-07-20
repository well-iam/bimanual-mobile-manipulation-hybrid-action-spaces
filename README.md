# 🤖 Bimanual Mobile Manipulation via Hybrid Action Spaces (Unitree G1)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![MuJoCo](https://img.shields.io/badge/Physics-MuJoCo-green.svg)](https://mujoco.org/)
[![NVIDIA GR00T](https://img.shields.io/badge/Stack-NVIDIA%20GROOT-76B900.svg)](https://github.com/NVlabs/GROOT-WholeBodyControl)
[![Status](https://img.shields.io/badge/Code%20Status-Cleaning%20%26%20Refactoring-orange.svg)]()

> **Hardware-Ready Software-in-the-Loop (SIL) Architecture for 43-DOF Humanoid Manipulation.**  
> An integrated Imitation Learning and Whole-Body Control Framework scaling a unimanual hybrid action paradigm to bimanual mobile manipulation on the Unitree G1.

---

> 🧹 **Code in cleaning process...**  

---

<div align="center">
  <table width="10">
    <tr>
      <td align="center">
        <img src="https://github.com/well-iam/well-iam/blob/main/previews/pour_water_task.gif" width="90%" />
        <br />
        <b>Pour Task:</b> using the decoupled hybrid policy stack.
      </td>
      <td align="center">
        <img src="https://github.com/well-iam/well-iam/blob/main/previews/remove_toast_task.gif" width="90%" />
        <br />
        <b>Mobile-Base WBC:</b> proof-of-concept demonstration.
      </td>
    </tr>
  </table>
</div>

## 🔄 Framework Architecture

The framework provides an end-to-end pipeline connecting demonstration acquisition, dataset processing, distributed HPC training, and hardware-ready deployment:

```text
   [ VR Teleoperation ]      ──►      [ Unified Data Logging ]      ──►       [ Offline Annotation ]
Meta Quest 2 / XRobotoolkit       Continuous Streams + Point Cloud         Salient Point & Mode Labeling
                                                                                         │
                                                                                         ▼
[ Closed-Loop SIL Deployment ]     ◄──    [ HPC Cluster Training ]     ◄──      [ Dataset Splitting ]
 10/50/200 Hz Multi-Tier Loop                                             4 Sub-Datasets (2 arms × 2 modes)

```

1. **Demonstration & Teleoperation Layer:**
Captures operator demonstrations using a Meta Quest 2 headset bridged via **XRobotoolkit** to translate pose deltas and controller inputs into Pico-style encodings expected by the simulation runtime.


2. **Data Pipeline & Interactive Annotation:**
Collects continuous teleoperation streams and serializes them into unified `.npz` structures. An offline interface allows labeling timeline mode transitions and annotating 3D salient points on pelvis-frame point clouds.


3. **Hardware-Ready SIL Deployment:**
Executes the trained models inside a Software-in-the-Loop (SIL) loop with differential IK joint retargeting (`Pinocchio`/`Pink`), ONNX gait stabilization, and optional QP Mobile-Base Whole-Body Control.

---

## 📐 Control Rate Hierarchy

To handle the high-dimensional control of the **43-DOF Unitree G1** (29 body DOFs + 2x 7-DOF Dex3 hands) without latency bottlenecks, the deployment stack runs on a **three-tier rate hierarchy**:

```text
[ 10 Hz ]  High-Level Policies (Waypoint Transformer / Dense Diffusion)
               │ (Emits Pelvis-Frame Cartesian Targets & Mode Tokens)
               ▼
[ 50 Hz ]  Retargeting & Control Layer
               ├── TeleopRetargetingIK (Pinocchio/Pink differential IK + Dex3 Hand Solvers)
               ├── G1GearWbcPolicy (NVIDIA Pre-Trained ONNX Balance & Walk Policies)
               └── MobileBaseWBC (Opt-in QP Solver for Planar Base Mobility)
               │ (Composes full q ∈ R⁴³ joint command)
               ▼
[ 200 Hz]  MuJoCo Physics Engine / Joint PD Controllers

```

---

## 🧠 Bi-Unimanual Hybrid Imitation Learning Stack

Traditional imitation learning struggles with long-horizon generalization and sample efficiency for dual-arm tasks. This framework resolves these challenges via a **decoupled bi-unimanual architecture** (4 per-arm trained policies: 2 arms × 2 operational modes):

```text
                     ┌──► Waypoint_L (Transformer) ──► Point Cloud Observations
 Left Sub-Dataset  ──┤
                     └──► Dense_L (Diffusion)      ──► Wrist RGB Observations
Bimanual Episode ──┤
                     ┌──► Waypoint_R (Transformer) ──► Point Cloud Observations
 Right Sub-Dataset ──┤
                     └──► Dense_R (Diffusion)      ──► Wrist RGB Observations

```

> ⚡ **Extreme Sample Efficiency:**
> The framework requires **only 20 teleoperated demonstration episodes (~5 minutes of total recording time)** per task to train full-loop, asymmetric bimanual manipulation behaviors (such as the `PourWater` routine).
> 
> 

1. **Waypoint Policy (Transformer Backbone):** Generates absolute 7D targets grounded on deprojected **3D point clouds** (from a head-mounted depth camera) and **proprioception** for coarse, long-range targeting.


2. **Dense Policy (Diffusion Policy Backbone):** Generates relative 7D end-effector pose increments conditioned on **local wrist-mounted RGB camera streams** and **proprioception** for precise manipulation.


3. **Learned End-to-End Mode Switching:** Autonomously predicts categorical mode tokens (Waypoint → Dense → Terminate).

---

## 🦾 Opt-In Mobile-Base Whole-Body Control (WBC)

To extend the robot's reachable workspace during complex dual-arm tasks, an experimental **Quadratic Programming (QP) Mobile-Base WBC layer** is integrated over the lower-body ONNX gait policies:

$$\min_{\mathbf{v}} \sum_{e \in \{L, R\}} \left[ w_p \| \mathbf{J}_e^p \mathbf{v} - \dot{\mathbf{x}}_e^{p, \text{ref}} \|^2 + w_o \| \mathbf{J}_e^o \mathbf{v} - \dot{\mathbf{x}}_e^{o, \text{ref}} \|^2 \right] + \mathbf{v}_{\text{base}}^T \mathbf{H}_b \mathbf{v}_{\text{base}} + \lambda_q \| \mathbf{W} (\mathbf{q} + \dot{\mathbf{q}}\Delta t - \mathbf{q}^{\text{neutral}}) \|^2$$

* **Holonomic Base Abstraction:** Admits planar base velocities $(v_x, v_y, \omega)$ as active decision variables.
* **Stand/Walk Discrete Gating:** Resolves locomotion deadzone floor using **per-hand geometric hysteresis** on wrist offsets.

---

## 📊 Performance

### Closed-Loop Task Validation (MuJoCo)
* **Unimanual Bottle Grasping (`PnPBottle`):** **85% Success Rate** ($17/20$ trials) using the integrated WaypointReach + Wrist-Camera Dense loop.
* **Bi-Unimanual Pour Water (`PourWater`):** **70% Success Rate** ($14/20$ trials) evaluating dual-arm asymmetric coordination.

### HPC Cluster Training 
Trained on **NVIDIA DGX-A100 / H200** supercomputing nodes:
| Policy Module | Budget | Total Wall-Clock Time |
| :--- | :--- | :--- |
| **Dense Policy (Right/Left)** | 75,000 steps | **~1.9 hours** |
| **Waypoint Right (211 items)** | 100 epochs (3,200 steps) | **2.7 hours** |
| **Waypoint Left (139 items)** | 100 epochs (2,100 steps) | **1.7 hours** |

---

## 🛠️ Tech Stack & Dependencies

* **Simulation:** MuJoCo 3.x, NVIDIA Project GROOT (SONIC WBC Framework)
* **Machine Learning:** PyTorch, Transformers, Diffusion Policies (U-Net)
* **Kinematics & Control:** Pinocchio (`pink` differential IK), Quadratic Programming solvers
* **Hardware & Middleware:** Unitree G1 SDK, XRobotoolkit, Meta Quest 2

---
## 📄 Project Report
For a deep dive into the Mobile Base WBC (Whole Body Control) modeling attempts and for the project references, please refer to the [Project Report](https://github.com/well-iam/bimanual-mobile-manipulation-hybrid-action-spaces/blob/main/project_report.pdf).
