# gym-PlanarPeg

A 2.5D physical simulation environment for planar peg insertion built using MuJoCo and Gymnasium. It is designed to verify out-of-distribution (OOD) to in-distribution (ID) generalization and collect high-quality human demonstrations via teleoperation.

## Installation
Set up the Conda workspace and package:

```bash
# Clone the repository
git clone https://github.com/YSL2683/gym-PlanarPeg.git
cd gym-PlanarPeg

# Create and activate Conda environment
conda create -n planar_peg python=3.10 -y
conda activate planar_peg

# Install the package in editable mode (auto-installs MuJoCo, Gymnasium, Pygame, H5Py)
pip install -e .
```

## Description
The task in this environment is for a 3-DoF rectangular peg (agent), which is weld-constrained to a virtual mocap target, to safely navigate through a narrow barrier gap in a grid-like maze and dock into a C-shaped goal slot located on the opposite side. 

While the simulation runs in a 3D MuJoCo physics engine, the agent's motion is physically constrained to a 2D plane ($X, Y, \Theta$) via slide and hinge joints. The goal slot is visualized as a green C-shaped structure opening to the left, while the agent is a light blue rectangular block.

---

## Maze Layout & Variations
The maze map is encoded discretely using a 2D text grid (list of strings). The cell encoding contains 5 core symbols:
* `W` or `#`: **Boundary Wall** (rendered in gray: `rgba="0.3 0.3 0.3 1.0"`)
* `O`: **Obstacle Wall** (rendered in blue: `rgba="0.2 0.4 0.8 1.0"`)
* `S` or `P`: **Agent Reset Location** (Start position)
* `C` or `G`: **C-shaped Goal Slot** (opens to the left)
* `.`: **Empty Space**

### Default Layout (12x7 Grid)
The default configuration creates a vertical barrier in the middle with a single central gap at row index 3. The agent spawns on the far left, and the C-shaped goal is centered on the far right:
```python
grid = [
    "WWWWWWWWWWWW",
    "W....O.....W",  # Top barrier (blue)
    "W....O.....W",
    "W.S......C.W",  # Start (S) on left, Goal (C) on right, Gap in middle
    "W....O.....W",
    "W....O.....W",  # Bottom barrier (blue)
    "WWWWWWWWWWWW"
]
```

---

## Action Space
The action space is a `Box(-1.0, 1.0, (3,), float32)`. The elements represent the normalized absolute target coordinates for the virtual mocap control body:

| Num | Action | Control Min | Control Max | Mapping Target (Physical Scale) |
| :--- | :--- | :---: | :---: | :--- |
| **0** | Absolute X Coordinate | -1.0 | 1.0 | $[-x\_limit, x\_limit]$ meters |
| **1** | Absolute Y Coordinate | -1.0 | 1.0 | $[-y\_limit, y\_limit]$ meters |
| **2** | Absolute Theta (Yaw) | -1.0 | 1.0 | $[-\pi, \pi]$ radians |

---

## Observation Space
The observation space is a goal-aware dictionary consisting of 3 keys:
* `image_top`: Top-down static RGB viewpoint rendering of the entire maze (`Box(0, 255, (84, 84, 3), dtype=np.uint8)`).
* `image_front`: First-person forward-looking RGB viewpoint rendering attached to the front face of the rectangular peg (`Box(0, 255, (84, 84, 3), dtype=np.uint8)`).
* `proprioception`: 3-dimensional kinematic array of the actual rectangular peg's absolute world pose (`Box(-inf, inf, (3,), dtype=np.float32)`):

| Num | Observation | Min | Max | Unit | Description |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **0** | Peg X Coordinate | -Inf | Inf | meters | Absolute world position along X axis |
| **1** | Peg Y Coordinate | -Inf | Inf | meters | Absolute world position along Y axis |
| **2** | Peg Yaw ($\Theta$) | -Inf | Inf | radians | Absolute rotation around Z axis |

---

## Rewards
* **Success (Sparse):** When the agent successfully docks parallel inside the goal slot, a reward of **`+1.0`** is returned, and the episode terminates.
  * **Success Criteria**: 
    1. *X alignment:* Agent center must be parked inside the C-slot ($0.0 \le dx \le 0.06\text{m}$).
    2. *Y alignment:* Vertical offset from the goal center must be minimal ($|dy| \le 0.015\text{m}$).
    3. *Yaw alignment:* Angular offset from parallel docking must be within $\pm 5^\circ$ ($|d\theta| \le 0.087\text{rad}$).
* **Time Penalty:** To encourage efficiency, a small time cost penalty of **`-0.01`** is applied at every step until docking is achieved.

---

## Starting State
When the environment is reset:
1. The discrete start cell `S` is converted into continuous Cartesian $(x,y)$ coordinates.
2. A uniform random spatial noise within range **`[-0.05, 0.05]` meters** is added to the starting coordinates.
3. A uniform random angular noise within range **`[-10.0°, 10.0°]`** ($[-0.174, 0.174]\text{rad}$) is added to the starting orientation.
4. The control Mocap target body is automatically snapped and synchronized to this randomized spawn pose.

---

## Episode End
* **terminated:** True when the peg reaches the docking success threshold inside the C-shaped goal.
* **truncated:** True when the elapsed steps reach `max_episode_steps` (default: `200`).

---

