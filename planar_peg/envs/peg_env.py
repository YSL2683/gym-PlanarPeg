import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from typing import Dict, Any, Tuple, Optional, Union

from planar_peg.envs.maze_model import MazeModel
from planar_peg.envs.map import get_map

class PlanarPegEnv(gym.Env):
    """
    PlanarPegEnv is a 2.5D physical environment built using MuJoCo and Gymnasium.
    The agent is a rectangular peg constrained to move in a 2D plane (X, Y, Theta).
    Impedance-like control is realized via a weld-constrained mocap body.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        grid: Optional[Union[str, list]] = "default",
        cell_size: float = 0.2,
        n_substeps: int = 5,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 250,
    ):
        """
        Initializes the environment.

        Args:
            grid (Optional[Union[str, list]]): The grid map layout. Can be a string (map name) or a list of strings.
            cell_size (float): Physical size of each grid cell.
            n_substeps (int): Number of simulator substeps per environment step.
            render_mode (Optional[str]): Supports 'human' (passive viewer) and 'rgb_array'.
            max_episode_steps (int): Timeout limit for episodes.
        """
        super().__init__()
        
           
        if isinstance(grid, str):
            self.grid = get_map(grid)
        else:
            self.grid = grid
        self.cell_size = cell_size
        self.n_substeps = n_substeps
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self._elapsed_steps = 0
        
        # Load and parse map geometry
        self.maze_model = MazeModel(grid=self.grid, cell_size=self.cell_size)
        self.xml_string = self.maze_model.generate_xml_string()
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_string(self.xml_string)
        self.data = mujoco.MjData(self.model)
        
        # Identify control targets
        self.mocap_id = self.model.body("mocap").mocapid[0]
        self.start_pos = self.maze_model.start_pos
        self.goal_pos = self.maze_model.goal_pos
        self.current_goal_pos = self.goal_pos
        self.current_goal_theta = 0.0
        
        # Physical boundary limits for scaling action inputs
        self.x_limit = (self.maze_model.width * self.cell_size) / 2.0
        self.y_limit = (self.maze_model.height * self.cell_size) / 2.0
        
        # Define Spaces
        # Action space: X, Y, and Theta (Z-axis rotation) in absolute physical coordinates
        self.action_space = spaces.Box(
            low=np.array([-self.x_limit, -self.y_limit, -np.pi]),
            high=np.array([self.x_limit, self.y_limit, np.pi]),
            dtype=np.float32
        )
        
        # Observation space: top/front RGB views (224x224x3) + proprioception pose [X, Y, Theta]
        self.observation_space = spaces.Dict({
            "observation.images.top": spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
            "observation.images.front": spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
            "observation.state": spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
            )
        })
        
        # Renderer for multi-view image observations
        self.renderer = mujoco.Renderer(self.model, height=224, width=224)
        
        # Lazy initialization for Passive Viewer
        self.viewer = None
        
    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Captures images and agent's physical joints to build the observation dictionary."""
        # 1. Render Top-view
        self.renderer.update_scene(self.data, camera="top")
        obs_images_top = self.renderer.render()
        
        # 2. Render Front-view (1st person)
        self.renderer.update_scene(self.data, camera="front")
        obs_images_front = self.renderer.render()
        
        # 3. Read agent physical coordinates (x, y, theta) from absolute world coordinates
        # Using body xpos directly to get global X and Y coordinates.
        x = self.data.body("agent").xpos[0]
        y = self.data.body("agent").xpos[1]
        theta = self._normalize_angle(self.data.qpos[2])
        
        obs_state = np.array([x, y, theta], dtype=np.float32)
        
        return {
            "observation.images.top": obs_images_top,
            "observation.images.front": obs_images_front,
            "observation.state": obs_state
        }
        
    def _normalize_angle(self, angle: float) -> float:
        """Helper to wrap orientation angles within [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _check_success(self, x: float, y: float, theta: float) -> bool:
        """
        Evaluates whether the agent is fully inside the C-shaped goal pocket.
        Since the pocket is narrow, we only need to check if the agent's center
        is deep enough inside the pocket.
        """
        gx, gy = self.current_goal_pos
        g_theta = self.current_goal_theta
        
        g_cos = np.cos(g_theta)
        g_sin = np.sin(g_theta)
        
        # Transform agent center (world) to goal pocket local frame
        dx_world = x - gx
        dy_world = y - gy
        lx = dx_world * g_cos + dy_world * g_sin
        ly = -dx_world * g_sin + dy_world * g_cos
        
        # The agent's half-width is 0.05. The pocket opens from lx=-0.04 and ends at lx=0.09.
        # If the center lx is >= 0.02, the agent is securely docked inside.
        if 0.02 <= lx <= 0.09 and -0.03 <= ly <= 0.03:
            return True
            
        return False
        
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Updates the target mocap pose, steps the physics, and calculates reward.
        
        Args:
            action (np.ndarray): Target pose delta/absolute [X, Y, Theta] in [-1, 1].

        Returns:
            Tuple: observation, reward, terminated, truncated, info.
        """
        self._elapsed_steps += 1
        
        # Ensure action values are constrained
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Actions are already absolute physical coordinates
        target_x, target_y, target_theta = action
        
        # Apply the goal pose to the weld-constrained mocap body
        self.data.mocap_pos[self.mocap_id] = [target_x, target_y, 0.025]
        
        # Build 3D quaternion rotation from target theta (yaw around Z-axis)
        cos_half = np.cos(target_theta / 2.0)
        sin_half = np.sin(target_theta / 2.0)
        self.data.mocap_quat[self.mocap_id] = [cos_half, 0, 0, sin_half]
        
        # Advance physics simulation
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
            
        # Draw passive human viewer frame if initialized
        if self.render_mode == "human":
            self.render()
            
        # Compile observation
        obs = self._get_obs()
        x, y, theta = obs["observation.state"]
        
        # Verify success
        success = self._check_success(x, y, theta)
        
        # Reward function
        # +1.0 for success, otherwise 0.0
        if success:
            reward = 1.0
            terminated = True
        else:
            reward = 0.0
            terminated = False
            
        truncated = self._elapsed_steps >= self.max_episode_steps
        
        info = {
            "success": success,
            "elapsed_steps": self._elapsed_steps,
            "distance_to_goal": np.linalg.norm(np.array([x, y]) - np.array(self.current_goal_pos))
        }
        
        return obs, reward, terminated, truncated, info
        
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Resets the agent's physical and controller states."""
        super().reset(seed=seed)
        self._elapsed_steps = 0
        
        # Reset physics registers
        mujoco.mj_resetData(self.model, self.data)
        
        # Spawn agent near the initial position with yaw within +-10 degrees
        noise_x = self.np_random.uniform(-0.05, 0.05)
        noise_y = self.np_random.uniform(-0.05, 0.05)
        noise_theta = self.np_random.uniform(-10.0 * np.pi / 180.0, 10.0 * np.pi / 180.0)
        
        self.data.qpos[0] = noise_x
        self.data.qpos[1] = noise_y
        self.data.qpos[2] = noise_theta
        
        # Clear residual linear/angular velocities
        self.data.qvel[:] = 0.0
        
        # Reset mocap body to match the randomized initial agent coordinate in world frame
        actual_start_x = self.start_pos[0] + noise_x
        actual_start_y = self.start_pos[1] + noise_y
        
        self.data.mocap_pos[self.mocap_id] = [actual_start_x, actual_start_y, 0.025]
        
        cos_half = np.cos(noise_theta / 2.0)
        sin_half = np.sin(noise_theta / 2.0)
        self.data.mocap_quat[self.mocap_id] = [cos_half, 0, 0, sin_half]
        
        # Randomize goal position (wider Y) and fix orientation (0.0) for clean OOD testing
        goal_noise_x = self.np_random.uniform(-0.05, 0.05)
        goal_noise_y = self.np_random.uniform(-0.15, 0.15)
        goal_noise_theta = 0.0
        
        gx = self.goal_pos[0] + goal_noise_x
        gy = self.goal_pos[1] + goal_noise_y
        self.current_goal_pos = (gx, gy)
        self.current_goal_theta = goal_noise_theta
        
        goal_body = self.model.body("goal_pocket")
        goal_body.pos[0] = gx
        goal_body.pos[1] = gy
        goal_body.quat[:] = [np.cos(goal_noise_theta / 2.0), 0, 0, np.sin(goal_noise_theta / 2.0)]
        
        # Compute first physics forward pass
        mujoco.mj_forward(self.model, self.data)
        
        # Synchronize viewer
        if self.render_mode == "human" and self.viewer is not None:
            self.viewer.sync()
            
        obs = self._get_obs()
        info = {
            "success": False,
            "elapsed_steps": 0,
            "distance_to_goal": np.linalg.norm(np.array([actual_start_x, actual_start_y]) - np.array(self.current_goal_pos))
        }
        
        return obs, info
        
    def render(self) -> Optional[np.ndarray]:
        """Provides visual rendering of the environment state."""
        if self.render_mode == "rgb_array":
            # Higher resolution render for evaluation
            high_res_renderer = mujoco.Renderer(self.model, height=512, width=512)
            high_res_renderer.update_scene(self.data, camera="top")
            return high_res_renderer.render()
        elif self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                
                # Set default camera view to 'top' camera
                top_cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "top")
                if top_cam_id >= 0:
                    self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    self.viewer.cam.fixedcamid = top_cam_id
            self.viewer.sync()
            return None
            
    def close(self):
        """Cleans up renderers and viewers."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
