import os
import sys
import time
import datetime
import numpy as np
import gymnasium as gym
import h5py
import pygame

# Add project root to path to ensure package resolution works cleanly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import planar_peg

def main():
    # 1. Initialize Pygame (for keyboard and joystick event reading)
    pygame.init()
    pygame.joystick.init()
    
    # Create a small window to capture input events
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("Planar Peg Teleoperation Panel")
    
    # Detect joystick controller
    joystick = None
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"\n[INFO] Joystick detected: {joystick.get_name()}")
    else:
        print("\n[INFO] No joystick detected. Using keyboard controls:")
        print("  - W / S : Control Y-axis (Up / Down in top view)")
        print("  - A / D : Control X-axis (Left / Right in top view)")
        print("  - Q / E : Control Theta / Yaw (Counter-Clockwise / Clockwise)")
        print("  - SPACE : Manually save current episode demonstration")
        print("  - R     : Reset current episode without saving")
        print("  - ESC   : Exit program")

    # 2. Instantiate the Gymnasium environment in human rendering mode
    # This automatically spins up the MuJoCo passive viewer window.
    print("\n[INFO] Loading Gymnasium environment 'PlanarPegInsertion-v0'...")
    env = gym.make("PlanarPegInsertion-v0", render_mode="human")
    
    # Save folder for demonstrations
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(save_dir, exist_ok=True)
    print(f"[INFO] HDF5 demonstrations will save to: {save_dir}")

    # Control hyper-parameters
    move_speed = 0.04
    rot_speed = 0.05
    ema_alpha = 0.15  # Exponential Moving Average smoothing factor
    
    # Frequency regulator (20Hz matches render_fps of the env)
    clock = pygame.time.Clock()
    
    # Buffers to store current episode data
    ep_images_top = []
    ep_images_front = []
    ep_proprioception = []
    ep_actions = []
    ep_rewards = []
    ep_terminated = []
    
    obs, info = env.reset()
    
    # Initialize target mocap action
    # Start target synchronized at the agent start position mapped back to [-1, 1]
    # Initialize target mocap action based on actual randomized spawn coordinates from reset
    raw_action = np.zeros(3, dtype=np.float32)
    agent_init_x, agent_init_y, agent_init_theta = obs["proprioception"]
    raw_action[0] = agent_init_x / env.unwrapped.x_limit
    raw_action[1] = agent_init_y / env.unwrapped.y_limit
    raw_action[2] = agent_init_theta / np.pi
    
    filtered_action = np.copy(raw_action)
    
    def reset_episode():
        nonlocal obs, info, raw_action, filtered_action
        obs, info = env.reset()
        agent_init_x, agent_init_y, agent_init_theta = obs["proprioception"]
        raw_action[0] = agent_init_x / env.unwrapped.x_limit
        raw_action[1] = agent_init_y / env.unwrapped.y_limit
        raw_action[2] = agent_init_theta / np.pi
        filtered_action = np.copy(raw_action)
        
        ep_images_top.clear()
        ep_images_front.clear()
        ep_proprioception.clear()
        ep_actions.clear()
        ep_rewards.clear()
        ep_terminated.clear()
        print("[INFO] Episode reset successfully.")

    def save_episode(success: bool = False):
        if len(ep_actions) == 0:
            print("[WARNING] Buffer empty, skipping save.")
            return
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        status = "success" if success else "failed"
        filename = f"demo_{timestamp}_{status}.hdf5"
        filepath = os.path.join(save_dir, filename)
        
        # Save buffers in HDF5 format with image compression
        with h5py.File(filepath, 'w') as f:
            f.create_dataset("image_top", data=np.array(ep_images_top, dtype=np.uint8), compression="gzip", chunks=True)
            f.create_dataset("image_front", data=np.array(ep_images_front, dtype=np.uint8), compression="gzip", chunks=True)
            f.create_dataset("proprioception", data=np.array(ep_proprioception, dtype=np.float32))
            f.create_dataset("action", data=np.array(ep_actions, dtype=np.float32))
            f.create_dataset("reward", data=np.array(ep_rewards, dtype=np.float32))
            f.create_dataset("terminated", data=np.array(ep_terminated, dtype=bool))
            
            # Save metadata attributes
            f.attrs["success"] = success
            f.attrs["steps"] = len(ep_actions)
            f.attrs["grid"] = env.unwrapped.grid
            
        print(f"[SUCCESS] Saved demonstration: {filepath} (Steps: {len(ep_actions)}, Success: {success})")
        reset_episode()

    running = True
    print("\n[INFO] Starting Teleoperation control. Focus the Pygame window to steer.")
    
    last_focus_warn_time = 0.0
    
    while running:
        clock.tick(20)  # Sleep to enforce 20Hz control frequency
        
        # Check window focus to prevent user confusion with MuJoCo viewer shortcuts
        if not pygame.key.get_focused() and joystick is None:
            current_time = time.time()
            if current_time - last_focus_warn_time > 2.0:
                print("\n⚠️  [WARNING] Pygame Window NOT Focused! Click on 'Planar Peg Teleoperation Panel' window to enable keyboard inputs.", flush=True)
                last_focus_warn_time = current_time
        
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        
        # Terminate Teleop on ESC or Window Close
        if keys[pygame.K_ESCAPE]:
            running = False
            break
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    reset_episode()
                elif event.key == pygame.K_SPACE:
                    save_episode(success=info.get("success", False))
                    
        if not running:
            break
            
        # Get current physical pose of the agent
        agent_x, agent_y, agent_theta = obs["proprioception"]
             # Local steering inputs relative to agent frame (+X is front, +Y is left)
        dx_local, dy_local, dtheta = 0.0, 0.0, 0.0
        
        # 3. Read Controller Input
        # 3.1. Joystick Axis control (Takes precedence if available)
        if joystick is not None:
            # Read horizontal/vertical analog axes
            joy_dx = joystick.get_axis(0)
            joy_dy = -joystick.get_axis(1)  # Invert Y-axis standard conventions
            
            # Filter analog stick micro-drifts (Deadzone)
            if abs(joy_dx) < 0.1: joy_dx = 0.0
            if abs(joy_dy) < 0.1: joy_dy = 0.0
            
            # Joystick: Y-axis controls forward/back (local X), X-axis controls yaw rotation (dtheta)
            dx_local = joy_dy * move_speed
            dtheta = -joy_dx * rot_speed
            
            # Map L1/R1 bumpers (4 and 5) to auxiliary rotation
            if joystick.get_button(4):
                dtheta += rot_speed
            elif joystick.get_button(5):
                dtheta -= rot_speed
                
        # 3.2. Keyboard Keyboard control
        else:
            # W/S controls local X (forward/backward)
            if keys[pygame.K_w]:
                dx_local = move_speed
            elif keys[pygame.K_s]:
                dx_local = -move_speed
                
            # A/D controls Theta (left/right rotation)
            if keys[pygame.K_a]:
                dtheta = rot_speed     # A: rotate counter-clockwise (left)
            elif keys[pygame.K_d]:
                dtheta = -rot_speed    # D: rotate clockwise (right)
                
        # Rotate local command [dx_local, dy_local] to global coordinates using agent's current heading
        cos_theta = np.cos(agent_theta)
        sin_theta = np.sin(agent_theta)
        dx_global = dx_local * cos_theta - dy_local * sin_theta
        dy_global = dx_local * sin_theta + dy_local * cos_theta
        
        # Accumulate input commands in target raw action buffer
        raw_action[0] = np.clip(raw_action[0] + dx_global, -1.0, 1.0)
        raw_action[1] = np.clip(raw_action[1] + dy_global, -1.0, 1.0)
        raw_action[2] = np.clip(raw_action[2] + dtheta, -1.0, 1.0)
        
        # Limit distance between target mocap pose and actual agent physical position.
        # This resolves the control lag ("self-dragging" effect) while safely preventing structural wall penetration.
        target_x = raw_action[0] * env.unwrapped.x_limit
        target_y = raw_action[1] * env.unwrapped.y_limit
        target_theta = raw_action[2] * np.pi
        
        diff_x = target_x - agent_x
        diff_y = target_y - agent_y
        dist = np.sqrt(diff_x**2 + diff_y**2)
        
        # 1. Spatial distance clamping (Max 0.08 meters error)
        max_dist = 0.08
        if dist > max_dist:
            target_x = agent_x + max_dist * (diff_x / dist)
            target_y = agent_y + max_dist * (diff_y / dist)
            
        # 2. Rotational difference clamping (Max 30 degrees error)
        max_diff_theta = 30.0 * np.pi / 180.0
        diff_theta = (target_theta - agent_theta + np.pi) % (2.0 * np.pi) - np.pi
        diff_theta = np.clip(diff_theta, -max_diff_theta, max_diff_theta)
        target_theta = agent_theta + diff_theta
        
        # Write back the safe clamped target coordinates into the raw action register
        raw_action[0] = np.clip(target_x / env.unwrapped.x_limit, -1.0, 1.0)
        raw_action[1] = np.clip(target_y / env.unwrapped.y_limit, -1.0, 1.0)
        raw_action[2] = np.clip(target_theta / np.pi, -1.0, 1.0)
        
        # Smooth commands using Exponential Moving Average (EMA) to prevent structural jerks
        filtered_action = ema_alpha * raw_action + (1.0 - ema_alpha) * filtered_action
        
        # 4. Advance Gym Step
        step_action = np.copy(filtered_action)
        next_obs, reward, terminated, truncated, step_info = env.step(step_action)
        
        # Buffer historical trajectories
        ep_images_top.append(obs["image_top"])
        ep_images_front.append(obs["image_front"])
        ep_proprioception.append(obs["proprioception"])
        ep_actions.append(step_action)
        ep_rewards.append(reward)
        ep_terminated.append(terminated)
        
        obs = next_obs
        info = step_info
        
        # Automatic success recording
        if terminated:
            print("\n[SUCCESS] Insertion complete! Goal pocket docked.")
            save_episode(success=True)
        elif truncated:
            print("\n[INFO] Episode limit reached.")
            reset_episode()
            
    env.close()
    pygame.quit()
    print("[INFO] Teleoperation closed clean.")

if __name__ == "__main__":
    main()
