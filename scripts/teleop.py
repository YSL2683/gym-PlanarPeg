import os
import sys
import time
import datetime
import argparse
import numpy as np
import gymnasium as gym
import pygame

# Add project root to path to ensure package resolution works cleanly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import planar_peg

def main():
    parser = argparse.ArgumentParser(description="Teleoperate Planar Peg")
    parser.add_argument("--task", type=str, default="ID_base", help="Task name for data organization")
    args = parser.parse_args()
    
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
        print("\n[INFO] No joystick detected. Using keyboard controls (Global Coordinates):")
        print("  - W / S : Translate Up / Down (Global Y)")
        print("  - A / D : Translate Left / Right (Global X)")
        print("  - Q / E : Rotate CCW / CW (Theta)")
        print("  - R     : Reset current episode manually")
        print("  - ESC   : Exit program")

    # 2. Instantiate the Gymnasium environment in human rendering mode
    # This automatically spins up the MuJoCo passive viewer window.
    print("\n[INFO] Loading Gymnasium environment 'PlanarPegInsertion-v0'...")
    env = gym.make("PlanarPegInsertion-v0", render_mode="human")
    
    print(f"[INFO] This is PLAY mode. Data will NOT be saved. Use 'python scripts/record.py' to collect datasets.")

    # Control hyper-parameters
    move_speed = 0.05
    rot_speed = 0.15
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
    agent_init_x, agent_init_y, agent_init_theta = obs["proprioception"]
    raw_action = np.array([agent_init_x, agent_init_y, agent_init_theta], dtype=np.float32)
    filtered_action = np.copy(raw_action)
    
    def reset_episode():
        nonlocal obs, info, raw_action, filtered_action
        obs, info = env.reset()
        agent_init_x, agent_init_y, agent_init_theta = obs["proprioception"]
        raw_action = np.array([agent_init_x, agent_init_y, agent_init_theta], dtype=np.float32)
        filtered_action = np.copy(raw_action)
        
        ep_images_top.clear()
        ep_images_front.clear()
        ep_proprioception.clear()
        ep_actions.clear()
        ep_rewards.clear()
        ep_terminated.clear()
        print("[INFO] Episode reset successfully.")

    def save_episode(success: bool = False):
        print(f"\n🎉 [SUCCESS] Goal Reached! (Note: Play mode does not save data. Run record.py for saving.)")
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
                    
        if not running:
            break
            
        # Get current physical pose of the agent
        agent_x, agent_y, agent_theta = obs["proprioception"]
        # Global steering inputs
        dx_global, dy_global, dtheta = 0.0, 0.0, 0.0
        
        # 3. Read Controller Input
        # 3.1. Joystick Axis control (Takes precedence if available)
        if joystick is not None:
            # Read Left Stick (Translation)
            joy_left_x = joystick.get_axis(0)
            joy_left_y = -joystick.get_axis(1)
            
            # Read Right Stick X (Rotation)
            joy_right_x = joystick.get_axis(2) if joystick.get_numaxes() > 2 else 0.0
            
            # Deadzone
            if abs(joy_left_x) < 0.1: joy_left_x = 0.0
            if abs(joy_left_y) < 0.1: joy_left_y = 0.0
            if abs(joy_right_x) < 0.1: joy_right_x = 0.0
            
            # Left Stick maps directly to Global X/Y
            dx_global = joy_left_x * move_speed
            dy_global = joy_left_y * move_speed
            
            # Right Stick or Bumpers for Rotation
            dtheta = -joy_right_x * rot_speed
            if joystick.get_button(4): dtheta += rot_speed
            if joystick.get_button(5): dtheta -= rot_speed
                
        # 3.2. Keyboard control
        else:
            # W/S controls Global Y (Up/Down on screen)
            if keys[pygame.K_w]: dy_global = move_speed      # W: Up
            elif keys[pygame.K_s]: dy_global = -move_speed   # S: Down
                
            # A/D controls Global X (Left/Right on screen)
            if keys[pygame.K_a]: dx_global = -move_speed     # A: Left
            elif keys[pygame.K_d]: dx_global = move_speed    # D: Right
                
            # Q/E controls Theta (left/right rotation)
            if keys[pygame.K_q]: dtheta = rot_speed        # Q: Rotate CCW
            elif keys[pygame.K_e]: dtheta = -rot_speed     # E: Rotate CW
        
        # Accumulate input commands in target raw action buffer
        raw_action[0] = np.clip(raw_action[0] + dx_global, -env.unwrapped.x_limit, env.unwrapped.x_limit)
        raw_action[1] = np.clip(raw_action[1] + dy_global, -env.unwrapped.y_limit, env.unwrapped.y_limit)
        raw_action[2] = np.clip(raw_action[2] + dtheta, -np.pi, np.pi)
        
        # Limit distance between target mocap pose and actual agent physical position.
        # This resolves the control lag ("self-dragging" effect) while safely preventing structural wall penetration.
        target_x = raw_action[0]
        target_y = raw_action[1]
        target_theta = raw_action[2]
        
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
        raw_action[0] = np.clip(target_x, -env.unwrapped.x_limit, env.unwrapped.x_limit)
        raw_action[1] = np.clip(target_y, -env.unwrapped.y_limit, env.unwrapped.y_limit)
        raw_action[2] = np.clip(target_theta, -np.pi, np.pi)
        
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
            # We ignore truncation in play mode to allow infinite free play
            pass
            
    env.close()
    pygame.quit()
    print("[INFO] Teleoperation closed clean.")

if __name__ == "__main__":
    main()
