import os
import sys
import argparse
import numpy as np
import gymnasium as gym
import zarr
import pygame

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import planar_peg

def main():
    parser = argparse.ArgumentParser(description="Record Planar Peg Demonstrations to Zarr")
    parser.add_argument("--task", type=str, default="ID_base", help="Task name for Zarr dataset")
    parser.add_argument("--dataset_name", type=str, default="dataset", help="Name of the Zarr dataset file")
    parser.add_argument("--num_episodes", type=int, default=50, help="Target number of successful episodes")
    parser.add_argument("--resume", action="store_true", help="Resume and append to existing Zarr dataset")
    args = parser.parse_args()
    
    # Setup Zarr Dataset path
    base_save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", args.task)
    os.makedirs(base_save_dir, exist_ok=True)
    zarr_path = os.path.join(base_save_dir, f"{args.dataset_name}.zarr")
    
    if os.path.exists(zarr_path):
        if args.resume:
            root = zarr.open(zarr_path, mode='a')
            
            # Safety check: Verify we are appending to the exact right dataset
            stored_task = root.attrs.get('task', None)
            stored_dataset = root.attrs.get('dataset_name', None)
            
            if stored_task is not None and stored_dataset is not None:
                if stored_task != args.task or stored_dataset != args.dataset_name:
                    print(f"\n[ERROR] Dataset Mismatch!")
                    print(f"Existing dataset: task='{stored_task}', dataset_name='{stored_dataset}'")
                    print(f"Requested arguments: task='{args.task}', dataset_name='{args.dataset_name}'")
                    print("Aborting to prevent data corruption.")
                    sys.exit(1)
            else:
                # Legacy dataset without metadata. Backfill it now.
                root.attrs['task'] = args.task
                root.attrs['dataset_name'] = args.dataset_name
                
            current_episodes = len(root['meta/episode_ends'])
            print(f"[INFO] --resume flag detected. Resuming existing dataset with {current_episodes} episodes.")
        else:
            print(f"\n[ERROR] Dataset {zarr_path} already exists!")
            sys.exit(1)
    else:
        if args.resume:
            print(f"\\n[ERROR] Cannot resume. Dataset {zarr_path} does not exist!")
            sys.exit(1)
            
        root = zarr.open(zarr_path, mode='w')
        # Inscribe metadata to prevent future corruption
        root.attrs['task'] = args.task
        root.attrs['dataset_name'] = args.dataset_name
        
        data = root.create_group('data')
        data.create_dataset('image_top', shape=(0, 224, 224, 3), chunks=(100, 224, 224, 3), dtype='uint8')
        data.create_dataset('image_front', shape=(0, 224, 224, 3), chunks=(100, 224, 224, 3), dtype='uint8')
        data.create_dataset('proprioception', shape=(0, 3), chunks=(100, 3), dtype='float32')
        data.create_dataset('action', shape=(0, 3), chunks=(100, 3), dtype='float32')
        data.create_dataset('reward', shape=(0,), chunks=(100,), dtype='float32')
        data.create_dataset('terminated', shape=(0,), chunks=(100,), dtype='bool')
        meta = root.create_group('meta')
        meta.create_dataset('episode_ends', shape=(0,), chunks=(100,), dtype='int64')
        current_episodes = 0

    if current_episodes >= args.num_episodes:
        print(f"[INFO] Target of {args.num_episodes} episodes already reached. Exiting.")
        sys.exit(0)
        
    # Pygame & Env Initialization
    pygame.init()
    pygame.joystick.init()
    screen = pygame.display.set_mode((400, 300))
    
    def update_title():
        pygame.display.set_caption(f"Recording: {args.task} | Episodes: {current_episodes}/{args.num_episodes}")
    update_title()
    
    joystick = None
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"[INFO] Joystick detected: {joystick.get_name()}")
        
    print("\n[INFO] Loading Gymnasium environment...")
    env = gym.make("PlanarPegInsertion-v0", render_mode="human")
    
    move_speed = 0.05
    rot_speed = 0.15
    ema_alpha = 0.15
    clock = pygame.time.Clock()
    
    ep_images_top = []
    ep_images_front = []
    ep_proprioception = []
    ep_actions = []
    ep_rewards = []
    ep_terminated = []
    
    obs, info = env.reset()
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

    def save_episode():
        nonlocal current_episodes
        if len(ep_actions) == 0: return
        
        # Append data directly to Zarr disk arrays
        root['data/image_top'].append(np.array(ep_images_top, dtype=np.uint8))
        root['data/image_front'].append(np.array(ep_images_front, dtype=np.uint8))
        root['data/proprioception'].append(np.array(ep_proprioception, dtype=np.float32))
        root['data/action'].append(np.array(ep_actions, dtype=np.float32))
        root['data/reward'].append(np.array(ep_rewards, dtype=np.float32))
        root['data/terminated'].append(np.array(ep_terminated, dtype=bool))
        
        ep_len = len(ep_actions)
        if len(root['meta/episode_ends']) == 0:
            last_end = 0
        else:
            last_end = root['meta/episode_ends'][-1]
            
        root['meta/episode_ends'].append(np.array([last_end + ep_len], dtype=np.int64))
        
        current_episodes += 1
        print(f"\n🎉 [SUCCESS] Saved episode {current_episodes}/{args.num_episodes} (Length: {ep_len} steps)")
        update_title()
        reset_episode()

    running = True
    print("\n🚀 [RECORD MODE ACTIVE] Steer agent into the goal to automatically save!")
    print("Press 'R' to discard the current trajectory and retry.")
    print("Press 'ESC' to exit safely without losing collected data.\n")
    
    while running and current_episodes < args.num_episodes:
        clock.tick(10)
        
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_ESCAPE]:
            running = False
            break
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    print("\n♻️  [INFO] User discarded episode. Resetting...")
                    reset_episode()
                    
        if not running:
            break
            
        agent_x, agent_y, agent_theta = obs["proprioception"]
        dx_global, dy_global, dtheta = 0.0, 0.0, 0.0
        
        if joystick is not None:
            joy_left_x = joystick.get_axis(0)
            joy_left_y = -joystick.get_axis(1)
            joy_right_x = joystick.get_axis(2) if joystick.get_numaxes() > 2 else 0.0
            
            if abs(joy_left_x) < 0.1: joy_left_x = 0.0
            if abs(joy_left_y) < 0.1: joy_left_y = 0.0
            if abs(joy_right_x) < 0.1: joy_right_x = 0.0
            
            dx_global = joy_left_x * move_speed
            dy_global = joy_left_y * move_speed
            dtheta = -joy_right_x * rot_speed
            if joystick.get_button(4): dtheta += rot_speed
            if joystick.get_button(5): dtheta -= rot_speed
        else:
            if keys[pygame.K_w]: dy_global = move_speed
            elif keys[pygame.K_s]: dy_global = -move_speed
            if keys[pygame.K_a]: dx_global = -move_speed
            elif keys[pygame.K_d]: dx_global = move_speed
            if keys[pygame.K_q]: dtheta = rot_speed
            elif keys[pygame.K_e]: dtheta = -rot_speed
            
        raw_action[0] = np.clip(raw_action[0] + dx_global, -env.unwrapped.x_limit, env.unwrapped.x_limit)
        raw_action[1] = np.clip(raw_action[1] + dy_global, -env.unwrapped.y_limit, env.unwrapped.y_limit)
        raw_action[2] = np.clip(raw_action[2] + dtheta, -np.pi, np.pi)
        
        target_x = raw_action[0]
        target_y = raw_action[1]
        target_theta = raw_action[2]
        
        diff_x = target_x - agent_x
        diff_y = target_y - agent_y
        dist = np.sqrt(diff_x**2 + diff_y**2)
        
        max_dist = 0.10
        if dist > max_dist:
            target_x = agent_x + max_dist * (diff_x / dist)
            target_y = agent_y + max_dist * (diff_y / dist)
            
        max_diff_theta = 30.0 * np.pi / 180.0
        diff_theta = (target_theta - agent_theta + np.pi) % (2.0 * np.pi) - np.pi
        diff_theta = np.clip(diff_theta, -max_diff_theta, max_diff_theta)
        target_theta = agent_theta + diff_theta
        
        raw_action[0] = np.clip(target_x, -env.unwrapped.x_limit, env.unwrapped.x_limit)
        raw_action[1] = np.clip(target_y, -env.unwrapped.y_limit, env.unwrapped.y_limit)
        raw_action[2] = np.clip(target_theta, -np.pi, np.pi)
        
        filtered_action = ema_alpha * raw_action + (1.0 - ema_alpha) * filtered_action
        step_action = np.copy(filtered_action)
        
        next_obs, reward, terminated, truncated, step_info = env.step(step_action)
        
        ep_images_top.append(obs["image_top"])
        ep_images_front.append(obs["image_front"])
        ep_proprioception.append(obs["proprioception"])
        ep_actions.append(step_action)
        ep_rewards.append(reward)
        ep_terminated.append(terminated)
        
        obs = next_obs
        info = step_info
        
        if terminated:
            save_episode()
        elif truncated:
            print("\n⏳ [INFO] Episode limit reached (Timeout). Discarding trajectory...")
            reset_episode()
            
    env.close()
    pygame.quit()
    
    if current_episodes >= args.num_episodes:
        print(f"\n🎯 [DONE] Target of {args.num_episodes} episodes reached! Dataset safely stored at: {zarr_path}")
    else:
        print(f"\n🛑 [STOPPED] Recording interrupted. Currently at {current_episodes}/{args.num_episodes} episodes.")

if __name__ == "__main__":
    main()
