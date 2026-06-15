import argparse
import os
import zarr
import cv2
import numpy as np

def draw_cv2_plot(proprio, actions, current_step, width, height):
    """Draws a line graph of proprioception and actions using purely OpenCV."""
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 30  # Dark gray background
    
    num_steps = len(proprio)
    if num_steps <= 1:
        return canvas
        
    # Keep absolute physical coordinates directly for the plot
    norm_proprio = np.copy(proprio)
    norm_actions = np.copy(actions)
    
    pad_x = 40
    row_height = height // 3
    pad_y = 15
    
    # Dynamic limits based on raw data max/min
    limits = []
    y_ticks = []
    for dim in range(3):
        min_val = min(np.min(norm_proprio[:, dim]), np.min(norm_actions[:, dim]))
        max_val = max(np.max(norm_proprio[:, dim]), np.max(norm_actions[:, dim]))
        
        margin = (max_val - min_val) * 0.1
        if margin == 0: margin = 0.1
        
        y_min = min_val - margin
        y_max = max_val + margin
        limits.append((y_min, y_max))
        y_ticks.append([y_min, (y_min + y_max) / 2.0, y_max])
    
    def map_coords(dim, step, val):
        y_min, y_max = limits[dim]
        local_h = row_height - 2 * pad_y
        
        x = int((step / max(1, num_steps - 1)) * (width - 2 * pad_x)) + pad_x
        # Invert y so positive is up
        y_norm = (val - y_min) / (y_max - y_min)
        y_local = int(local_h - y_norm * local_h) + pad_y
        y = dim * row_height + y_local
        return (x, y)
        
    colors_solid = [(0, 0, 255), (0, 200, 0), (255, 50, 50)]
    colors_dash = [(150, 150, 255), (150, 255, 150), (255, 150, 150)]
    labels = ["X (m)", "Y (m)", "Theta (rad)"]
        
    for dim in range(3):
        # 1. Draw Dividers and Zero axes
        if dim > 0:
            cv2.line(canvas, (0, dim * row_height), (width, dim * row_height), (100, 100, 100), 1)
            
        if limits[dim][0] <= 0.0 <= limits[dim][1]:
            z1 = map_coords(dim, 0, 0.0)
            z2 = map_coords(dim, num_steps - 1, 0.0)
            cv2.line(canvas, z1, z2, (80, 80, 80), 1)
        
        # 2. Draw Y-axis labels and Legend
        cv2.putText(canvas, labels[dim], (width - 80, dim * row_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors_solid[dim], 1, cv2.LINE_AA)
        
        for y_val in y_ticks[dim]:
            y_coord = map_coords(dim, 0, y_val)[1]
            cv2.putText(canvas, f"{y_val:>4.1f}", (2, y_coord + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
            
        # 3. Draw Action Target (Dashed/Dotted)
        for i in range(num_steps - 1):
            if i % 2 == 0:
                pt1 = map_coords(dim, i, norm_actions[i, dim])
                pt2 = map_coords(dim, i + 1, norm_actions[i + 1, dim])
                cv2.line(canvas, pt1, pt2, colors_dash[dim], 1, cv2.LINE_AA)
                
        # 4. Draw Proprioception State (Solid)
        for i in range(num_steps - 1):
            pt1 = map_coords(dim, i, norm_proprio[i, dim])
            pt2 = map_coords(dim, i + 1, norm_proprio[i + 1, dim])
            cv2.line(canvas, pt1, pt2, colors_solid[dim], 1, cv2.LINE_AA)
            
    # Draw X-axis numbers (Start, Middle, End) on the bottom
    for step_val in [0, num_steps//2, num_steps-1]:
        x_coord = map_coords(2, step_val, limits[2][0])[0]
        cv2.putText(canvas, f"{step_val}", (x_coord - 10, height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
            
    # Draw vertical indicator for current frame across all subplots
    cx = map_coords(0, current_step, 0)[0]
    cv2.line(canvas, (cx, 10), (cx, height - 15), (255, 255, 255), 1)
    
    return canvas

def main():
    parser = argparse.ArgumentParser(description="View saved Zarr demonstrations (Pure OpenCV).")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the .zarr dataset directory")
    parser.add_argument("--fps", type=int, default=10, help="Playback FPS (default: 10)")
    parser.add_argument("--scale", type=int, default=2, help="Image upscale factor (default: 2)")
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"[Error] Dataset {args.dataset} not found.")
        return

    print(f"[INFO] Loading Zarr Dataset: {args.dataset}")
    root = zarr.open(args.dataset, mode='r')
    
    img_top_all = root['data/image_top']
    img_front_all = root['data/image_front']
    proprio_all = root['data/proprioception']
    actions_all = root['data/action']
    
    has_reward = 'data/reward' in root
    if has_reward:
        rewards_all = root['data/reward']
        terminated_all = root['data/terminated']
        
    episode_ends = root['meta/episode_ends'][:]
    
    total_episodes = len(episode_ends)
    print(f"[INFO] Total episodes recorded: {total_episodes}")
    if total_episodes == 0:
        print("[INFO] Dataset is empty.")
        return

    print("\\n--- Playback Controls ---")
    print(" [SPACE] : Play / Pause")
    print(" [  W  ] : Next Episode")
    print(" [  S  ] : Previous Episode")
    print(" [  A  ] : Step Backward Frame (when paused)")
    print(" [  D  ] : Step Forward Frame (when paused)")
    print(" [Q/ESC] : Quit")
    print("-------------------------\\n")
    
    delay = int(1000 / args.fps)
    paused = False
    
    ep_idx = 0
    
    while ep_idx < total_episodes:
        start_idx = 0 if ep_idx == 0 else episode_ends[ep_idx - 1]
        end_idx = episode_ends[ep_idx]
        
        # Load current episode data into memory for fast scrubbing
        ep_img_top = img_top_all[start_idx:end_idx]
        ep_img_front = img_front_all[start_idx:end_idx]
        ep_proprio = proprio_all[start_idx:end_idx]
        ep_actions = actions_all[start_idx:end_idx]
        
        if has_reward:
            ep_rewards = rewards_all[start_idx:end_idx]
            ep_terminated = terminated_all[start_idx:end_idx]
            
        ep_length = end_idx - start_idx
        frame_idx = 0
        
        # Loop through frames of the current episode
        while frame_idx < ep_length:
            top = ep_img_top[frame_idx]
            front = ep_img_front[frame_idx]
            
            # Convert RGB to BGR for OpenCV
            top_bgr = cv2.cvtColor(top, cv2.COLOR_RGB2BGR)
            front_bgr = cv2.cvtColor(front, cv2.COLOR_RGB2BGR)
            
            # Upscale
            s = args.scale
            h, w, _ = top_bgr.shape
            top_bgr = cv2.resize(top_bgr, (w * s, h * s), interpolation=cv2.INTER_NEAREST)
            front_bgr = cv2.resize(front_bgr, (w * s, h * s), interpolation=cv2.INTER_NEAREST)
            
            img_combined = np.hstack((top_bgr, front_bgr))
            
            plot_height = 400
            plot_width = img_combined.shape[1]
            plot_img = draw_cv2_plot(ep_proprio, ep_actions, frame_idx, plot_width, plot_height)
            
            display_img = np.vstack((img_combined, plot_img))
            
            if has_reward:
                info_text = f"Ep: {ep_idx+1}/{total_episodes} | Frame: {frame_idx+1}/{ep_length} | Reward: {ep_rewards[frame_idx]:.2f} | Terminated: {bool(ep_terminated[frame_idx])}"
            else:
                info_text = f"Ep: {ep_idx+1}/{total_episodes} | Frame: {frame_idx+1}/{ep_length}"
                
            text_bar = np.zeros((40, display_img.shape[1], 3), dtype=np.uint8)
            cv2.putText(text_bar, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
            if paused:
                cv2.putText(text_bar, "PAUSED", (display_img.shape[1] - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                
            display_img = np.vstack((display_img, text_bar))
            cv2.imshow("Zarr Data Viewer", display_img)
            
            key = cv2.waitKey(0 if paused else delay) & 0xFF
            
            if key == 27 or key == ord('q'):
                cv2.destroyAllWindows()
                return
            elif key == ord(' '):
                paused = not paused
            elif key == ord('w'): # Next Episode
                ep_idx = min(total_episodes - 1, ep_idx + 1)
                break # break frame loop to load next episode
            elif key == ord('s'): # Prev Episode
                ep_idx = max(0, ep_idx - 1)
                break
            elif key == ord('a') and paused:
                frame_idx = max(0, frame_idx - 1)
            elif key == ord('d') and paused:
                frame_idx = min(ep_length - 1, frame_idx + 1)
            elif not paused:
                frame_idx += 1
                if frame_idx >= ep_length:
                    # Auto-advance to next episode when current ends
                    ep_idx += 1
                    break
                    
    cv2.destroyAllWindows()
    print("[INFO] Viewer closed.")

if __name__ == "__main__":
    main()
