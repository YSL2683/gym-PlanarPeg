import argparse
import os
import h5py
import cv2
import numpy as np

def draw_cv2_plot(proprio, actions, current_step, width, height):
    """Draws a line graph of proprioception and actions using purely OpenCV."""
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 30  # Dark gray background
    
    num_steps = len(proprio)
    if num_steps <= 1:
        return canvas
        
    # Normalize proprioception to [-1, 1] using environment physical limits 
    # to match the action space scale and prevent vertical squishing
    norm_proprio = np.copy(proprio)
    norm_proprio[:, 0] /= 1.2      # X limit
    norm_proprio[:, 1] /= 0.7      # Y limit
    norm_proprio[:, 2] /= np.pi    # Theta limit
    
    # Set plot limits tightly around the [-1, 1] normalized range
    y_min, y_max = -1.2, 1.2
    pad_x, pad_y = 30, 30
    
    def map_coords(step, val):
        x = int((step / (num_steps - 1)) * (width - 2 * pad_x)) + pad_x
        # Invert y so positive is up
        y = int(height - ((val - y_min) / (y_max - y_min)) * (height - 2 * pad_y)) - pad_y
        return (x, y)
        
    # Draw zero axis
    z1 = map_coords(0, 0.0)
    z2 = map_coords(num_steps - 1, 0.0)
    cv2.line(canvas, z1, z2, (100, 100, 100), 1)
    
    # BGR Colors: X=Red, Y=Green, Theta=Blue
    colors_solid = [(0, 0, 255), (0, 200, 0), (255, 50, 50)]
    # Brighter/Lighter colors for the dashed action lines
    colors_dash = [(150, 150, 255), (150, 255, 150), (255, 150, 150)]
    
    for dim in range(3):
        # 1. Draw Action (Dashed/Dotted)
        # To make it dashed, we draw line segments only on even chunks
        for i in range(num_steps - 1):
            if (i // 2) % 2 == 0:  # Creates a dashed pattern by skipping every 2 frames
                pt1 = map_coords(i, actions[i, dim])
                pt2 = map_coords(i + 1, actions[i + 1, dim])
                cv2.line(canvas, pt1, pt2, colors_dash[dim], 1, cv2.LINE_AA)
                
        # 2. Draw Proprioception (Solid) - using normalized values
        for i in range(num_steps - 1):
            pt1 = map_coords(i, norm_proprio[i, dim])
            pt2 = map_coords(i + 1, norm_proprio[i + 1, dim])
            cv2.line(canvas, pt1, pt2, colors_solid[dim], 1, cv2.LINE_AA)
            
    # Draw vertical indicator for current frame
    cx, _ = map_coords(current_step, 0)
    cv2.line(canvas, (cx, pad_y // 2), (cx, height - pad_y // 2), (255, 255, 255), 1)
    cv2.circle(canvas, (cx, height - pad_y // 2), 3, (255, 255, 255), -1)
    
    # Add Legend
    cv2.putText(canvas, "Red: X | Green: Y | Blue: Theta", (pad_x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Solid: Agent State | Dashed(Light): Target Action", (pad_x, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return canvas

def main():
    parser = argparse.ArgumentParser(description="View saved HDF5 demonstrations (Pure OpenCV).")
    parser.add_argument("--file", type=str, required=True, help="Path to the .hdf5 demo file")
    parser.add_argument("--fps", type=int, default=10, help="Playback FPS (default: 10)")
    parser.add_argument("--scale", type=int, default=2, help="Image upscale factor (default: 2)")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"[Error] File {args.file} not found.")
        return

    with h5py.File(args.file, 'r') as f:
        print(f"[INFO] Loaded HDF5: {args.file}")
        
        # Verify required keys exist (Handle both new and old format gracefully)
        has_obs_group = "obs" in f
        
        if has_obs_group:
            img_top = f["obs/image_top"][:]
            img_front = f["obs/image_front"][:]
            proprio = f["obs/proprioception"][:]
        else:
            img_top = f["image_top"][:]
            img_front = f["image_front"][:]
            proprio = f["proprioception"][:]
            
        actions = f["action"][:]
        rewards = f["reward"][:]
        terminated = f["terminated"][:]
        
        num_steps = img_top.shape[0]
        print(f"[INFO] Total steps recorded: {num_steps}")
        print("\n--- Playback Controls ---")
        print(" [SPACE] : Play / Pause")
        print(" [ A ]   : Step Backward (when paused)")
        print(" [ D ]   : Step Forward (when paused)")
        print(" [Q/ESC] : Quit")
        print("-------------------------\n")
        
        delay = int(1000 / args.fps)
        paused = False
        i = 0
        
        while i < num_steps:
            # 1. Fetch images for current step
            top = img_top[i]
            front = img_front[i]
            
            # 2. Convert RGB to BGR for OpenCV
            top_bgr = cv2.cvtColor(top, cv2.COLOR_RGB2BGR)
            front_bgr = cv2.cvtColor(front, cv2.COLOR_RGB2BGR)
            
            # 3. Upscale images dynamically based on original resolution
            s = args.scale
            h, w, _ = top_bgr.shape
            top_bgr = cv2.resize(top_bgr, (w * s, h * s), interpolation=cv2.INTER_NEAREST)
            front_bgr = cv2.resize(front_bgr, (w * s, h * s), interpolation=cv2.INTER_NEAREST)
            
            # Combine side by side (Left: Top, Right: Front)
            img_combined = np.hstack((top_bgr, front_bgr))
            
            # 4. Generate Plot using custom OpenCV drawing
            plot_height = 400
            plot_width = img_combined.shape[1]
            plot_img = draw_cv2_plot(proprio, actions, i, plot_width, plot_height)
            
            # Stack Images (Top: Cameras, Bottom: Plot)
            display_img = np.vstack((img_combined, plot_img))
            
            # 5. Generate Info Text Panel
            info_text = f"Step: {i:03d} / {num_steps-1:03d} | Reward: {rewards[i]:.1f} | Terminated: {bool(terminated[i])}"
            text_bar = np.zeros((40, display_img.shape[1], 3), dtype=np.uint8)
            cv2.putText(text_bar, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Add Pause indicator
            if paused:
                cv2.putText(text_bar, "PAUSED", (display_img.shape[1] - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                
            display_img = np.vstack((display_img, text_bar))
            
            # 6. Display
            cv2.imshow("PlanarPeg HDF5 Viewer (Pure OpenCV)", display_img)
            
            # 7. Handle keyboard inputs
            key = cv2.waitKey(0 if paused else delay) & 0xFF
            
            if key == 27 or key == ord('q'):  # ESC or Q to quit
                break
            elif key == ord(' '):             # Space toggles pause
                paused = not paused
            elif key == ord('a') and paused:  # Step backward
                i = max(0, i - 1)
            elif key == ord('d') and paused:  # Step forward
                i = min(num_steps - 1, i + 1)
            elif not paused:                  # Normal playback
                i += 1
                
        cv2.destroyAllWindows()
        print("[INFO] Viewer closed.")

if __name__ == "__main__":
    main()
