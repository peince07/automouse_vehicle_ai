import os
import sys
import time
import math
import argparse
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from models import NvidiaSteeringCNN, TrafficSignCNN
from road_generator import render_road_camera_frame, generate_traffic_sign, CAM_WIDTH, CAM_HEIGHT
from vehicle_controller import AutonomousVehicleController
from train import train_steering_model, train_traffic_sign_model, plot_training_results

def ensure_models_trained():
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "models_checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    s_path = os.path.join(checkpoint_dir, "steering_model.pth")
    t_path = os.path.join(checkpoint_dir, "traffic_sign_model.pth")
    
    if not os.path.exists(s_path) or not os.path.exists(t_path):
        print("\n[*] Model checkpoints not found. Initiating fast model training...\n")
        s_hist = train_steering_model(epochs=5, batch_size=32, save_path=s_path)
        t_hist = train_traffic_sign_model(epochs=5, batch_size=32, save_path=t_path)
        plot_training_results(s_hist, t_hist, output_img="training_curves.png")
        print("[✓] Models successfully trained and weights saved!\n")
    else:
        print(f"[✓] Existing model weights detected in {checkpoint_dir}.\n")
        
    return s_path, t_path

def run_headless_simulation(num_steps: int = 120, output_img: str = "autonomous_drive_telemetry.png"):
    print("=" * 70)
    print("      AUTONOMOUS VEHICLE AI - CLOSED LOOP SIMULATION RUNNER        ")
    print("=" * 70)
    
    s_path, t_path = ensure_models_trained()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Models
    steering_model = NvidiaSteeringCNN().to(device)
    steering_model.load_state_dict(torch.load(s_path, map_location=device, weights_only=True))
    steering_model.eval()
    
    sign_model = TrafficSignCNN().to(device)
    sign_model.load_state_dict(torch.load(t_path, map_location=device, weights_only=True))
    sign_model.eval()
    
    norm_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    controller = AutonomousVehicleController(initial_speed=30.0)
    
    # Telemetry logging arrays
    timestamps = []
    speeds = []
    target_speeds = []
    predicted_steers = []
    ideal_steers = []
    car_positions = []
    states = []
    captured_frames = []
    
    # Track scenario definition
    track_s = 0.0
    car_offset = 0.0
    track_length = 750.0  # meters
    dt = 0.1  # 100ms per simulation step
    
    track_signs = [
        (0.15, 3, "Speed Limit 60"),
        (0.35, 5, "Sharp Turn Ahead"),
        (0.60, 1, "STOP Sign"),
        (0.85, 2, "Speed Limit 30"),
    ]
    
    print(f"{'Step':<6} | {'Speed':<10} | {'Steering':<11} | {'Detected Sign':<18} | {'State':<12}")
    print("-" * 70)
    
    for step in range(num_steps):
        t = step * dt
        # Track curvature profile
        angle = track_s * 2 * math.pi
        curv = 0.60 * math.sin(3 * angle) + 0.25 * math.cos(2 * angle)
        
        # Check nearby signs
        sign_id = 0
        sign_dist = 1.0
        for s_pos, s_id, s_name in track_signs:
            diff = (s_pos - track_s) % 1.0
            if 0.005 < diff < 0.065:
                sign_id = s_id
                sign_dist = 1.0 - (diff / 0.065)
                break
                
        # Render synthetic camera frame from driver POV
        frame = render_road_camera_frame(
            curvature=curv,
            car_offset=car_offset,
            sign_id=sign_id,
            sign_distance=sign_dist
        )
        
        # AI Inference: Steering Prediction
        input_t = norm_transform(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_steer = steering_model(input_t).item()
            
        # AI Inference: Traffic Sign Detection
        sign_patch = generate_traffic_sign(sign_id, size=64, augment=False)
        sign_t = norm_transform(sign_patch).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = sign_model(sign_t)
            probs = torch.softmax(logits, dim=1)
            cid = torch.argmax(probs, dim=1).item()
            conf = probs[0, cid].item()
            
        detected_sign = TrafficSignCNN.SIGN_CLASSES[cid]
        
        # Controller update
        controller.update_telemetry(pred_steer, detected_sign, conf, dt=dt)
        
        # Physics update
        v_mps = (controller.speed * 1000.0) / 3600.0
        track_s = (track_s + (v_mps * dt) / track_length) % 1.0
        car_offset += (controller.steering_angle - curv) * 0.06 * (controller.speed / 50.0)
        car_offset = np.clip(car_offset * 0.96, -0.85, 0.85)
        
        ideal_steer = np.clip(0.72 * curv - 0.68 * car_offset, -1.0, 1.0)
        
        # Log data
        timestamps.append(t)
        speeds.append(controller.speed)
        target_speeds.append(controller.target_speed)
        predicted_steers.append(pred_steer * 28.0)
        ideal_steers.append(ideal_steer * 28.0)
        states.append(controller.state)
        
        # Top-down coordinates for visualization
        r = 160 + 35 * math.sin(3 * angle)
        car_positions.append((250 + r * math.cos(angle), 250 + r * math.sin(angle)))
        
        if step % 25 == 0 or (sign_id == 1 and len(captured_frames) < 3):
            captured_frames.append((step, frame.copy(), detected_sign, controller.speed))
            
        if step % 10 == 0 or sign_id > 0:
            st = controller.get_status_dict()
            print(f"{step:<6} | {st['speed_kmh']:>4.1f} km/h | {st['steering_deg']:>+5.1f}°     | {detected_sign:<18} | {st['state']:<12}")
            
    print("-" * 70)
    print(f"[✓] Simulation drive completed successfully across {num_steps} steps ({num_steps * dt:.1f}s simulated time)!")
    print(f"    Total distance covered: {controller.distance_traveled:.1f} meters.\n")
    
    # Generate Multi-Panel Telemetry Figure
    fig = plt.figure(figsize=(15, 9), facecolor="#f8f9fa")
    
    # 1. Top-Down Track & Trajectory
    ax1 = fig.add_subplot(2, 2, 1)
    track_angles = np.linspace(0, 2 * np.pi, 200)
    tx = 250 + (160 + 35 * np.sin(3 * track_angles)) * np.cos(track_angles)
    ty = 250 + (160 + 35 * np.sin(3 * track_angles)) * np.sin(track_angles)
    ax1.plot(tx, ty, color="#495057", lw=12, label="Asphalt Road")
    ax1.plot(tx, ty, color="#f8f9fa", lw=1.5, linestyle="--", label="Center Line")
    
    cxs = [p[0] for p in car_positions]
    cys = [p[1] for p in car_positions]
    ax1.plot(cxs, cys, color="#0077b6", lw=2.5, label="Car Trajectory (Autonomous)")
    ax1.scatter([cxs[-1]], [cys[-1]], color="#d90429", s=100, zorder=5, label="Current Position")
    
    # Mark signs
    for s_pos, s_id, s_name in track_signs:
        sa = s_pos * 2 * np.pi
        sr = 160 + 35 * np.sin(3 * sa)
        sx = 250 + (sr + 25) * np.cos(sa)
        sy = 250 + (sr + 25) * np.sin(sa)
        ax1.scatter([sx], [sy], color="#ffb703", s=80, edgecolors="black", zorder=4)
        ax1.text(sx + 8, sy, s_name, fontsize=8, fontweight="bold")
        
    ax1.set_title("1. Top-Down Autonomous Vehicle Trajectory", fontsize=11, fontweight="bold")
    ax1.axis("off")
    ax1.legend(loc="lower right", fontsize=8)
    
    # 2. Steering Angle Plot
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(timestamps, ideal_steers, label="Ground Truth / Ideal Steering (°)", color="#6c757d", linestyle="--", lw=1.5)
    ax2.plot(timestamps, predicted_steers, label="NVIDIA AI Predicted Steering (°)", color="#0077b6", lw=2)
    ax2.set_title("2. Steering Angle: AI Prediction vs Lane Geometry", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Simulation Time (s)")
    ax2.set_ylabel("Steering Angle (Degrees)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8)
    
    # 3. Longitudinal Velocity & State Machine
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(timestamps, target_speeds, label="Target Cruise Speed (km/h)", color="#e63946", linestyle=":", lw=2)
    ax3.plot(timestamps, speeds, label="Actual Vehicle Speed (km/h)", color="#2a9d8f", lw=2)
    ax3.set_title("3. Longitudinal Control & Speed Adaptation", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Simulation Time (s)")
    ax3.set_ylabel("Speed (km/h)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="lower left", fontsize=8)
    
    # 4. Captured Camera Frames Montage
    ax4 = fig.add_subplot(2, 2, 4)
    if captured_frames:
        step_idx, sample_frame, sign_n, sp = captured_frames[-1]
        ax4.imshow(sample_frame)
        ax4.set_title(f"4. AI Camera Feed (Step {step_idx}: {sign_n} | Speed {sp:.1f} km/h)", fontsize=11, fontweight="bold")
    ax4.axis("off")
    
    plt.tight_layout()
    plt.savefig(output_img, dpi=150)
    plt.close()
    print(f"[✓] Multi-panel telemetry visualization saved to: {output_img}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Vehicle AI System")
    parser.add_argument("--steps", type=int, default=120, help="Number of simulation steps")
    parser.add_argument("--gui", action="store_true", help="Launch interactive Tkinter desktop GUI")
    args = parser.parse_args()
    
    if args.gui:
        from gui_simulator import launch_gui
        print("Starting interactive desktop GUI...")
        launch_gui()
    else:
        run_headless_simulation(num_steps=args.steps)
