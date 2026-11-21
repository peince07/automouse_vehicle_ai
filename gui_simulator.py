import os
import math
import time
import tkinter as tk
from tkinter import ttk
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageTk, ImageDraw

from models import NvidiaSteeringCNN, TrafficSignCNN
from road_generator import render_road_camera_frame, generate_traffic_sign, CAM_WIDTH, CAM_HEIGHT
from vehicle_controller import AutonomousVehicleController

class AutonomousVehicleSimulatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Autonomous Vehicle AI Simulator - NVIDIA End-to-End & Traffic Sign Recognition")
        self.root.geometry("1100x720")
        self.root.configure(bg="#1e1e24")

        # Load models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.steering_model = NvidiaSteeringCNN().to(self.device)
        self.sign_model = TrafficSignCNN().to(self.device)
        self.load_model_weights()
        
        self.norm_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Vehicle physics & controller
        self.controller = AutonomousVehicleController(initial_speed=35.0)
        
        # Track definition (Parametric closed loop)
        # s in [0, 1] parameterizing the track
        self.track_s = 0.0
        self.track_length_m = 800.0
        self.car_lateral_offset = 0.0  # in [-1.0, 1.0] relative to lane center
        
        # Signs on track at fractions of track: (s_pos, sign_id, name)
        self.track_signs = [
            (0.12, 3, "Speed Limit 60"),
            (0.32, 5, "Sharp Turn"),
            (0.55, 1, "STOP"),
            (0.78, 2, "Speed Limit 30"),
        ]

        self.is_running = True
        self.autonomous_mode = True

        self.setup_ui()
        self.update_simulation_step()

    def load_model_weights(self):
        s_path = os.path.join(os.path.dirname(__file__), "models_checkpoint", "steering_model.pth")
        t_path = os.path.join(os.path.dirname(__file__), "models_checkpoint", "traffic_sign_model.pth")
        
        if os.path.exists(s_path):
            self.steering_model.load_state_dict(torch.load(s_path, map_location=self.device, weights_only=True))
            print(f"Loaded steering model from {s_path}")
        else:
            print("Note: Steering weights not found, using initialized weights.")
            
        if os.path.exists(t_path):
            self.sign_model.load_state_dict(torch.load(t_path, map_location=self.device, weights_only=True))
            print(f"Loaded traffic sign model from {t_path}")
        else:
            print("Note: Traffic sign weights not found, using initialized weights.")
            
        self.steering_model.eval()
        self.sign_model.eval()

    def get_track_geometry(self, s: float):
        """Returns curvature and top-down coordinates (x, y) for track point s in [0, 1]."""
        s = s % 1.0
        angle = s * 2 * math.pi
        
        # Oval with chicane wave
        r = 160 + 40 * math.sin(3 * angle) + 20 * math.cos(2 * angle)
        cx, cy = 250, 260
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        
        # Local curvature approximation: derivative of tangent
        curv = 0.65 * math.sin(3 * angle) + 0.3 * math.cos(2 * angle)
        return x, y, curv

    def setup_ui(self):
        # Header title
        header = tk.Frame(self.root, bg="#121217", height=50)
        header.pack(fill=tk.X, side=tk.TOP)
        title_lbl = tk.Label(
            header,
            text="Autonomous Vehicle Deep Learning Dashboard",
            font=("Segoe UI", 16, "bold"),
            fg="#00e5ff",
            bg="#121217"
        )
        title_lbl.pack(pady=10)

        # Main layout frame
        main_pane = tk.Frame(self.root, bg="#1e1e24")
        main_pane.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        # Left Column: Overhead Track Canvas
        left_col = tk.Frame(main_pane, bg="#272730", bd=2, relief=tk.GROOVE)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        tk.Label(left_col, text="Top-Down Track View & Real-Time Telemetry", font=("Segoe UI", 11, "bold"), fg="#ffffff", bg="#272730").pack(pady=5)
        self.track_canvas = tk.Canvas(left_col, width=500, height=520, bg="#18181f", highlightthickness=0)
        self.track_canvas.pack(padx=10, pady=5)

        # Right Column: AI Camera View, Perception HUD, and Controls
        right_col = tk.Frame(main_pane, bg="#272730", bd=2, relief=tk.GROOVE, width=520)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Front Camera Feed
        cam_hdr = tk.Label(right_col, text="Driver On-Board Camera (AI Input Feed)", font=("Segoe UI", 11, "bold"), fg="#ffffff", bg="#272730")
        cam_hdr.pack(pady=4)
        
        self.cam_label = tk.Label(right_col, bg="#000000", bd=2, relief=tk.SUNKEN)
        self.cam_label.pack(pady=4)

        # Perception HUD
        hud_frame = tk.Frame(right_col, bg="#18181f", bd=1, relief=tk.RIDGE)
        hud_frame.pack(fill=tk.X, padx=15, pady=8)

        # Gauge Canvas (Steering Angle Wheel & Speedometer)
        self.hud_canvas = tk.Canvas(hud_frame, width=480, height=140, bg="#18181f", highlightthickness=0)
        self.hud_canvas.pack(pady=5)

        # Telemetry Labels
        info_frame = tk.Frame(right_col, bg="#272730")
        info_frame.pack(fill=tk.X, padx=20, pady=4)
        
        self.lbl_speed = tk.Label(info_frame, text="Speed: 0.0 km/h", font=("Consolas", 11, "bold"), fg="#00ffcc", bg="#272730")
        self.lbl_speed.grid(row=0, column=0, sticky="w", padx=10, pady=2)
        
        self.lbl_steer = tk.Label(info_frame, text="Steering: 0.0°", font=("Consolas", 11, "bold"), fg="#ffcc00", bg="#272730")
        self.lbl_steer.grid(row=0, column=1, sticky="w", padx=10, pady=2)
        
        self.lbl_sign = tk.Label(info_frame, text="Sign: None", font=("Consolas", 11, "bold"), fg="#ff4466", bg="#272730")
        self.lbl_sign.grid(row=1, column=0, sticky="w", padx=10, pady=2)
        
        self.lbl_state = tk.Label(info_frame, text="State: CRUISING", font=("Consolas", 11, "bold"), fg="#44ff88", bg="#272730")
        self.lbl_state.grid(row=1, column=1, sticky="w", padx=10, pady=2)

        # Controls
        ctrl_frame = tk.Frame(right_col, bg="#1e1e24", pady=6)
        ctrl_frame.pack(fill=tk.X, padx=15, pady=6)

        self.btn_auto = tk.Button(
            ctrl_frame,
            text="Autopilot: ON",
            font=("Segoe UI", 10, "bold"),
            bg="#00b0ff",
            fg="#ffffff",
            activebackground="#0091ea",
            command=self.toggle_autopilot,
            width=14
        )
        self.btn_auto.pack(side=tk.LEFT, padx=8)

        self.btn_pause = tk.Button(
            ctrl_frame,
            text="Pause",
            font=("Segoe UI", 10, "bold"),
            bg="#546e7a",
            fg="#ffffff",
            command=self.toggle_pause,
            width=10
        )
        self.btn_pause.pack(side=tk.LEFT, padx=8)

        self.btn_reset = tk.Button(
            ctrl_frame,
            text="Reset",
            font=("Segoe UI", 10, "bold"),
            bg="#455a64",
            fg="#ffffff",
            command=self.reset_sim,
            width=10
        )
        self.btn_reset.pack(side=tk.LEFT, padx=8)

    def toggle_autopilot(self):
        self.autonomous_mode = not self.autonomous_mode
        if self.autonomous_mode:
            self.btn_auto.config(text="Autopilot: ON", bg="#00b0ff")
        else:
            self.btn_auto.config(text="Autopilot: OFF", bg="#78909c")

    def toggle_pause(self):
        self.is_running = not self.is_running
        self.btn_pause.config(text="Resume" if not self.is_running else "Pause")

    def reset_sim(self):
        self.track_s = 0.0
        self.car_lateral_offset = 0.0
        self.controller = AutonomousVehicleController(initial_speed=35.0)

    def check_nearby_traffic_sign(self, current_s: float):
        """Detects if a traffic sign is in front of the vehicle along the track."""
        for s_pos, sign_id, name in self.track_signs:
            dist = (s_pos - current_s) % 1.0
            if 0.005 < dist < 0.075:
                norm_dist = dist / 0.075
                return sign_id, name, 1.0 - norm_dist
        return 0, "No Sign", 1.0

    def run_ai_inference(self, road_pil_img, sign_id):
        # 1. Steering Prediction (NVIDIA CNN)
        input_tensor = self.norm_transform(road_pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred_steering = self.steering_model(input_tensor).item()

        # 2. Traffic Sign Recognition (Sign CNN)
        sign_patch = generate_traffic_sign(sign_id, size=64, augment=False)
        sign_tensor = self.norm_transform(sign_patch).unsqueeze(0).to(self.device)
        with torch.no_grad():
            sign_logits = self.sign_model(sign_tensor)
            probs = torch.softmax(sign_logits, dim=1)
            pred_class_id = torch.argmax(probs, dim=1).item()
            conf = probs[0, pred_class_id].item()

        pred_sign_name = TrafficSignCNN.SIGN_CLASSES[pred_class_id]
        return pred_steering, pred_sign_name, conf

    def update_simulation_step(self):
        if self.is_running:
            dt = 0.05
            # Get track state
            _, _, curv = self.get_track_geometry(self.track_s)
            sign_id, actual_sign_name, sign_closeness = self.check_nearby_traffic_sign(self.track_s)

            # Render camera frame from driver perspective
            road_frame = render_road_camera_frame(
                curvature=float(curv),
                car_offset=float(self.car_lateral_offset),
                width=CAM_WIDTH,
                height=CAM_HEIGHT,
                sign_id=sign_id,
                sign_distance=float(sign_closeness)
            )

            # AI Inference
            pred_steer, det_sign, conf = self.run_ai_inference(road_frame, sign_id)

            # Controller update
            if self.autonomous_mode:
                self.controller.update_telemetry(pred_steer, det_sign, conf, dt=dt)
                
                # Update vehicle physics position along track
                speed_mps = (self.controller.speed * 1000.0) / 3600.0
                ds = (speed_mps * dt) / self.track_length_m
                self.track_s = (self.track_s + ds) % 1.0
                
                # Lateral movement: steering corrects lateral offset
                # Positive steering turns right, increasing offset
                self.car_lateral_offset += (self.controller.steering_angle - curv) * 0.08 * (self.controller.speed / 50.0)
                self.car_lateral_offset = max(-0.85, min(0.85, self.car_lateral_offset * 0.96))

            # Display Camera Frame in GUI
            disp_cam = road_frame.resize((400, 132), Image.Resampling.BILINEAR)
            self.tk_cam_img = ImageTk.PhotoImage(disp_cam)
            self.cam_label.config(image=self.tk_cam_img)

            # Update HUD and Top-Down Track
            self.draw_track_view()
            self.draw_hud_gauges(pred_steer, det_sign, conf)

            # Update Labels
            st = self.controller.get_status_dict()
            self.lbl_speed.config(text=f"Speed: {st['speed_kmh']} km/h (Target: {st['target_speed_kmh']})")
            self.lbl_steer.config(text=f"Steering: {st['steering_deg']}°")
            self.lbl_sign.config(text=f"Sign: {st['detected_sign']} ({st['sign_confidence']})")
            self.lbl_state.config(text=f"State: {st['state']}")

        # Loop at ~20-30 FPS
        self.root.after(45, self.update_simulation_step)

    def draw_track_view(self):
        self.track_canvas.delete("all")
        
        # Draw closed track boundary
        track_pts = []
        for i in range(120):
            s = i / 120.0
            x, y, _ = self.get_track_geometry(s)
            track_pts.append((x, y))
            
        # Draw asphalt circuit
        for i in range(len(track_pts)):
            x0, y0 = track_pts[i]
            x1, y1 = track_pts[(i + 1) % len(track_pts)]
            self.track_canvas.create_line(x0, y0, x1, y1, fill="#373a3e", width=42, capstyle=tk.ROUND)

        # Draw center dashed line
        for i in range(0, len(track_pts), 2):
            x0, y0 = track_pts[i]
            x1, y1 = track_pts[(i + 1) % len(track_pts)]
            self.track_canvas.create_line(x0, y0, x1, y1, fill="#ffffff", width=2, dash=(6, 6))

        # Draw roadside traffic signs on track
        for s_pos, s_id, s_name in self.track_signs:
            sx, sy, _ = self.get_track_geometry(s_pos)
            color_map = {1: "#e53935", 2: "#fb8c00", 3: "#43a047", 5: "#fdd835"}
            self.track_canvas.create_oval(sx + 24, sy - 12, sx + 44, sy + 8, fill=color_map.get(s_id, "#fff"), outline="#ffffff", width=2)
            self.track_canvas.create_text(sx + 34, sy - 2, text=s_name.split()[0], fill="#000000", font=("Segoe UI", 7, "bold"))

        # Draw Autonomous Car
        car_x, car_y, _ = self.get_track_geometry(self.track_s)
        next_x, next_y, _ = self.get_track_geometry(self.track_s + 0.01)
        heading = math.atan2(next_y - car_y, next_x - car_x)
        
        # Apply lateral offset
        norm_x = -math.sin(heading) * (self.car_lateral_offset * 12.0)
        norm_y = math.cos(heading) * (self.car_lateral_offset * 12.0)
        car_x += norm_x
        car_y += norm_y

        # Car chassis rectangle oriented along heading
        c_len, c_w = 20, 10
        corners = [
            (car_x + c_len * math.cos(heading) - c_w * math.sin(heading), car_y + c_len * math.sin(heading) + c_w * math.cos(heading)),
            (car_x + c_len * math.cos(heading) + c_w * math.sin(heading), car_y + c_len * math.sin(heading) - c_w * math.cos(heading)),
            (car_x - c_len * math.cos(heading) + c_w * math.sin(heading), car_y - c_len * math.sin(heading) - c_w * math.cos(heading)),
            (car_x - c_len * math.cos(heading) - c_w * math.sin(heading), car_y - c_len * math.sin(heading) + c_w * math.cos(heading)),
        ]
        flat_corners = [coord for pt in corners for coord in pt]
        self.track_canvas.create_polygon(flat_corners, fill="#00e5ff", outline="#ffffff", width=2)
        
        # Headlights / camera beam
        beam_x = car_x + 35 * math.cos(heading)
        beam_y = car_y + 35 * math.sin(heading)
        self.track_canvas.create_line(car_x, car_y, beam_x, beam_y, fill="#ffff00", width=2, arrow=tk.LAST)

    def draw_hud_gauges(self, steering_angle, sign_name, confidence):
        self.hud_canvas.delete("all")
        
        # 1. Virtual Steering Wheel Gauge
        cx, cy, r = 110, 70, 48
        self.hud_canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#546e7a", width=5)
        # Steering angle in radians
        steer_rad = steering_angle * math.radians(70)
        sx = cx + (r - 6) * math.sin(steer_rad)
        sy = cy - (r - 6) * math.cos(steer_rad)
        self.hud_canvas.create_line(cx, cy, sx, sy, fill="#ffcc00", width=4, capstyle=tk.ROUND)
        self.hud_canvas.create_oval(cx - 8, cy - 8, cx + 8, cy + 8, fill="#ffffff")
        self.hud_canvas.create_text(cx, cy + r + 15, text=f"Steer: {steering_angle * 28.0:+.1f}°", fill="#ffcc00", font=("Consolas", 10, "bold"))

        # 2. Speedometer Arc
        sp_cx, sp_cy, sp_r = 250, 70, 48
        self.hud_canvas.create_arc(sp_cx - sp_r, sp_cy - sp_r, sp_cx + sp_r, sp_cy + sp_r, start=-30, extent=240, style=tk.ARC, outline="#37474f", width=6)
        # Speed ratio (0 to 80 km/h)
        speed_ratio = min(1.0, max(0.0, self.controller.speed / 80.0))
        speed_extent = speed_ratio * 240
        self.hud_canvas.create_arc(sp_cx - sp_r, sp_cy - sp_r, sp_cx + sp_r, sp_cy + sp_r, start=210, extent=-speed_extent, style=tk.ARC, outline="#00e5ff", width=6)
        self.hud_canvas.create_text(sp_cx, sp_cy, text=f"{int(self.controller.speed)}", fill="#ffffff", font=("Segoe UI", 16, "bold"))
        self.hud_canvas.create_text(sp_cx, sp_cy + 18, text="KM/H", fill="#90a4ae", font=("Segoe UI", 8))
        self.hud_canvas.create_text(sp_cx, sp_cy + sp_r + 15, text=f"Target: {int(self.controller.target_speed)}", fill="#00e5ff", font=("Consolas", 10))

        # 3. Traffic Sign Recognition Card
        bx, by, bw, bh = 345, 25, 120, 90
        self.hud_canvas.create_rectangle(bx, by, bx + bw, by + bh, fill="#21212a", outline="#37474f", width=2)
        self.hud_canvas.create_text(bx + bw // 2, by + 18, text="DETECTED SIGN", fill="#78909c", font=("Segoe UI", 8, "bold"))
        sign_color = "#ff5252" if "STOP" in sign_name else ("#ffd740" if "Turn" in sign_name else "#69f0ae")
        self.hud_canvas.create_text(bx + bw // 2, by + 45, text=sign_name, fill=sign_color, font=("Segoe UI", 10, "bold"))
        self.hud_canvas.create_text(bx + bw // 2, by + 70, text=f"Conf: {confidence * 100:.1f}%", fill="#b0bec5", font=("Consolas", 9))

def launch_gui():
    root = tk.Tk()
    app = AutonomousVehicleSimulatorApp(root)
    root.mainloop()

if __name__ == "__main__":
    launch_gui()
