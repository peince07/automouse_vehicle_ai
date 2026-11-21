# Autonomous Vehicle AI: Deep Learning Behavioral Cloning & Traffic Sign Recognition

An end-to-end Autonomous Vehicle (Self-Driving Car) system engineered in Python and PyTorch. The project integrates **NVIDIA's End-to-End Convolutional Neural Network (DAVE-2)** for lateral steering angle prediction with a **Convolutional Traffic Sign Classifier** and a **Closed-Loop Vehicle Physics Controller**.

---

## 🏎️ Key Features

1. **NVIDIA End-to-End Steering Angle Prediction (Behavioral Cloning)**:
   - Takes raw front-facing road camera frames (RGB $66 \times 200$).
   - Feature extraction via 5 convolutional layers (strided & non-strided) with ELU activations.
   - Outputs continuous steering angles $\in [-1.0, 1.0]$ mapped to physical steering angles $\pm 28^\circ$.

2. **Multi-Class Traffic Sign Recognition**:
   - Detects and classifies roadside traffic signs in real time:
     - `STOP`: Triggers automatic deceleration and brings vehicle to a complete halt.
     - `Speed Limit 30`: Enforces low-speed city cruising limit.
     - `Speed Limit 60`: Accelerates vehicle to high-speed cruise.
     - `Sharp Turn Ahead`: Proactively decelerates to maintain vehicle stability through high-curvature curves.
     - `Yield`: Moderates speed for right-of-way compliance.
     - `No Sign`: Standard road cruising.

3. **Closed-Loop Vehicle Controller**:
   - **Lateral Control**: First-order filtered steering actuator with lane centering recovery.
   - **Longitudinal Control**: Proportional speed controller managing throttle and regenerative braking.
   - **State Machine**: Coordinates transitions between `CRUISING`, `BRAKING`, `STOPPED`, and `RESUMING`.

4. **Dual Simulation Modes**:
   - **Interactive Desktop GUI (`gui_simulator.py`)**: Real-time dual display showing top-down circuit track, live driver camera feed, steering wheel gauge, speedometer arc, and control toggles.
   - **Automated Telemetry Runner (`run_project.py`)**: Executes closed-loop road tests and exports multi-panel analytics graphs (`autonomous_drive_telemetry.png`).

---

## 📁 Project Structure

```
autonomous_vehicle_ai/
│
├── models.py                 # PyTorch architectures (NvidiaSteeringCNN, TrafficSignCNN)
├── road_generator.py         # Realistic perspective road camera & sign rendering engine
├── dataset.py                # Synthetic dataset generators with data augmentation
├── train.py                  # Model training pipeline with loss & accuracy curves
├── vehicle_controller.py     # Vehicle dynamics, state machine, and throttle/braking logic
├── gui_simulator.py          # Interactive Tkinter desktop graphical simulator
├── run_project.py            # Master script: automated simulation & telemetry visualizer
├── evaluate.py               # Comprehensive model evaluation & confusion matrix
├── models_checkpoint/        # Saved model weights (.pth)
│   ├── steering_model.pth
│   └── traffic_sign_model.pth
└── README.md                 # Complete documentation & hardware guide
```

---

## 🚀 Quick Start Guide

### 1. Run Automated Autonomous Driving Simulation
To run the simulation and generate telemetry plots:
```bash
python run_project.py --steps 100
```
This generates `autonomous_drive_telemetry.png` showing:
- Overhead track trajectory and sign trigger points.
- Steering angle comparison (AI predicted vs ideal lane geometry).
- Speed curve and braking events (e.g. stopping at STOP signs).
- Captured front-camera image frames.

### 2. Launch Interactive Desktop GUI Simulator
To open the live interactive dashboard:
```bash
python gui_simulator.py
# or
python run_project.py --gui
```
- Click **"Autopilot: ON/OFF"** to switch between AI self-driving and manual intervention.
- Observe real-time steering angle changes, speedometer updates, and traffic sign responses.

### 3. Evaluate Model Accuracy
To calculate Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and generate confusion matrices:
```bash
python evaluate.py
```
Outputs `evaluation_metrics.png`.

### 4. Retrain Models
To retrain the steering and traffic sign models:
```bash
python train.py
```
Outputs `training_curves.png`.

---

## 🛠️ Deploying to Physical Hardware (RC Car / Raspberry Pi)

This software architecture is designed for direct transferability to physical hardware:

1. **Hardware Components**:
   - **Onboard Computer**: Raspberry Pi 4 / 5 or NVIDIA Jetson Nano.
   - **Camera**: Raspberry Pi Camera Module V2 or Wide-Angle USB Webcam.
   - **Motor Driver**: L298N Dual H-Bridge Motor Driver.
   - **Chassis**: 2WD or 4WD robotic RC car chassis with DC gear motors.
   - **Power Supply**: 7.4V / 11.1V LiPo battery (for motors) + 5V Power Bank (for Raspberry Pi).

2. **Physical Control Loop**:
   - Replace `road_generator.py` with `cv2.VideoCapture(0)`.
   - Pass captured frames to `NvidiaSteeringCNN` to obtain steering value.
   - Map normalized steering $[-1.0, 1.0]$ to PWM servo signals (`RPi.GPIO` or `pigpio`).
   - Map controller target speed to motor driver PWM pins.
