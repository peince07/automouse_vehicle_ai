import math
import numpy as np

class AutonomousVehicleController:
    """
    Vehicle dynamics, state machine, and autonomous control system.
    Integrates deep learning steering predictions with traffic sign reaction rules.
    """
    def __init__(self, initial_speed: float = 30.0):
        # Physics state
        self.speed = initial_speed  # km/h
        self.target_speed = 50.0    # default cruise target (km/h)
        self.steering_angle = 0.0   # normalized [-1.0, 1.0]
        self.max_steering_angle_deg = 28.0
        
        # State machine
        self.state = "CRUISING"  # CRUISING, BRAKING, STOPPED, ACCELERATING
        self.active_sign = "No Sign"
        self.sign_confidence = 0.0
        self.stop_timer = 0.0
        self.stop_duration = 2.0  # seconds to stay stopped at a STOP sign
        
        # Performance counters
        self.distance_traveled = 0.0
        self.odometer = 0.0

    def update_telemetry(self, predicted_steering: float, sign_class: str, sign_confidence: float, dt: float = 0.1):
        """
        Updates vehicle state using deep learning outputs.
        dt: time step in seconds.
        """
        # Smooth steering actuation (first-order low-pass filter to prevent jerky motion)
        alpha = 0.45
        self.steering_angle = (1 - alpha) * self.steering_angle + alpha * predicted_steering
        self.active_sign = sign_class
        self.sign_confidence = sign_confidence

        # Traffic sign reaction state machine
        if sign_class == "STOP" and sign_confidence > 0.65:
            if self.state != "STOPPED" and self.state != "RESUMING":
                self.state = "BRAKING"
                self.target_speed = 0.0
        elif sign_class == "Speed Limit 30" and sign_confidence > 0.60:
            self.target_speed = 30.0
            if self.state not in ["STOPPED", "BRAKING"]:
                self.state = "CRUISING"
        elif sign_class == "Speed Limit 60" and sign_confidence > 0.60:
            self.target_speed = 60.0
            if self.state not in ["STOPPED", "BRAKING"]:
                self.state = "CRUISING"
        elif sign_class == "Sharp Turn" and sign_confidence > 0.60:
            self.target_speed = 25.0  # Slow down for safe cornering

        # Longitudinal dynamics (Speed PID / Throttle & Brake control)
        if self.state == "BRAKING":
            # Decelerate firmly
            decel = 28.0 * dt  # km/h per second
            self.speed = max(0.0, self.speed - decel)
            if self.speed <= 0.5:
                self.speed = 0.0
                self.state = "STOPPED"
                self.stop_timer = 0.0

        elif self.state == "STOPPED":
            self.speed = 0.0
            self.stop_timer += dt
            if self.stop_timer >= self.stop_duration:
                self.state = "RESUMING"
                self.target_speed = 40.0  # resume normal cruising speed

        elif self.state in ["CRUISING", "RESUMING"]:
            if self.speed < self.target_speed:
                # Smooth acceleration
                accel = 12.0 * dt
                self.speed = min(self.target_speed, self.speed + accel)
            elif self.speed > self.target_speed:
                # Coast down / moderate braking
                decel = 16.0 * dt
                self.speed = max(self.target_speed, self.speed - decel)

        # Distance calculation: speed (km/h) * (1000/3600) * dt (m)
        v_mps = (self.speed * 1000.0) / 3600.0
        self.distance_traveled += v_mps * dt

    def get_status_dict(self):
        return {
            "speed_kmh": round(self.speed, 1),
            "target_speed_kmh": round(self.target_speed, 1),
            "steering_deg": round(self.steering_angle * self.max_steering_angle_deg, 1),
            "state": self.state,
            "detected_sign": self.active_sign,
            "sign_confidence": f"{self.sign_confidence * 100:.1f}%",
            "distance_m": round(self.distance_traveled, 1)
        }
