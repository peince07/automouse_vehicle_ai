import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from road_generator import (
    render_road_camera_frame,
    generate_traffic_sign,
    CAM_WIDTH,
    CAM_HEIGHT,
    SIGN_SIZE
)

class SyntheticSteeringDataset(Dataset):
    """
    Synthesizes diverse road camera images and corresponding target steering angles
    for behavioral cloning / end-to-end self-driving car training.
    """
    def __init__(self, num_samples: int = 1500, augment: bool = True):
        self.num_samples = num_samples
        self.augment = augment
        
        # Standard PyTorch normalization for vision models
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        # Generate random curvature and car offset
        # Distribution: mostly smooth curves with occasional sharp turns and lane recovery
        curvature = np.clip(np.random.normal(0.0, 0.42), -1.0, 1.0)
        car_offset = np.clip(np.random.normal(0.0, 0.35), -1.0, 1.0)
        
        # Occasional roadside sign
        sign_id = random.choice([0, 0, 0, 1, 2, 3, 4, 5])
        sign_dist = random.uniform(0.3, 0.9)
        
        brightness = random.uniform(0.75, 1.25) if self.augment else 1.0
        add_shadow = (random.random() > 0.6) if self.augment else False
        
        img = render_road_camera_frame(
            curvature=float(curvature),
            car_offset=float(car_offset),
            sign_id=sign_id,
            sign_distance=sign_dist,
            brightness=brightness,
            add_shadow=add_shadow
        )
        
        # Ground truth steering calculation:
        # Steer to follow curvature + counter-steer to recover lane center
        # If car is shifted right (car_offset > 0), steer left (< 0)
        target_steering = float(np.clip(0.72 * curvature - 0.68 * car_offset, -1.0, 1.0))

        # Data augmentation: Horizontal flip
        if self.augment and random.random() > 0.5:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            target_steering = -target_steering

        tensor_img = self.to_tensor(img)
        return tensor_img, torch.tensor(target_steering, dtype=torch.float32)


class SyntheticTrafficSignDataset(Dataset):
    """
    Synthesizes traffic sign training images across 6 classes:
    0: No Sign (Background)
    1: STOP
    2: Speed Limit 30
    3: Speed Limit 60
    4: Yield
    5: Sharp Turn
    """
    def __init__(self, samples_per_class: int = 250, augment: bool = True):
        self.num_classes = 6
        self.samples_per_class = samples_per_class
        self.total_samples = samples_per_class * self.num_classes
        self.augment = augment

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int):
        class_id = idx % self.num_classes
        img = generate_traffic_sign(class_id, size=SIGN_SIZE, augment=self.augment)
        tensor_img = self.transform(img)
        return tensor_img, torch.tensor(class_id, dtype=torch.long)
