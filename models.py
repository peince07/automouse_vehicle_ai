import torch
import torch.nn as nn
import torch.nn.functional as F

class NvidiaSteeringCNN(nn.Module):
    """
    NVIDIA End-to-End Deep Learning Architecture for Self-Driving Cars
    Reference: Bojarski et al., "End to End Learning for Self-Driving Cars", NVIDIA 2016.
    
    Predicts normalized steering angle in [-1.0, 1.0] from a front-facing road camera image.
    Input size: (batch_size, 3, 66, 200) RGB image.
    """
    def __init__(self, dropout_rate: float = 0.3):
        super(NvidiaSteeringCNN, self).__init__()
        
        # Convolutional Feature Extractor
        # 3 5x5 convolutions with 2x2 stride
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        
        # 2 3x3 non-strided convolutions
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Regularization
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Fully Connected Layers (Dense controller)
        # Input shape after conv5 on (66, 200):
        # 66 -> 31 -> 14 -> 5 -> 3 -> 1
        # 200 -> 98 -> 47 -> 22 -> 20 -> 18
        # Shape: 64 * 1 * 18 = 1152
        self.fc1 = nn.Linear(64 * 1 * 18, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc_out = nn.Linear(10, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Exponential Linear Unit (ELU) activations provide smooth gradient flow
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = F.elu(self.conv5(x))
        
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1)
        
        x = F.elu(self.fc1(x))
        x = self.dropout(x)
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        steering = torch.tanh(self.fc_out(x))  # Bound output to [-1.0, 1.0]
        return steering.squeeze(-1)


class TrafficSignCNN(nn.Module):
    """
    Convolutional Neural Network for Autonomous Vehicle Traffic Sign Classification.
    Input size: (batch_size, 3, 64, 64)
    Classes:
      0: Normal Road / No Sign
      1: STOP Sign
      2: SPEED LIMIT 30
      3: SPEED LIMIT 60
      4: YIELD Sign
      5: SHARP TURN Sign
    """
    SIGN_CLASSES = [
        "No Sign",
        "STOP",
        "Speed Limit 30",
        "Speed Limit 60",
        "Yield",
        "Sharp Turn"
    ]

    def __init__(self, num_classes: int = 6, dropout_rate: float = 0.4):
        super(TrafficSignCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x64 -> 32x32
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        logits = self.classifier(feat)
        return logits
