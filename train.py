import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from models import NvidiaSteeringCNN, TrafficSignCNN
from dataset import SyntheticSteeringDataset, SyntheticTrafficSignDataset

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def train_steering_model(
    epochs: int = 6,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    save_path: str = "models_checkpoint/steering_model.pth"
):
    print("=" * 60)
    print("Training NVIDIA End-to-End Steering Angle CNN (Behavioral Cloning)")
    print("=" * 60)
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Dataset & Loaders
    full_dataset = SyntheticSteeringDataset(num_samples=1200, augment=True)
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    model = NvidiaSteeringCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            
        train_loss = running_loss / len(train_set)
        
        # Validation
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                preds = model(images)
                loss = criterion(preds, targets)
                val_running_loss += loss.item() * images.size(0)
                
        val_loss = val_running_loss / len(val_set)
        scheduler.step(val_loss)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        dt = time.time() - t0
        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({dt:.1f}s) - Train MSE: {train_loss:.5f} | Val MSE: {val_loss:.5f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            
    print(f"--> Best Steering Model saved to: {save_path} (Best Val MSE: {best_val_loss:.5f})\n")
    return history

def train_traffic_sign_model(
    epochs: int = 6,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    save_path: str = "models_checkpoint/traffic_sign_model.pth"
):
    print("=" * 60)
    print("Training Traffic Sign Recognition CNN")
    print("=" * 60)
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Dataset & Loaders
    full_dataset = SyntheticTrafficSignDataset(samples_per_class=200, augment=True)
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    model = TrafficSignCNN(num_classes=6).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {"train_loss": [], "val_acc": []}
    best_val_acc = 0.0
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            
        train_loss = running_loss / len(train_set)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                logits = model(images)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
                
        val_acc = (correct / total) * 100.0
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        
        dt = time.time() - t0
        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({dt:.1f}s) - Train Loss: {train_loss:.5f} | Val Accuracy: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            
    print(f"--> Best Traffic Sign Model saved to: {save_path} (Best Val Acc: {best_val_acc:.2f}%)\n")
    return history

def plot_training_results(steering_hist, traffic_hist, output_img="training_curves.png"):
    plt.figure(figsize=(12, 5))
    
    # Steering Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(steering_hist["train_loss"], label="Train MSE Loss", color="royalblue", lw=2)
    plt.plot(steering_hist["val_loss"], label="Val MSE Loss", color="crimson", lw=2, linestyle="--")
    plt.title("NVIDIA Steering CNN Training Curve", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Traffic Sign Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(traffic_hist["val_acc"], label="Val Accuracy (%)", color="forestgreen", lw=2)
    plt.title("Traffic Sign CNN Validation Accuracy", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_img, dpi=150)
    plt.close()
    print(f"Training curves saved to: {output_img}")

if __name__ == "__main__":
    t_start = time.time()
    s_hist = train_steering_model(epochs=5, batch_size=32)
    t_hist = train_traffic_sign_model(epochs=5, batch_size=32)
    plot_training_results(s_hist, t_hist)
    print(f"Total training pipeline completed in {time.time() - t_start:.1f} seconds.")
