import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from models import NvidiaSteeringCNN, TrafficSignCNN
from dataset import SyntheticSteeringDataset, SyntheticTrafficSignDataset

def evaluate_system():
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "models_checkpoint")
    s_path = os.path.join(checkpoint_dir, "steering_model.pth")
    t_path = os.path.join(checkpoint_dir, "traffic_sign_model.pth")
    
    if not os.path.exists(s_path) or not os.path.exists(t_path):
        print("Model checkpoints not found. Run train.py or run_project.py first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating models on device: {device}")

    # 1. Evaluate Steering CNN
    print("\n" + "=" * 50)
    print("Evaluating NVIDIA Steering CNN Model...")
    print("=" * 50)
    s_model = NvidiaSteeringCNN().to(device)
    s_model.load_state_dict(torch.load(s_path, map_location=device, weights_only=True))
    s_model.eval()

    test_steer_data = SyntheticSteeringDataset(num_samples=300, augment=False)
    steer_loader = DataLoader(test_steer_data, batch_size=32, shuffle=False)

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for imgs, tgts in steer_loader:
            imgs = imgs.to(device)
            preds = s_model(imgs)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(tgts.numpy().tolist())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    mae = np.mean(np.abs(all_preds - all_targets))
    mse = np.mean((all_preds - all_targets) ** 2)
    rmse = np.sqrt(mse)
    corr = np.corrcoef(all_preds, all_targets)[0, 1]

    print(f"Test Samples:     {len(all_preds)}")
    print(f"Mean Abs Error:   {mae:.4f} (approx {mae * 28.0:.2f}°)")
    print(f"Root Mean Sq Err: {rmse:.4f}")
    print(f"Correlation:      {corr:.4f}")

    # 2. Evaluate Traffic Sign CNN
    print("\n" + "=" * 50)
    print("Evaluating Traffic Sign Recognition CNN...")
    print("=" * 50)
    t_model = TrafficSignCNN().to(device)
    t_model.load_state_dict(torch.load(t_path, map_location=device, weights_only=True))
    t_model.eval()

    test_sign_data = SyntheticTrafficSignDataset(samples_per_class=50, augment=False)
    sign_loader = DataLoader(test_sign_data, batch_size=32, shuffle=False)

    all_sign_preds = []
    all_sign_targets = []
    with torch.no_grad():
        for imgs, tgts in sign_loader:
            imgs = imgs.to(device)
            logits = t_model(imgs)
            preds = torch.argmax(logits, dim=1)
            all_sign_preds.extend(preds.cpu().numpy().tolist())
            all_sign_targets.extend(tgts.numpy().tolist())

    acc = np.mean(np.array(all_sign_preds) == np.array(all_sign_targets)) * 100.0
    print(f"Overall Accuracy: {acc:.2f}%\n")
    print("Classification Report:")
    print(classification_report(all_sign_targets, all_sign_preds, target_names=TrafficSignCNN.SIGN_CLASSES, digits=3))

    # Plot evaluation metrics
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot: Target vs Predicted Steering
    axes[0].scatter(all_targets * 28.0, all_preds * 28.0, alpha=0.6, color="#0077b6", edgecolors="none")
    ideal_line = np.linspace(-28, 28, 50)
    axes[0].plot(ideal_line, ideal_line, color="#d90429", linestyle="--", lw=2, label="Ideal 1:1 Response")
    axes[0].set_title(f"Steering Angle Prediction (MAE: {mae*28.0:.2f}°, r={corr:.3f})", fontweight="bold")
    axes[0].set_xlabel("Target Steering Angle (°)")
    axes[0].set_ylabel("Predicted Steering Angle (°)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Confusion matrix
    cm = confusion_matrix(all_sign_targets, all_sign_preds)
    im = axes[1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1].set_title(f"Traffic Sign Confusion Matrix (Acc: {acc:.1f}%)", fontweight="bold")
    fig.colorbar(im, ax=axes[1])
    tick_marks = np.arange(len(TrafficSignCNN.SIGN_CLASSES))
    axes[1].set_xticks(tick_marks)
    axes[1].set_xticklabels([c.replace(" ", "\n") for c in TrafficSignCNN.SIGN_CLASSES], fontsize=8)
    axes[1].set_yticks(tick_marks)
    axes[1].set_yticklabels(TrafficSignCNN.SIGN_CLASSES, fontsize=8)
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[1].text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    output_eval_img = "evaluation_metrics.png"
    plt.savefig(output_eval_img, dpi=150)
    plt.close()
    print(f"\n[✓] Evaluation visualization saved to {output_eval_img}")

if __name__ == "__main__":
    evaluate_system()
