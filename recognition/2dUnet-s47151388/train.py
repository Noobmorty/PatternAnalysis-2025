"""
Training script for 2D UNet on HipMRI2d dataset
Author: Isaac Tiang
"""

import os
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from modules import UNet2D, DiceLoss
from dataset import HipMRIdata


def train_one_epoch(model, device, loader, criterion, optimizer):
    model.train()
    running_loss = []
    for images, masks in tqdm(loader, desc="Training", leave=False):
        images, masks = images.to(device), masks.to(device).long()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss.append(loss.item())
    return np.mean(running_loss)


def validate_one_epoch(model, device, loader, criterion):
    model.eval()
    running_loss = []
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validating", leave=False):
            images, masks = images.to(device), masks.to(device).long()
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss.append(loss.item())
    return np.mean(running_loss)


def main():
    # Hyper parameters
    batch_size = 8
    learning_rate = 0.001
    num_epochs = 20

    # Paths
    data_root = r"C:\Users\isaac\keras_slices_data"
    model_dir = r"C:\Users\isaac\keras_slices_data\models"
    outputs_dir = r"C:\Users\isaac\keras_slices_data\outputs"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Datasets and Loaders
    train_dataset = HipMRIdata(data_root, img_set="train", img_size=(256, 256), apply_transform=True)
    val_dataset = HipMRIdata(data_root, img_set="validate", img_size=(256, 256))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model, Loss, Optimizer
    model = UNet2D().to(device)
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_epoch = -1

    print("\nStarting Training!!!\n")
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, device, train_loader, criterion, optimizer)
        val_loss = validate_one_epoch(model, device, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save prediction images every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                for i, (images, masks) in enumerate(val_loader):
                    images, masks = images.to(device), masks.to(device).long()
                    outputs = model(images)

                    for j in range(min(4, images.size(0))):  # save first 4 images
                        img = images[j]
                        mask = masks[j]

                        # Ensure image is (C,H,W)
                        if img.dim() == 2:
                            img = img.unsqueeze(0)

                        # Predicted mask: argmax over channels → (1,H,W)
                        pred_mask = torch.argmax(outputs[j], dim=0, keepdim=True).float() / (outputs.size(1)-1)

                        # Ground truth mask: scale to 0-1 for visualization → (1,H,W)
                        if mask.dim() == 2:
                            gt_mask = mask.unsqueeze(0).float() / (outputs.size(1)-1)
                        elif mask.dim() == 3 and mask.size(0) == 1:
                            gt_mask = mask.float() / (outputs.size(1)-1)
                        else:
                            raise ValueError(f"Unexpected mask shape: {mask.shape}")

                        # Concatenate along width for visualization: (C,H,W*3)
                        save_image(
                            torch.cat([img, pred_mask, gt_mask], dim=2),
                            os.path.join(outputs_dir, f"epoch{epoch+1}_batch{i}_img{j}.png")
                        )

                    break  # only first batch

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_path = os.path.join(model_dir, "unet_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"!! New best model saved at epoch {best_epoch} (Val Loss: {best_val_loss:.4f})")

    # Plot training curves
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Loss")
    plt.legend()
    plt.title("UNet Training Progress")
    plt.savefig(os.path.join(model_dir, "training_curve.png"))
    plt.close()

    #  Save final model
    final_path = os.path.join(model_dir, "unet_latest.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")
    print(f"!! Best model was from epoch {best_epoch} with val loss {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
