"""
Performs model validation and computes average Dice coefficient.
Author: Isaac Tiang
"""

import torch
from torch.utils.data import DataLoader
from modules import UNet2D, DiceLoss
from dataset import HipMRIdata


def validate(model_path, data_root, device):
    """
    Validates the trained UNet model on the validation set.
    """
    # Load trained model
    model = UNet2D(in_channels=1, num_classes=6, base_channels=32)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load dataset
    val_dataset = HipMRIdata(data_root, img_set="validate", img_size=(256,256), apply_transform=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Dice loss (1 - Dice coefficient)
    dice = DiceLoss()
    scores = []

    print("Running validation...")
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device).squeeze(1).long()  # squeeze channel dim
            preds = model(images)
            loss_val = dice(preds, masks)
            scores.append(1 - loss_val.item())  # Dice coefficient

    avg_dice = sum(scores) / len(scores)
    print(f"Average Dice coefficient: {avg_dice:.4f}")
    return avg_dice


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Path to your trained model
    model_file = r"C:\Users\isaac\keras_slices_data\models\unet_best.pth"

    # Path to your dataset root
    data_dir = r"C:\Users\isaac\keras_slices_data"

    validate(model_file, data_dir, device)
