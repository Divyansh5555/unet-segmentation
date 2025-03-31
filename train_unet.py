import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from ast import literal_eval
import argparse

from torch import nn
from torch.utils.data import Dataset, DataLoader
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

# ------------------ U-Net Model ---------------------
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        self.enc1 = nn.Sequential(CBR(3, 64), CBR(64, 64))
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(CBR(64, 128), CBR(128, 128))
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(CBR(128, 256), CBR(256, 256))

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(CBR(256, 128), CBR(128, 128))

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(CBR(128, 64), CBR(64, 64))

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))

        dec2 = self.dec2(torch.cat([self.up2(bottleneck), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.up1(dec2), enc1], dim=1))

        return torch.sigmoid(self.final(dec1))

# ------------- Dataset & Mask Loader ----------------
class PolypDataset(Dataset):
    def __init__(self, image_dir, masks_dict, transform=None):
        self.image_dir = image_dir
        self.filenames = list(masks_dict.keys())
        self.masks_dict = masks_dict
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = cv2.imread(os.path.join(self.image_dir, filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = self.masks_dict[filename]

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0).float()

        return image, mask

def load_via_annotations(csv_path, image_dir):
    df = pd.read_csv(csv_path)
    masks = {}

    for _, row in df.iterrows():
        filename = row['filename']
        shape_attr = row['region_shape_attributes']

        if pd.isna(shape_attr):
            continue

        shape_attr = literal_eval(shape_attr)

        if filename not in masks:
            img_path = os.path.join(image_dir, filename)
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            masks[filename] = np.zeros((h, w), dtype=np.uint8)

        if shape_attr['name'] == 'polygon':
            x_points = shape_attr['all_points_x']
            y_points = shape_attr['all_points_y']
            points = np.array(list(zip(x_points, y_points)), dtype=np.int32)
            cv2.fillPoly(masks[filename], [points], 1)

    return masks

# ---------------- Training Function -----------------
def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0

    for images, masks in tqdm(dataloader):
        images, masks = images.to(device), masks.to(device)

        preds = model(images)
        loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# ----------------------- Main ------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    # Training config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 4
    image_size = 256
    learning_rate = 1e-4

    # Augmentation
    transforms = Compose([
        Resize(image_size, image_size),
        Normalize(),
        ToTensorV2()
    ])

    # Load data
    train_masks = load_via_annotations("train_annotations.csv", "train_images")
    train_dataset = PolypDataset("train_images", train_masks, transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCELoss()

    # Training loop
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "unet_trained.pth")
    print("Model saved to unet_trained.pth")

if __name__ == "__main__":
    main()