import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import os

class IndianaXrayDataset(Dataset):
    def __init__(self, image_dir, projections_csv, reports_csv, split, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.split = split.upper()
        
        # Load data
        self.projections = pd.read_csv(projections_csv)
        self.reports = pd.read_csv(reports_csv)

        # Merge data on UID
        self.data = self.projections.merge(self.reports, on='uid')
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_filename = f"{row['uid']}.jpg"  # adjust extension if needed
        img_path = os.path.join(self.image_dir, img_filename)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        caption = row['findings']  # Extracting caption from reports
        return image, caption

    def __len__(self):
        return len(self.data)
