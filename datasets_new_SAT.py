import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import os
import glob  # import at top-level

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
        uid = str(row['uid'])

        # Search files starting with uid in image_dir
        files = glob.glob(os.path.join(self.image_dir, f"{uid}*.png"))  # adjust extension if needed

        if len(files) == 0:
            raise FileNotFoundError(f"No image file found for UID {uid}")

        img_path = files[0]  # Take the first match

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        caption = row['findings']
        return image, caption

    def __len__(self):
        return len(self.data)
