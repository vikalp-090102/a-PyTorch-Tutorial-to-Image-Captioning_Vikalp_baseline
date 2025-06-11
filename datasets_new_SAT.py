import os
import glob
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

class IndianaXrayDataset(Dataset):
    def __init__(self, image_dir, projections_csv, reports_csv, split, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.split = split.upper()

        # Load CSVs
        self.projections = pd.read_csv(projections_csv)
        self.reports = pd.read_csv(reports_csv)

        # Merge on UID
        self.data = self.projections.merge(self.reports, on='uid')

        # Find all image files (recursively, in case they are in subfolders)
        all_files = glob.glob(os.path.join(self.image_dir, '**', '*.*'), recursive=True)
        all_files_map = {os.path.basename(f): f for f in all_files}

        # Build UID-to-file map using substring match
        self.uid_to_file = {}
        missing_uids = []

        for uid in self.data['uid'].astype(str):
            matched = False
            for fname, fpath in all_files_map.items():
                if uid in fname:
                    self.uid_to_file[uid] = fpath
                    matched = True
                    break
            if not matched:
                missing_uids.append(uid)

        if missing_uids:
            print(f"[WARNING] {len(missing_uids)} UIDs from CSV did not match any image file. Sample: {missing_uids[:5]}")

        # Filter out rows with missing image files
        available_uids = set(self.uid_to_file.keys())
        self.data = self.data[self.data['uid'].astype(str).isin(available_uids)].reset_index(drop=True)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        uid = str(row['uid'])

        if uid not in self.uid_to_file:
            raise FileNotFoundError(f"No image file found for UID {uid}")

        img_path = self.uid_to_file[uid]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        caption = row['findings']
        return image, caption

    def __len__(self):
        return len(self.data)
