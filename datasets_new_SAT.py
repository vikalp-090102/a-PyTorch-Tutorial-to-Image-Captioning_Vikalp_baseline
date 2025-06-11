import os
import glob
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torchvision import transforms

class IndianaXrayDataset(Dataset):
    def __init__(self, image_dir, projections_csv, reports_csv, split, word_map, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.split = split.upper()
        self.word_map = word_map  # dictionary: word -> index

        # Load and merge metadata
        self.projections = pd.read_csv(projections_csv)
        self.reports = pd.read_csv(reports_csv)
        self.data = self.projections.merge(self.reports, on='uid')

        # Map UID to file
        all_files = glob.glob(os.path.join(self.image_dir, '**', '*.*'), recursive=True)
        all_files_map = {os.path.basename(f): f for f in all_files}
        self.uid_to_file = {}
        for uid in self.data['uid'].astype(str):
            for fname, fpath in all_files_map.items():
                if uid in fname:
                    self.uid_to_file[uid] = fpath
                    break
        self.data = self.data[self.data['uid'].astype(str).isin(self.uid_to_file)].reset_index(drop=True)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        uid = str(row['uid'])
        img_path = self.uid_to_file[uid]

        # Load and transform image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Convert caption (findings) to word indices
        caption = row['findings']
        tokens = caption.lower().split()  # Basic tokenization
        encoded = [self.word_map.get('<start>')] + \
                  [self.word_map.get(w, self.word_map.get('<unk>')) for w in tokens] + \
                  [self.word_map.get('<end>')]

        cap_tensor = torch.tensor(encoded, dtype=torch.long)
        cap_len = torch.tensor(len(encoded), dtype=torch.long)

        return image, cap_tensor, cap_len

    def __len__(self):
        return len(self.data)
