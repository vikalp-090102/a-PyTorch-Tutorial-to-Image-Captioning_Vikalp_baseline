import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
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

        # Optional: filter by split if 'split' column exists in data
        if 'split' in self.data.columns:
            self.data = self.data[self.data['split'].str.upper() == self.split].reset_index(drop=True)

        # Map UID to file paths
        all_files = glob.glob(os.path.join(self.image_dir, '**', '*.*'), recursive=True)
        all_files_map = {os.path.basename(f): f for f in all_files}
        self.uid_to_file = {}
        for uid in self.data['uid'].astype(str):
            # Expecting exact filename match: e.g., "12345.png"
            filename = f"{uid}.png"  # update extension if needed
            if filename in all_files_map:
                self.uid_to_file[uid] = all_files_map[filename]
            else:
                # Handle missing files gracefully
                print(f"Warning: Image file for UID {uid} not found. Skipping.")
        
        # Keep only rows for which image exists
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
        tokens = caption.lower().split()  # Basic tokenization; update if needed
        encoded = [self.word_map.get('<start>')] + \
                  [self.word_map.get(w, self.word_map.get('<unk>')) for w in tokens] + \
                  [self.word_map.get('<end>')]

        cap_tensor = torch.tensor(encoded, dtype=torch.long)
        cap_len = torch.tensor(len(encoded), dtype=torch.long)

        return image, cap_tensor, cap_len

    def __len__(self):
        return len(self.data)


def collate_fn(data):
    """
    Custom collate function to pad variable length captions
    """
    images, captions, lengths = zip(*data)

    images = torch.stack(images, 0)
    lengths = torch.tensor(lengths, dtype=torch.long)

    max_len = max(lengths)
    padded_captions = torch.zeros(len(captions), max_len, dtype=torch.long)
    for i, cap in enumerate(captions):
        end = lengths[i]
        padded_captions[i, :end] = cap[:end]

    return images, padded_captions, lengths


if __name__ == "__main__":
    import json

    # Example usage setup

    # Paths
    image_dir = "/kaggle/input/chest-xrays-indiana-university/images/images_normalized"
    projections_csv = "/kaggle/input/chest-xrays-indiana-university/projections.csv"
    reports_csv = "/kaggle/input/chest-xrays-indiana-university/indiana_reports.csv"
    word_map_path = "/kaggle/working/word_map.json"  # Update this path

    # Load word map
    with open(word_map_path, 'r') as j:
        word_map = json.load(j)

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Create dataset and dataloader
    dataset = IndianaXrayDataset(
        image_dir=image_dir,
        projections_csv=projections_csv,
        reports_csv=reports_csv,
        split='train',  # or 'val', 'test' depending on your CSV
        word_map=word_map,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    # Quick check
    for imgs, caps, cap_lens in dataloader:
        print(f"Batch images shape: {imgs.shape}")
        print(f"Batch captions shape: {caps.shape}")
        print(f"Batch caption lengths: {cap_lens}")
        break
