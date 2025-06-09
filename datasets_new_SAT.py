import os
import glob

class IndianaXrayDataset(Dataset):
    def __init__(self, image_dir, projections_csv, reports_csv, split, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.split = split.upper()
        self.projections = pd.read_csv(projections_csv)
        self.reports = pd.read_csv(reports_csv)
        self.data = self.projections.merge(self.reports, on='uid')

        # Build UID-to-filename map:
        self.uid_to_file = {}
        all_files = glob.glob(os.path.join(self.image_dir, '*'))  # all files in folder
        for f in all_files:
            filename = os.path.basename(f)
            # Extract UID from filename, you can adjust this logic
            # For example, if UID is embedded in filename:
            # Let's say UID is always first 4 digits:
            # uid_in_file = filename.split('_')[0]  # or your custom parsing
            # Or if UID is anywhere in filename:
            for uid in self.data['uid'].astype(str):
                if uid in filename:
                    self.uid_to_file[uid] = f
                    break  # stop after first match

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
