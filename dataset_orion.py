import os
import random

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

# ================= Data paths (placeholders) =================
# Expected directory layout:
#   <root>/he/<id>.png
#   <root>/<MarkerName>/<id>.png   (one folder per biomarker, same file names as he/)
DATA_ROOT = {
    "train": "PATH/TO/Orion-CRC/train",
    "val":   "PATH/TO/Orion-CRC/val",
}

# The order defines the marker label id (0-15). Keep it consistent across train/test.
MARKER_LIST = [
    "CD3e", "CD4", "CD8a", "CD20",
    "CD31", "CD45", "CD45RO", "CD68",
    "CD163", "E-cadherin", "FOXP3", "Hoechst",
    "Ki67", "Pan-CK", "PD-L1", "SMA",
]


class OrionDataset(Dataset):
    def __init__(self, mode='train', img_size=256, root=None):
        super().__init__()
        self.mode = mode
        self.img_size = img_size
        self.root = root if root is not None else DATA_ROOT[mode]

        self.markers = MARKER_LIST
        self.marker2id = {name: i for i, name in enumerate(self.markers)}

        print(f"[{mode.upper()}] Loading Orion-CRC Dataset...")
        print(f"    Root: {self.root}")
        print(f"    Markers ({len(self.markers)}): {self.markers}")

        self.he_dir = os.path.join(self.root, "he")
        if not os.path.exists(self.he_dir):
            raise FileNotFoundError(f"HE directory not found: {self.he_dir}")

        self.file_list = sorted([f for f in os.listdir(self.he_dir) if f.endswith(('.jpg', '.png', '.tif'))])
        print(f"    Found {len(self.file_list)} HE images.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        filename = self.file_list[index]

        # 1. Load H&E image (input scaffold)
        he_path = os.path.join(self.he_dir, filename)
        he_img = Image.open(he_path).convert('RGB')

        # 2. Pick a marker: random for training, deterministic (index % N) for validation
        if self.mode == 'train':
            marker_name = random.choice(self.markers)
        else:
            marker_name = self.markers[index % len(self.markers)]
        label_id = self.marker2id[marker_name]

        # 3. Load the corresponding IHC marker image (target)
        ihc_path = os.path.join(self.root, marker_name, filename)
        if not os.path.exists(ihc_path):
            raise FileNotFoundError(f"Target marker image not found: {ihc_path}")
        ihc_img = Image.open(ihc_path).convert('RGB')

        # 4. Resize + (train-only) flips + normalize to [-1, 1]
        he_img = TF.resize(he_img, (self.img_size, self.img_size))
        ihc_img = TF.resize(ihc_img, (self.img_size, self.img_size))

        if self.mode == 'train':
            if random.random() > 0.5:
                he_img = TF.hflip(he_img)
                ihc_img = TF.hflip(ihc_img)
            if random.random() > 0.5:
                he_img = TF.vflip(he_img)
                ihc_img = TF.vflip(ihc_img)

        he_tensor = TF.to_tensor(he_img) * 2.0 - 1.0
        ihc_tensor = TF.to_tensor(ihc_img) * 2.0 - 1.0

        return he_tensor, ihc_tensor, label_id
