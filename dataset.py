import os
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Default ImageNet normalization (overridden at runtime by processor values)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class VirtualStainingDataset(Dataset):


    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_size: int = 224,
        mean: tuple = IMAGENET_MEAN,
        std: tuple = IMAGENET_STD,
        augment: bool = True,
    ):
        self.hes_dir = Path(root_dir) / split / "HES"
        self.cd30_dir = Path(root_dir) / split / "CD30"
        self.image_size = image_size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

        # Intersect filenames to guarantee perfect pairing
        hes_files = set(os.listdir(self.hes_dir))
        cd30_files = set(os.listdir(self.cd30_dir))
        self.filenames = sorted(hes_files & cd30_files)

        if not self.filenames:
            raise ValueError(
                f"No matching files between {self.hes_dir} and {self.cd30_dir}"
            )

        # --- Spatial transforms (applied identically to both images) ----------
        spatial_ops = []
        if augment:
            spatial_ops += [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ]
        spatial_ops.append(A.Resize(image_size, image_size))

        self.spatial_transform = A.Compose(
            spatial_ops,
            additional_targets={"target": "image"},
        )

        # --- Color augmentation (HES input only) -----------------------------
        self.color_transform = None
        if augment:
            self.color_transform = A.Compose(
                [
                    A.ColorJitter(
                        brightness=0.1,
                        contrast=0.1,
                        saturation=0.1,
                        hue=0.02,
                        p=0.5,
                    ),
                ]
            )

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        fname = self.filenames[idx]

        # Load images (OpenCV reads BGR → convert to RGB)
        hes = cv2.imread(str(self.hes_dir / fname))
        hes = cv2.cvtColor(hes, cv2.COLOR_BGR2RGB)

        cd30 = cv2.imread(str(self.cd30_dir / fname))
        cd30 = cv2.cvtColor(cd30, cv2.COLOR_BGR2RGB)

        # Shared spatial augmentations
        augmented = self.spatial_transform(image=hes, target=cd30)
        hes = augmented["image"]
        cd30 = augmented["target"]

        # HES-only colour augmentation
        if self.color_transform is not None:
            hes = self.color_transform(image=hes)["image"]

        # uint8 → float32 [0, 1]
        hes = hes.astype(np.float32) / 255.0
        cd30 = cd30.astype(np.float32) / 255.0

        # Normalise HES with encoder-specific statistics
        hes = (hes - self.mean) / self.std

        # (H, W, C) → (C, H, W) tensors
        hes = torch.from_numpy(hes).permute(2, 0, 1).contiguous()
        cd30 = torch.from_numpy(cd30).permute(2, 0, 1).contiguous()

        return hes, cd30, fname
