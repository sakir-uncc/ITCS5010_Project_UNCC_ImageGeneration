import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as T

from config import CONFIG


class CampusLoraDataset(Dataset):
    """
    Simple dataset:
      - images_dir: flat folder of .jpg images
      - captions_dir: parallel folder of .txt files, same stem as image
    """

    def __init__(
        self,
        images_dir: Path,
        captions_dir: Path,
        resolution: int = 512,
        center_crop: bool = True,
        file_list: Optional[List[str]] = None,
    ):
        super().__init__()
        self.images_dir = Path(images_dir)
        self.captions_dir = Path(captions_dir)
        self.resolution = resolution
        self.center_crop = center_crop

        all_files = sorted(
            [f for f in os.listdir(self.images_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        )

        if file_list is not None:
            # filter according to supplied list
            file_set = set(file_list)
            self.image_files = [f for f in all_files if f in file_set]
        else:
            self.image_files = all_files

        # transforms
        transforms = [
            T.Resize(resolution, interpolation=T.InterpolationMode.BILINEAR),
        ]
        if center_crop:
            transforms.append(T.CenterCrop(resolution))
        transforms.extend(
            [
                T.ToTensor(),
                T.Normalize([0.5], [0.5]),  # for latent diffusion
            ]
        )
        self.transform = T.Compose(transforms)

    def __len__(self) -> int:
        return len(self.image_files)

    def _load_image(self, path: Path) -> Image.Image:
        image = Image.open(path).convert("RGB")
        return image

    def _load_caption(self, stem: str) -> str:
        txt_path = self.captions_dir / f"{stem}.txt"
        if not txt_path.exists():
            raise FileNotFoundError(f"Caption file not found: {txt_path}")
        with open(txt_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()
        return caption

    def __getitem__(self, idx: int) -> dict:
        image_name = self.image_files[idx]
        img_path = self.images_dir / image_name
        stem = Path(image_name).stem

        image = self._load_image(img_path)
        image = self.transform(image)

        caption = self._load_caption(stem)

        return {
            "pixel_values": image,
            "caption": caption,
            "image_name": image_name,
        }


def build_train_val_dataloaders(
    cfg=CONFIG,
    val_fraction: float = 0.05,
) -> Tuple[DataLoader, DataLoader]:
    paths = cfg["paths"]
    train_cfg = cfg["training"]

    dataset = CampusLoraDataset(
        images_dir=paths.images_dir,
        captions_dir=paths.captions_dir,
        resolution=train_cfg.image_resolution,
        center_crop=train_cfg.center_crop,
    )

    val_size = max(1, int(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print(f"Dataset split: {train_size} train, {val_size} val")

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader
