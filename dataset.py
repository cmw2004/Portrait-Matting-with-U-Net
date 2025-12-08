# dataset.py
import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

def load_image(path):
    return np.array(Image.open(path).convert('RGB'))

def load_mask(path):
    return np.array(Image.open(path).convert('L')) / 255.0

class PortraitMattingDataset(Dataset):
    """
    Expects dir structure:
    root/
      images/   (rgb)
      mattes/   (grayscale alpha in [0,1])
      fg/       (optional foreground-only images)
    """
    def __init__(self, root, split='train', size=(320,320), bg_dir=None):
        super().__init__()
        self.root = root
        self.img_dir = os.path.join(root, 'images')
        # support both 'mattes' and 'masks' naming conventions
        candidate_mask_dirs = [os.path.join(root, 'mattes'), os.path.join(root, 'masks')]
        self.mask_dir = None
        if os.path.isdir(candidate_mask_dirs[0]):
            self.mask_dir = candidate_mask_dirs[0]
        elif os.path.isdir(candidate_mask_dirs[1]):
            self.mask_dir = candidate_mask_dirs[1]
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Images directory not found: {self.img_dir}. Make sure `--data_root` points to a folder containing an `images/` subfolder.")
        if self.mask_dir is None:
            raise FileNotFoundError(f"Mask directory not found. Expected one of: {candidate_mask_dirs}. Make sure `--data_root` points to a folder containing `mattes/` or `masks/`.")
        # build image-mask pairs by matching filename stems (handles different extensions)
        img_list = [f for f in sorted(os.listdir(self.img_dir)) if os.path.isfile(os.path.join(self.img_dir, f))]
        mask_list = [f for f in sorted(os.listdir(self.mask_dir)) if os.path.isfile(os.path.join(self.mask_dir, f))]
        # map mask stem -> mask filename (keep first seen if multiple extensions)
        mask_map = {}
        for m in mask_list:
            stem = os.path.splitext(m)[0]
            if stem not in mask_map:
                mask_map[stem] = m
        pairs = []
        missing = []
        for img_name in img_list:
            stem = os.path.splitext(img_name)[0]
            if stem in mask_map:
                pairs.append((img_name, mask_map[stem]))
            else:
                missing.append(img_name)
        if missing:
            print(f"Warning: {len(missing)} images have no corresponding mask and will be skipped. Example: {missing[:5]}")
        self.files = pairs
        self.size = size
        self.bg_dir = bg_dir
        if bg_dir:
            if not os.path.isdir(bg_dir):
                raise FileNotFoundError(f"Background directory not found: {bg_dir}. Provide an existing folder with background images, or pass `--bg_dir None` to disable background augmentation.")
            # only include files
            self.bgs = [os.path.join(bg_dir, f) for f in os.listdir(bg_dir) if os.path.isfile(os.path.join(bg_dir, f))]
        else:
            self.bgs = []
        self.aug = A.Compose([
            A.Resize(*self.size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.ColorJitter(p=0.5),
            # Add more data augmentation to improve edge robustness
            A.GaussNoise(p=0.3),  # Simplified parameters
            A.Blur(blur_limit=3, p=0.2),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            ToTensorV2(),
        ])

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        img_name, mask_name = self.files[idx]
        img = load_image(os.path.join(self.img_dir, img_name))
        mask = load_mask(os.path.join(self.mask_dir, mask_name))
        # Optionally composite with random background for augmentation
        if self.bgs and random.random() < 0.5:
            bg = load_image(random.choice(self.bgs))
            bg = np.array(Image.fromarray(bg).resize((img.shape[1], img.shape[0])))
            alpha = np.expand_dims(mask, 2)
            img = (img * alpha + bg * (1-alpha)).astype(np.uint8)
        augmented = self.aug(image=img, mask=mask)
        image = augmented['image'].float()
        mask = augmented['mask'][None, ...].float()
        return image, mask, img_name
