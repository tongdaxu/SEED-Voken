import argparse
import os
import torch.distributed as dist

from diffusers import AutoencoderKL
from glob import glob
from PIL import Image
from torchvision.datasets import VisionDataset
import torchvision.transforms as transforms
import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
from omegaconf import OmegaConf
from einops import rearrange

import torch.nn.functional as F

from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance

class SimpleDataset(VisionDataset):
    def __init__(self, root: str, transforms=None):
        super().__init__(root, transforms)

        if root.endswith(".txt"):
            with open(root) as f:
                lines = f.readlines()
            self.fpaths = [line.strip("\n") for line in lines]
        else:
            self.fpaths = sorted(glob(root + "/**/*.JPEG", recursive=True))
            self.fpaths += sorted(glob(root + "/**/*.jpg", recursive=True))
            self.fpaths += sorted(glob(root + "/**/*.png", recursive=True))

        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert("RGB")

        if self.transforms is not None:
            img = self.transforms(img)

        return fpath, img
    

def main():

    img_transforms = transforms.Compose(
        [
            transforms.Resize(128, antialias=True),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
        ]
    )

    bs = 500

    ref_dataset = SimpleDataset("/workspace/cogview_dev/xutd/xu/LightningDiT/output/sd3unet_gq_0.25_even/lightningdit_lb-1-ckpt-0080000-euler-250-interval0.11-cfg10.00-shift0.30", img_transforms)

    ref_dataloader = DataLoader(
        ref_dataset,
        bs,
        shuffle=False,
        num_workers=8,
        drop_last=False,
    )
    is_computer = InceptionScore(normalize=True).cuda()

    for i, (_, ref_data) in tqdm(enumerate(ref_dataloader)):
        ref_data = ref_data.cuda()
        is_computer.update(ref_data)
    print(is_computer.compute())

if __name__ == "__main__":
    main()