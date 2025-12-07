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

    # dis_npz = "/workspace/cogview_dev/xutd/npz/lfq/samples/top_k_0_temp_1.15_top_p_1.0_cfg_4.0/epoch=100-step=252803.ckpt/sample.npz"
    dis_npz = "/workspace/cogview_dev/xutd/npz/bsq_ffhq/samples/top_k_0_temp_1.15_top_p_1.0_cfg_1.0/epoch=299-step=41100.ckpt/sample.npz"

    bs = 500

    ref_dataset = SimpleDataset("/workspace/cogview_dev/xutd/xu/datasets/ffhq/thumbnails128x128", img_transforms)
    dis_datas = np.load(dis_npz)['arr_0']

    ref_dataloader = DataLoader(
        ref_dataset,
        bs,
        shuffle=False,
        num_workers=8,
        drop_last=False,
    )
    is_computer = InceptionScore(normalize=True).cuda()
    fid_computer = FrechetInceptionDistance(normalize=True).cuda()

    for i, (_, ref_data) in tqdm(enumerate(ref_dataloader)):
        ref_data = ref_data.cuda()
        dis_data = (torch.tensor(dis_datas[i * bs: (i+1)*bs]).float() / 255.0).permute(0,3,1,2).cuda()
        is_computer.update(dis_data)
        fid_computer.update(F.interpolate(ref_data, 299),real=True)
        fid_computer.update(F.interpolate(dis_data, 299),real=False)
    print(is_computer.compute())
    print(fid_computer.compute())

if __name__ == "__main__":
    main()