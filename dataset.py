import os
import os.path as osp
import math

from typing import Union, Dict, List
from pathlib import Path

from PIL import Image, ImageDraw
import cv2
import numpy as np

import torch.utils.data
import torch.utils
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms as TF


def RandomBrush(
    max_tries,
    s,
    min_num_vertex=4,
    max_num_vertex=18,
    mean_angle=2*math.pi / 5,
    angle_range=2*math.pi / 15,
    min_width=12,
    max_width=48
):
    H, W = s, s
    average_radius = math.sqrt(H*H+W*W) / 8
    mask = Image.new('L', (W, H), 0)
    for _ in range(np.random.randint(max_tries)):
        num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
        angle_min = mean_angle - np.random.uniform(0, angle_range)
        angle_max = mean_angle + np.random.uniform(0, angle_range)
        angles = []
        vertex = []
        for i in range(num_vertex):
            if i % 2 == 0:
                angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
            else:
                angles.append(np.random.uniform(angle_min, angle_max))

        h, w = mask.size
        vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
        for i in range(num_vertex):
            r = np.clip(
                np.random.normal(loc=average_radius, scale=average_radius//2),
                0, 2*average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))

        draw = ImageDraw.Draw(mask)
        width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width//2,
                          v[1] - width//2,
                          v[0] + width//2,
                          v[1] + width//2),
                         fill=1)
        if np.random.random() > 0.5:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.random() > 0.5:
            mask.transpose(Image.FLIP_TOP_BOTTOM)
    mask = np.asarray(mask, np.uint8)
    if np.random.random() > 0.5:
        mask = np.flip(mask, 0)
    if np.random.random() > 0.5:
        mask = np.flip(mask, 1)
    return mask


def RandomMask(s, hole_range=[0,1]):
    coef = min(hole_range[0] + hole_range[1], 1.0)
    while True:
        mask = np.ones((s, s), np.uint8)

        def Fill(max_size):
            w, h = np.random.randint(max_size), np.random.randint(max_size)
            ww, hh = w // 2, h // 2
            x, y = np.random.randint(-ww, s - w + ww), np.random.randint(-hh, s - h + hh)
            mask[max(y, 0): min(y + h, s), max(x, 0): min(x + w, s)] = 0

        def MultiFill(max_tries, max_size):
            for _ in range(np.random.randint(max_tries)):
                Fill(max_size)

        MultiFill(int(10 * coef), s // 2)
        MultiFill(int(5 * coef), s)
        mask = np.logical_and(mask, 1 - RandomBrush(int(20 * coef), s))
        hole_ratio = 1 - np.mean(mask)
        if hole_range is not None and (hole_ratio <= hole_range[0] or hole_ratio >= hole_range[1]):
            continue
        return (mask * 255).astype(np.uint8)
    


class SimpleInferDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_dir: Path,
        mask_dir: Path = None,
        resolution: int = 512
    ):
        super(SimpleInferDataset, self).__init__()

        img_extensions = {".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"}
        self.img_paths  = sorted([i for i in Path(img_dir).iterdir() if i.suffix in img_extensions])
        self.img_dir = img_dir

        if mask_dir:
            self.mask_paths = sorted([i for i in Path(mask_dir).iterdir() if i.suffix in img_extensions])
            self.mask_dir = mask_dir

        self.resolution = resolution

    def __getitem__(self, index):
        if self.mask_dir:
            mask_path = Path(self.mask_paths[index])
            mask = Image.open(mask_path).convert("L")
        else:
            mask = RandomMask(img.size[0])
            mask = Image.fromarray(mask).convert("L")
        
        mask = mask.resize((self.resolution, self.resolution), Image.NEAREST)

        img_path = Path(self.img_paths[index])
        img_name = os.path.basename(img_path)

        img = Image.open(img_path).convert("RGB")
        if img.size[0] != self.resolution or img.size[1] != self.resolution:
            img = img.resize((self.resolution, self.resolution), Image.BICUBIC)

        return img, mask, img_name

    def __len__(self):
        return len(self.img_paths)