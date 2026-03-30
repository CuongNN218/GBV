import os
import pandas as pd
import random
import numpy as np
import torch
import math

from torchvision.transforms.v2 import functional as fv2
from enum import Enum
from typing import Dict, List, Optional, Tuple
from torch import Tensor
from  torchvision.transforms import functional as F, InterpolationMode
from  torchvision.transforms import AutoAugment, RandAugment, TrivialAugmentWide
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset

# implementation of gaussian noise 

def gaussian_noise_image(image: torch.Tensor, mean: float = 0.0, sigma: float = 0.1, clip: bool = True) -> torch.Tensor:
        if not image.is_floating_point():
            raise ValueError(f"Input tensor is expected to be in float dtype, got dtype={image.dtype}")
        if sigma < 0:
            raise ValueError(f"sigma shouldn't be negative. Got {sigma}")
        noise = mean + torch.randn_like(image) * sigma
        out = image + noise
        if clip:
            out = torch.clamp(out, 0, 1)
        return out

def _augmentation_space(num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
            "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
#            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Rotate": (torch.linspace(0.0, 90, num_bins), True),
#            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True), 
            "Brightness": (torch.linspace(0.0, 5.0, num_bins), True), # new
            "Color": (torch.linspace(0.0, 5.0, num_bins), True),
#            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
#            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 5.0, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "GaussianBlur": (torch.linspace(0.1, 5.0, num_bins), False),
            "GaussianNoise": (torch.linspace(0.1, 3.0, num_bins),False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
            "Invert": (torch.tensor(0.0), False),
        }

def _apply_op(
    img: Tensor, op_name: str, magnitude: float, interpolation: InterpolationMode, fill: Optional[List[float]]
):
    
#    prob = np.random.uniform(low=0.2, high=0.8) 
#    p = np.random.uniform(low=0., high=1.)
#    if p >= prob:
#        return img
    
    if op_name == "ShearX":
        # magnitude should be arctan(magnitude)
        # official autoaug: (1, level, 0, 0, 1, 0)
        # https://github.com/tensorflow/models/blob/dd02069717128186b88afa8d857ce57d17957f03/research/autoaugment/augmentation_transforms.py#L290
        # compared to
        # torchvision:      (1, tan(level), 0, 0, 1, 0)
        # https://github.com/pytorch/vision/blob/0c2373d0bba3499e95776e7936e207d8a1676e65/torchvision/transforms/functional.py#L976
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(math.atan(magnitude)), 0.0],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "ShearY":
        # magnitude should be arctan(magnitude)
        # See above
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(math.atan(magnitude))],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "TranslateX":
        img = F.affine(
            img,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "Rotate":
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
#        img = F.adjust_brightness(img, 1.0 + magnitude)
        img = F.adjust_brightness(img, magnitude)
    elif op_name == "Color":
#        img = F.adjust_saturation(img, 1.0 + magnitude)
        img = F.adjust_saturation(img, magnitude)
    elif op_name == "Contrast":
        img = F.adjust_contrast(img, magnitude)
#        img = F.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
#        img = F.adjust_sharpness(img, 1.0 + magnitude)
        img = F.adjust_sharpness(img, magnitude)
    elif op_name == "Posterize":
        img = F.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = F.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = F.autocontrast(img)
    elif op_name == "Equalize":
        img = F.equalize(img)
    elif op_name == "Invert":
        img = F.invert(img)
    elif op_name == "GaussianBlur":
        img = F.gaussian_blur(img, 11, magnitude)
    elif op_name == "GaussianNoise":
        img = gaussian_noise_image(img, 0.0, sigma=magnitude)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    return img

def _random_augment(self, img, target):
    
    op_names = random.sample(list(self.op_meta.keys()), k=2)
    
    # always apply GaussianNoise after
    if "GaussianNoise" in op_names[0]:
        op_names = [op_names[1], op_names[0]]
    
    for op_name in op_names:
        magnitudes, signed = self.op_meta[op_name]
        magnitude = (
            float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item())
            if magnitudes.ndim > 0
            else 0.0
        )
        
        if signed and torch.randint(2, (1,)):
            magnitude *= -1.0
        
        if op_name is "GaussianNoise":
            img = self.base_transform[1](img)               
            img = _apply_op(img, op_name, magnitude, interpolation=InterpolationMode.NEAREST, fill=None)
        else: 
            img = _apply_op(img, op_name, magnitude, interpolation=InterpolationMode.NEAREST, fill=None)
    
    if not "GaussianNoise" in op_names[1]:
        img = self.base_transform[1](img)

    return img, target, 1.0

def _trivial_augment(self, img, target):
    op_index = int(torch.randint(len(self.op_meta), (1,)).item())
    op_name = list(self.op_meta.keys())[op_index]
    magnitudes, signed = self.op_meta[op_name]
    magnitude = (
        float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item())
        if magnitudes.ndim > 0
        else 0.0
    )
    
    if signed and torch.randint(2, (1,)):
        magnitude *= -1.0
    
    if op_name is "GaussianNoise":
        img = self.base_transform[1](img)               
        img = _apply_op(img, op_name, magnitude, interpolation=InterpolationMode.NEAREST, fill=None)
    else: 
        img = _apply_op(img, op_name, magnitude, interpolation=InterpolationMode.NEAREST, fill=None)
        img = self.base_transform[1](img)
    
    return img, target, 1.0

def _weight_augment(self, img, target, idx):
    if  self.weights_df.shape[0] > 1:
        op_idx  = random.choices(population=self.weights_df.index.tolist(),
                                 weights=self.weights_df["probability"].tolist(),
                                 k=1)[0] 
        op_name = self.weights_df.iat[op_idx, self.weights_df.columns.get_loc("op_name")]
        magnitude = self.weights_df.iat[op_idx, self.weights_df.columns.get_loc("magnitude")]
        weight = self.weights_df.iat[op_idx, self.weights_df.columns.get_loc("probability")]
        signed = self.weights_df.iat[op_idx, self.weights_df.columns.get_loc("signed")] 
    
    else:
        op_name = self.weights_df.iloc[0]["op_name"]
        magnitude = self.weights_df.iloc[0]["magnitude"]
        signed = self.weights_df.iloc[0]["signed"]
        weight = 1.0

    if signed and torch.randint(2, (1,)):
        magnitude *= -1.0
    
    if "GaussianNoise" in op_name:
        img = self.base_transform[1](img)               
        img = _apply_op(img, op_name, magnitude, interpolation=InterpolationMode.NEAREST, fill=None)
    else: 
        img = _apply_op(img, op_name, magnitude, interpolation=InterpolationMode.NEAREST, fill=None)
        img = self.base_transform[1](img)
    
    return idx, img, target, weight

    
def _ent_augment(self, img, target, index):

    op_index = int(torch.randint(len(self.op_meta), (1,)).item())
    op_name = list(self.op_meta.keys())[op_index]
    magnitudes, signed = self.op_meta[op_name]
    level_max = self.num_bins - 1        
    level = min(int(level_max * self.MAGNITUDE[index]) + 1, level_max)       
    magnitude = (float(magnitudes[level].item()) if magnitudes.ndim > 0 else 0.0)

    if signed and torch.randint(2, (1,)):
        magnitude *= -1.0
    
    if op_name is "GaussianNoise":
        img = self.base_transform[1](img)               
        img = _apply_op(img, op_name, magnitude, interpolation=InterpolationMode.NEAREST, fill=None)
    else: 
        img = _apply_op(img, op_name, magnitude, interpolation=InterpolationMode.NEAREST, fill=None)
        img = self.base_transform[1](img)
    return index, img, target, 1.0


class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'

    def __init__(self, 
                root, 
                train=True, 
                weights_df=None,
                base_transform=[],
                img_size=(224,224),
                num_bins=5,
                loader=default_loader, 
                download=True,
                strategy="weight"):

        self.root = os.path.expanduser(root)
        
        self.weights_df = weights_df
        self.base_transform = base_transform
        self.loader = default_loader
        self.train = train
        self.num_bins = num_bins
        
        print("Num Bins: ", num_bins)
        self.strategy = strategy
        print() 
        # load data
        self._load_metadata()
        print("Data len", len(self.data))
        if self.strategy == "aa":
            print("Using AutoAugment")
            self.transform = AutoAugment()
        
        elif self.strategy == "ra":
            print("Using Rand Augment")
            self.op_meta = _augmentation_space(self.num_bins, image_size=img_size)
        
        elif self.strategy == "ta":
            print(f"Using Trivial Augmentation with {self.num_bins}.")
            self.op_meta = _augmentation_space(self.num_bins, image_size=img_size)
            print("Op_meta",self.op_meta)
        elif self.strategy == 'ent':
            print(f"Using EntAugment")
            self.op_meta = _augmentation_space(self.num_bins, img_size)
            self.MAGNITUDE = torch.zeros(len(self.data))
        elif self.strategy == 'sra':
            print("Using sample aware augment")
            self.op_meta = _augmentation_space(self.num_bins, img_size)
        else:
            print("Using the weighted version")
            self.transform = None
            self.op_meta = _augmentation_space(self.num_bins, img_size)
    

    def _set_magnitude(self, idx, magnitude):
        self.MAGNITUDE[idx] = magnitude


    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]
    

    def _random_augment(self, img, target):
        
        op_names = random.sample(list(self.op_meta.keys()), k=2)
        
        # always apply GaussianNoise after
        if "GaussianNoise" in op_names[0]:
            op_names = [op_names[1], op_names[0]]
        
        for op_name in op_names:
            magnitudes, signed = self.op_meta[op_name]
            magnitude = (
                float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item())
                if magnitudes.ndim > 0
                else 0.0
            )
            
            if signed and torch.randint(2, (1,)):
                magnitude *= -1.0
            
            if "GaussianNoise" in op_name:
                img = self.base_transform[1](img)               
                img = _apply_op(img, op_name, magnitude, interpolation=InterpolationMode.NEAREST, fill=None)
            else: 
                img = _apply_op(img, op_name, magnitude, interpolation=InterpolationMode.NEAREST, fill=None)
        
        if not "GaussianNoise" in op_names[1]:
            img = self.base_transform[1](img)

        return img, target, 1.0

    def _trivial_augment(self, img, target):
        op_index = int(torch.randint(len(self.op_meta), (1,)).item())
        op_name = list(self.op_meta.keys())[op_index]
        magnitudes, signed = self.op_meta[op_name]
        magnitude = (
            float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item())
            if magnitudes.ndim > 0
            else 0.0
        )
        
        if signed and torch.randint(2, (1,)):
            magnitude *= -1.0
        
        if "GaussianNoise" in op_name:
            img = self.base_transform[1](img)               
            img = _apply_op(img, op_name, magnitude, interpolation=InterpolationMode.NEAREST, fill=None)
        else: 
            img = _apply_op(img, op_name, magnitude, interpolation=InterpolationMode.NEAREST, fill=None)
            img = self.base_transform[1](img)
        
        return img, target, 1.0
    
    
    def _ent_augment(self, img, target, index):

        op_index = int(torch.randint(len(self.op_meta), (1,)).item())
        op_name = list(self.op_meta.keys())[op_index]
        magnitudes, signed = self.op_meta[op_name]
        level_max = self.num_bins - 1        
        level = min(int(level_max * self.MAGNITUDE[index]) + 1, level_max)       
        magnitude = (float(magnitudes[level].item()) if magnitudes.ndim > 0 else 0.0)

        if signed and torch.randint(2, (1,)):
            magnitude *= -1.0
        
        if op_name is "GaussianNoise":
            img = self.base_transform[1](img)               
            img = _apply_op(img, op_name, magnitude, interpolation=InterpolationMode.NEAREST, fill=None)
        else: 
            img = _apply_op(img, op_name, magnitude, interpolation=InterpolationMode.NEAREST, fill=None)
            img = self.base_transform[1](img)
        return index, img, target, 1.0
    
    def _sample_aware_augment(self, img, target):
        
#        op_names = random.sample(list(self.op_meta.keys()), k=2)
#        ori_img = img 
#        # always apply GaussianNoise after
#        if "GaussianNoise" in op_names[0]:
#            op_names = [op_names[1], op_names[0]]
#        
#        for op_name in op_names:
#            magnitudes, signed = self.op_meta[op_name]
#            magnitude = (
#                float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item())
#                if magnitudes.ndim > 0
#                else 0.0
#            )
#            if signed and torch.randint(2, (1,)):
#                magnitude *= -1.0
#            
#            if op_name is "GaussianNoise":
#                img = self.base_transform[1](img)               
#                img = _apply_op(img, op_name, magnitude, interpolation=InterpolationMode.NEAREST, fill=None)
#            else: 
#                img = _apply_op(img, op_name, magnitude, interpolation=InterpolationMode.NEAREST, fill=None)
#        
#        if not "GaussianNoise" in op_names[1]:
#            img = self.base_transform[1](img)
#        return F.to_tensor(ori_img), img, target, 1.0
        return F.to_tensor(img), target, 1.0


    def _weight_augment(self, img, target, idx):
        if  self.weights_df.shape[0] > 1:
            op_idx  = random.choices(population=self.weights_df.index.tolist(),
                                     weights=self.weights_df["probability"].tolist(),
                                     k=1)[0] 
            op_name = self.weights_df.iat[op_idx, self.weights_df.columns.get_loc("op_name")]
            magnitude = self.weights_df.iat[op_idx, self.weights_df.columns.get_loc("magnitude")]
            weight = self.weights_df.iat[op_idx, self.weights_df.columns.get_loc("probability")]
            signed = self.weights_df.iat[op_idx, self.weights_df.columns.get_loc("signed")] 
        
        else:
            op_name = self.weights_df.iloc[0]["op_name"]
            magnitude = self.weights_df.iloc[0]["magnitude"]
            signed = self.weights_df.iloc[0]["signed"]
            weight = 1.0

        if signed and torch.randint(2, (1,)):
            magnitude *= -1.0
        
        if "GaussianNoise" in op_name:
            img = self.base_transform[1](img)               
            img = _apply_op(img, op_name, magnitude, interpolation=InterpolationMode.NEAREST, fill=None)
        else: 
            img = _apply_op(img, op_name, magnitude, interpolation=InterpolationMode.NEAREST, fill=None)
            img = self.base_transform[1](img)
        
        return idx, img, target, weight

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)
        # for crop and resize
        img = self.base_transform[0](img)

        if self.train:
            if "aa" in self.strategy:
                img = self.transform(img)
                img = self.base_transform[1](img)
                return img, target, 1.0
            elif "ent" in self.strategy:
                return self._ent_augment(img, target, idx)
            elif "sra" in self.strategy:
                return self._sample_aware_augment(img, target)
            elif "ra" in self.strategy:
                return self._random_augment(img, target)
            elif "ta" in self.strategy:
                return self._trivial_augment(img, target)
            elif "weight" in self.strategy:
                return self._weight_augment(img, target, idx)
        else:
            img = self.base_transform[0](img)
            img = self.base_transform[1](img)
            return img, target, 1.0
