import datetime as dt
import os
import scipy.io
import torch

from pathlib import Path
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from tqdm import tqdm
from torchvision.transforms import functional as F, InterpolationMode
from torchvision.transforms import AutoAugment
from cub_aug_cls import gaussian_noise_image, _augmentation_space
from cub_aug_cls import _random_augment, _trivial_augment, _weight_augment, _ent_augment

class StanfordDogs(Dataset):
    """`Stanford Dogs <http://vision.stanford.edu/aditya86/ImageNetDogs/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``omniglot-py`` exists.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        download (bool, optional): If true, downloads the dataset tar files from the internet and
            puts it in root directory. If the tar files are already downloaded, they are not
            downloaded again.
    """
    folder = 'StanfordDogs'
    download_url_prefix = 'http://vision.stanford.edu/aditya86/ImageNetDogs'
	
    def __init__(self,
                 root,
                 train=True,
                 weight_df=None,
                 base_transform=None,
                 img_size=(224,224),
                 num_bins=5,
                 strategy="weight",
                 download=False):
		
        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.train = train
        self.base_transform = base_transform
        
        self.img_size = img_size 
        self.weights_df = weight_df
        self.num_bins = num_bins
        
        if download:
            self.download()

        split = self.load_split()
        self.images_folder = os.path.join(self.root, 'Images')

        self._breed_images = [(annotation+'.jpg', idx) for annotation, idx in split]
        self._flat_breed_images = self._breed_images
        
        print("Num samples: ", len(self._flat_breed_images))
        print(f"Using strategy: {strategy}")
        print(f"Mode train: {self.train} ")
        if strategy == "aa":
            self.strategy = "aa"
            self.op_meta = _augmentation_space(self.num_bins, self.img_size)
            self.transform = AutoAugment()
        elif strategy == "ra":
            self.strategy = "ra"
            self.op_meta = _augmentation_space(self.num_bins, self.img_size)
            self.transform = _random_augment
        elif strategy == "ta":
            self.op_meta = _augmentation_space(self.num_bins, self.img_size)
            self.strategy = "ta"
            self.transform = _trivial_augment
        elif strategy == "ent":
            print("Using entropy augment")
            self.op_meta = _augmentation_space(self.num_bins, self.img_size)
            self.strategy = "ent"
            self.transform = _ent_augment
            self.MAGNITUDE = torch.zeros(len(self._flat_breed_images))
        elif strategy == "sra":
            print("Using sample aware rand augmentation")
            self.strategy = strategy
            self.op_meta = _augmentation_space(self.num_bins, self.img_size)
            self.transform = None
        elif "weight" in strategy:
            self.strategy = "weight"
            self.transform = _weight_augment
        else:
            self.transform = None
        
    def load_split(self):
        if self.train:
            split = scipy.io.loadmat(os.path.join(self.root, 'train_list.mat'))['annotation_list']
            labels = scipy.io.loadmat(os.path.join(self.root, 'train_list.mat'))['labels']
        else:
            split = scipy.io.loadmat(os.path.join(self.root, 'test_list.mat'))['annotation_list']
            labels = scipy.io.loadmat(os.path.join(self.root, 'test_list.mat'))['labels']

        split = [item[0][0] for item in split]
        labels = [item[0]-1 for item in labels]
        return list(zip(split, labels))
    
    def _set_magnitude(self, idx, magnitude):
        self.MAGNITUDE[idx] = magnitude

    def __len__(self):
        return len(self._flat_breed_images)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        image_name, target_class = self._flat_breed_images[index]
        image_path = os.path.join(self.images_folder, image_name)
        image = Image.open(image_path).convert('RGB')
        image = self.base_transform[0](image)
        
        if self.train:
            if self.strategy == "aa":
                image = self.transform(image)
                image = self.base_transform[1](image)
                return image, target_class, 1.0
            elif self.strategy == "sra":
                return F.to_tensor(image), target_class, 1.0
            elif self.strategy == "ra":
                image, target_class, weight = _random_augment(self, image, target_class)
            elif self.strategy == 'ent':
                return _ent_augment(self, image,target_class, index)
            elif self.strategy == "ta":
                image, target_class, weight = _trivial_augment(self, image, target_class)
            elif "weight" in self.strategy:
                return _weight_augment(self, image, target_class, index)
            return image, target_class, weight
        else:
            image = self.base_transform[0](image)
            image = self.base_transform[1](image)
            return image, target_class, 1.0

    def download(self):
        import tarfile

        if os.path.exists(os.path.join(self.root, 'Images')) and os.path.exists(os.path.join(self.root, 'Annotation')):
            if len(os.listdir(os.path.join(self.root, 'Images'))) == len(os.listdir(os.path.join(self.root, 'Annotation'))) == 120:
                return

        for filename in ['images', 'annotation', 'lists']:
            tar_filename = filename + '.tar'
            url = self.download_url_prefix + '/' + tar_filename
            download_url(url, self.root, tar_filename, None)
            print('Extracting downloaded file: ' + os.path.join(self.root, tar_filename))
            with tarfile.open(os.path.join(self.root, tar_filename), 'r') as tar_file:
                tar_file.extractall(self.root)
            os.remove(os.path.join(self.root, tar_filename))

	
