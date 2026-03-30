import os
import os.path as osp

import albumentations as A
import cv2
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader


class CUB2011(Dataset):
    def __init__(self,
                 train=True,
                 transforms=None,
                 base_transform=None,
                 weights=None,
                 cfg=None,
                 parts=[]):
        self.root = cfg.dataset.path
        self.transforms = transforms
        self.base_transform = base_transform
        
        self.parts = parts # to select part id
        self.train = train
        self.remove = True
        self.weights = weights
        
        self._load_metadata()
        self.task_name = task_name
        
        self.transforms = transforms
        
        self.resize = A.Compose([A.geometric.resize.Resize(224, 224, cv2.INTER_CUBIC)],
                                keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'images.txt'),
                             sep=' ',
                             names=['img_id', 'filepath'])

        image_part_labels = pd.read_csv(
            os.path.join(self.root, 'parts', 'part_locs.txt'),
            sep=' ',
            names=['img_id', 'part_id', 'x', 'y', 'visible'])

        train_test_split = pd.read_csv(os.path.join(self.root, 'train_test_split.txt'),
                                        sep=' ', 
                                        names=['img_id', 'is_training_img'])
        
#        train_test_split = pd.read_csv(
#            os.path.join(self.root, 'new_train_test_split.csv'))
        self.data = images.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
            image_part_labels = image_part_labels[
                image_part_labels.img_id.isin(self.data.img_id.values)]
            image_part_labels = image_part_labels[image_part_labels.part_id.isin(self.parts)]
            self.keypoints = image_part_labels
            self.keypoints = image_part_labels[image_part_labels["visible"] != 0]
            self.data = self.data[self.data.img_id.isin(self.keypoints["img_id"].values)]
        
        else:
            self.data = self.data[self.data.is_training_img == 0]
            image_part_labels = image_part_labels[
                image_part_labels.img_id.isin(self.data.img_id.values)]
            image_part_labels = image_part_labels[
                image_part_labels.part_id.isin(self.parts)]
            self.keypoints = image_part_labels
            if self.remove:
                self.keypoints = image_part_labels[image_part_labels["visible"] != 0]
            else:
                self.keypoints = image_part_labels
            self.data = self.data[self.data.img_id.isin(self.keypoints['img_id'].values)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        info = self.data.iloc[idx]
        file_path = info['filepath']
        img = cv2.imread(osp.join(self.root, 'images', file_path))[..., ::-1]

        keypoints = self.keypoints.loc[self.keypoints['img_id'] ==info['img_id']]
        keypoints = keypoints[['x', 'y']].astype(int)
        keypoints = [tuple(x) for x in keypoints.values]
        
        #resize image to the target size
        resized = self.resize(image=img, keypoints=keypoints)
        img, keypoints = resized["image"], resized["keypoints"]

        # select transformation form a list with probs
        if self.train:

            if len(self.transforms) > 1:
                transform_idx = random.choice(self.transform_idxs, self.weights, k=1)
                transformed = self.transforms[transform_idx](image=img, keypoints=keypoints)
            else:
                transformed = self.transforms[0](image=img, keypoints=keypoints)
            
            transformed = self.base_transform(image=transformed["image"],
                    keypoints=transformed["keypoints"])
            
            img, keypoint_list = transformed['image'], transformed['keypoints']
            gt = torch.Tensor(keypoint_list).flatten()
            return img, gt, weight
        else:
            transformed = self.base_transform(image=img,
                                            keypoints=keypoints)
            img, keypoint_list = transformed["image"], transformed["keypoints"] 
            gt = torch.Tensor(keypoint_list).flatten()
            return img, gt

def build_cub_aug_dataset(cfg, transform, weights, parts=[]):
    
    base_transform = A.Compose([A.Normalize(), 
                                ToTensorV2()],
                                keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    train_dataset = CUB2011(train=True, transforms=transform, weigghts=weights, base_transform=base_transform, cfg, parts=parts)
    test_dataset = CUB2011(train=False, transforms=None, base_transform=base_transform, cfg=cfg, parts=parts)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(test_dataset, batch_size, shuffle= False, num_workers=2)

    return train_loader, valid_loader
