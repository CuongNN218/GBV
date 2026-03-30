import os 
import torch 
import pickle as pkl
import numpy as np
import warnings 
import torch.utils.data as torchdata
import random

from torch.utils.data import Dataset, DataLoader
from utils import worker_init_reset_seed 
from torch.utils.data.sampler import Sampler, BatchSampler
from torchvision.datasets import CIFAR10
warnings.filterwarnings("ignore")


class NoisyDataset(Dataset):
    def __init__(self, root_dir, pkl_file, transform=None, k=-1,train=False):
        
        self.file_dir = os.path.join(root_dir, pkl_file)
        self.train = train
        with open(self.file_dir, 'rb') as input_file:
            data = pkl.load(input_file)
        
        self.transform = transform
        self.imgs = data[0]
        self.labels = data[1]

        if k != -1:
            self.imgs, self.labels = self.get_subset(data, subset_size=k)
        else:
            self.imgs = data[0]
            self.labels = data[1]
        print("dataset image size: ", self.imgs.shape[0])

    def __len__(self):
        return self.imgs.shape[0]
    
    
    def get_subset(self, data, subset_size):
        
        print(f"Selecting {subset_size} samples per class.") 
        total_idxs = []
        unique_labels = np.unique(data[1])

        for label in unique_labels:
            label_idxs = np.where(data[1] == label)[0].tolist()
            selected_idxs = random.sample(label_idxs, k=subset_size)
            total_idxs.extend(selected_idxs)

        imgs = data[0][total_idxs]
        labels = data[1][total_idxs]

        print("Get subset of imgs shape: ", imgs.shape)
        print("Get subset of labels shape: ", labels.shape )
            
        return imgs, labels
     
    
    def __getitem__(self, idx):
        img, label = self.imgs[idx], self.labels[idx] 
        
        if self.transform:
            img = self.transform(img)
        if self.train:
            return idx, img, label, 1.0
        else:
            return idx, img, label, 1.0

class TestCIFAR10(CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
    
    def __getitem__(self, index):
        img, target = super().__getitem__(index)  # Get the original image and label
        return img, target, 1.0  # Return an additional 1.0 value



class WeightedSampler(Sampler):
    
    '''
        Custom weighted sampler from annotators.
    '''
    
    def __init__(self, dataset_size, dataset_weights, idxs, batch_size):
        """
            Custom sampler to sample a batch from multiple datasets based on weights.
            Args:
                datasets (list): List of PyTorch datasets.
                dataset_weights (list): Sampling weights for each dataset.
                batch_size (int): Number of samples in a batch.
        """
        self.dataset_size = dataset_size
        self.dataset_weights = dataset_weights
        print(f"Using sampler with weights: {self.dataset_weights}")
        self.idxs = idxs
        self.max_iters = self.dataset_size * len(self.dataset_weights) // batch_size
        self.batch_size = batch_size

    def _make_batch(self):
        batch = []
        for _ in range(self.batch_size):
            dataset_id = random.choices(self.idxs, weights=self.dataset_weights, k=1)[0]
            sample_idx = random.randint(0,  self.dataset_size - 1)
            batch.append([dataset_id, sample_idx])
        return batch
    
    def __iter__(self):
        multi_batches = []
        
        for _ in range(self.max_iters):
            single_batch = self._make_batch()
            multi_batches.extend(single_batch)

        return iter(multi_batches)

    def __len__(self):
        return self.max_iters

class WeightedBuyerDataset(Dataset):
    
    def __init__(self, datasets, transform=None):
        
        self.datasets = datasets
        self.dataset_lengths = [len(ds) for ds in datasets]
        self.transform = transform
        
        for i, dataset in enumerate(datasets):
            print(i, dataset.imgs.shape, dataset.labels.shape)

    def __len__(self):
        return len(self.datasets[0])    
    
    def __getitem__(self, idx):
        
        dataset_id, sample_idx = idx
        img, label = self.datasets[dataset_id].imgs[sample_idx], self.datasets[dataset_id].labels[sample_idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, label



class WeightedBuyerDatasetV2(Dataset):
    
    def __init__(self, datasets, weights, transform=None):
        
        self.datasets = datasets
        self.dataset_lengths = [len(ds) for ds in datasets]
        self.transform = transform
        
        for (dataset, weight) in zip(datasets, weights):
            print(dataset.imgs.shape, dataset.labels.shape, weight)

        imgs = [dataset.imgs for dataset in datasets]
        labels = [dataset.labels for dataset in datasets] 
        dataset_weights = [weights[i] * np.ones_like(dataset.labels) for i, dataset in enumerate(datasets)]
        
        self.imgs = np.concatenate(imgs, axis=0)
        self.labels = np.concatenate(labels, axis=0)
        self.weights = np.concatenate(dataset_weights, axis=0)
        
        print("Imgs: ", self.imgs.shape)
        print("Labels: ", self.labels.shape)
        print("Weights: ", self.weights.shape)
    
    def __len__(self):
        return self.imgs.shape[0]    
    
    def __getitem__(self, idx):
        
        img, label, weight  = self.imgs[idx], self.labels[idx], self.weights[idx]
        
        if self.transform:
            img = self.transform(img)
        
        return idx, img, label, weight

def build_single_dataloader(cfg, train_file, valid_file, transform=None):
    
    root = cfg.dataset.path
    k = cfg.dataset.subset_size
    print("Using subset size: ", k)    
    train_dataset = NoisyDataset(root_dir=root, 
                                 pkl_file=train_file, 
                                 transform=transform, 
                                 k=k,
                                 train=True)
    
    valid_dataset = NoisyDataset(root_dir='',
                                 pkl_file=valid_file,
                                 transform=transform)
    
    print(f"Using train file: {train_file}")
    print(f"Using valid file: {valid_file}")

    print(f"Number of training samples: {len(train_dataset)}.")
    print(f"Number of testing samples: {len(valid_dataset)}")

    batch_size = cfg.training.batch_size
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, worker_init_fn=worker_init_reset_seed)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=2, worker_init_fn=worker_init_reset_seed)
    
    return train_loader, valid_loader
    
    
def build_buyer_dataloader(cfg, train_files, valid_file, weights=None, idxs=[], transform=None, use_test_set=False):

    root = cfg.dataset.path
    weight_type = cfg.dataset.weights

    train_datasets = []
    print("annotator idxs", idxs)
    print("annotator weights", weights)
    
    for train_file, weight  in zip(train_files, weights):
        print(train_file, weight)
        # using full annotator dataset for training buyer model
        train_dataset = NoisyDataset(root_dir=root, 
                                     pkl_file=train_file, 
                                     transform=transform,
                                     k=-1)
        train_datasets.append(train_dataset)
    
    print(f"Having {len(train_datasets)} annotators.")
    if use_test_set:
        valid_dataset = TestCIFAR10(root="./datasets/cifar_10/", 
                                    train=False,
                                    transform=transform,
                                    download=True)
    else:
        valid_dataset = NoisyDataset(root_dir='',
                                     pkl_file=valid_file,
                                     transform=transform)
    
    print(f"Using valid file: {valid_file}")
    print(f"Number of testing samples: {len(valid_dataset)}")

    batch_size = cfg.training.batch_size

    valid_sampler = torchdata.SequentialSampler(valid_dataset)
    
    if "uniform" in weight_type:
        print("Using uniform", weights) 
        buyer_train_dataset = WeightedBuyerDatasetV2(train_datasets, weights, transform)
        train_loader = DataLoader(buyer_train_dataset, batch_size=8, shuffle=True, num_workers=2)
    
    elif "loss" in weight_type:
        print("Using weight loss")
        buyer_train_dataset = WeightedBuyerDatasetV2(train_datasets, weights, transform)
        train_loader = DataLoader(buyer_train_dataset, 
                                  batch_size=8, 
                                  shuffle=True, 
                                  num_workers=2,
                                  worker_init_fn=worker_init_reset_seed)
    
    elif "sample" in weight_type:
        print("Using non uniform.")
        buyer_train_dataset = WeightedBuyerDataset(datasets=train_datasets,
                                                   transform=transform)
    
        train_loader = DataLoader(buyer_train_dataset, 
                                  batch_size=8, 
                                  sampler=train_sampler, 
                                  num_workers=2)
    
    valid_loader = DataLoader(valid_dataset, 
                              batch_size=8, 
                              shuffle=False, 
                              num_workers=2, 
                              worker_init_fn=worker_init_reset_seed)

    return train_loader, valid_loader








    






