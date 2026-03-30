import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CUB200Annotator(Dataset):
    base_folder = 'CUB_200_2011/images'
    def __init__(self, 
                 root='/content', 
                 train=True, 
                 transform=None, 
                 train_file=None,
                 valid_file=None,
                 sample_per_class=-1,
                 loader=default_loader,
                 seed=0):
      
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train
        self.seed = seed

        if self.train:
            self.data = pd.read_csv(os.path.join(self.root, train_file))
        elif self.train is not True and valid_file is not None:
            self.data = pd.read_csv(valid_file)
        else:
            print("Using the full test set.")
            self._load_metadata()
        if self.train:
            status = "Training"
        else:
            status = "Testing"
        print(f"CUB-200-2011: {status} with {len(self.data)} samples")
                
        if sample_per_class > 0:
          self.sub_sample()


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
    
    def sub_sample(self):
        # Randomly select 3 rows per target
        df_sampled = self.data.groupby('target', group_keys=False).apply(lambda x: x.sample(n=3, random_state=self.seed))

        # Reset index
        df_sampled = df_sampled.reset_index(drop=True)
        self.data = df_sampled        

    # cái này cũng không cần đổi 
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        if self.train:
            return idx, img, target, 1.0
        else:
            return idx, img, target, 1.0

class CUB200Buyer(Dataset):
    base_folder = 'CUB_200_2011/images'
    def __init__(self,
                 root='/content',
                 train=True,
                 transform=None,
                 train_files=[],
                 weights = [],
                 valid_file='',
                 loader=default_loader):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        # load dataframes from
        dataframes = []
        for train_file, weight in zip(train_files, weights):
          dataframe = pd.read_csv(os.path.join(self.root, train_file))
          # adding weights to a new colums
          print(train_file, weights)
          dataframe['weight'] = weight
          dataframes.append(dataframe)

        if train:
          self.data = pd.concat(dataframes)
        else:
          self.data = pd.read_csv(self.valid_file)

    # cái này cũng không cần đổi
    def __len__(self):
        return len(self.data)

    # cái này không cần đổi
    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        weight = sample.weight
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        return idx, img, target, weight


def build_single_cub_dataloader(cfg, train_file, valid_file, transform=None):

    root = cfg.dataset.path
    k = cfg.dataset.subset_size
    
    train_dataset = CUB200Annotator(root=root,
                                    train=True,
                                    transform=transform,
                                    train_file=train_file,
                                    sample_per_class=k,
                                    loader=default_loader,
                                    seed=cfg.seed) 
    
    valid_dataset = CUB200Annotator(root=root,
                                    train=False,
                                    transform=transform,
                                    train_file=train_file,
                                    valid_file=valid_file,
                                    sample_per_class=-1,
                                    loader=default_loader)

    print(f"Using train file: {train_file}")
    print(f"Using valid file: {valid_file}")

    print(f"Number of training samples: {len(train_dataset)}.")
    print(f"Number of testing samples: {len(valid_dataset)}")

    batch_size = cfg.training.batch_size

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, valid_loader

def build_buyer_cub_dataloader(cfg, train_files, valid_file, weights, idxs=None, transform=None, use_test_set=False): 

    root = cfg.dataset.path
    train_dataset = CUB200Buyer(root,
                                train=True,
                                transform=transform,
                                train_files=train_files,
                                weights=weights,
                                valid_file=valid_file)
    if use_test_set:
        valid_dataset = CUB200Annotator(root, False, transform, None, None)
    else:
        valid_dataset = CUB200Annotator(root, False, transform, None, valid_file)

    
    print(f"Using train file: {train_files}")
    print(f"Using valid file: {valid_file}")

    print(f"Number of training samples: {len(train_dataset)}.")
    print(f"Number of testing samples: {len(valid_dataset)}")
    

    batch_size = cfg.training.batch_size 


    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size, False, num_workers=2)
    
    return train_loader, valid_loader
