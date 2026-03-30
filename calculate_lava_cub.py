import os
import torch
import re
import pickle as pkl 
import numpy as np 
import random 
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from LAVA.otdd.pytorch.distance_fast import DatasetDistance, FeatureCost
from models.resnet import resnet18
from torchvision.datasets.folder import default_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(0)

#device = 'cpu'

def get_OT_dual_sol(feature_extractor, 
                    trainloader, 
                    testloader, 
                    training_size=10000, 
                    p=2, 
                    dims=(3, 32, 32), 
                    device='cuda'):
    
    embedder = feature_extractor.to(device)
    embedder.fc = torch.nn.Identity()
    for p in embedder.parameters():
        p.requires_grad = False

    # Here we use same embedder for both datasets
    feature_cost = FeatureCost(src_embedding = embedder,
                               src_dim = dims, #MNIST: (1, 32, 32), CIFAR10: (3, 32, 32)
                               tgt_embedding = embedder,
                               tgt_dim = dims, #MNIST: (1, 32, 32), CIFAR10: (3, 32, 32)
                               p = 2,
                               device=device)

    dist = DatasetDistance(trainloader, testloader,
                           inner_ot_method = 'exact',
                           debiased_loss = True,
                           feature_cost = feature_cost,
                           λ_x=1.0, λ_y=1.0,
                           sqrt_method = 'spectral',
                           sqrt_niters=10,
                           precision='single',
                           p = 2, entreg = 1e-1,
                           device=device)

    dual_sol = dist.dual_sol(maxsamples = training_size, return_coupling = True)

    for i in range(len(dual_sol)):
        dual_sol[i] = dual_sol[i].to('cpu')

    return dual_sol


def value(dual_sol, training_size):
    dualsol = dual_sol

    f1k = np.array(dual_sol[0].squeeze())

    trainGradient = [0]*training_size
    trainGradient = (1+1/(training_size-1))*f1k - sum(f1k)/(training_size-1)

    return list(trainGradient)


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
        
        self.targets = torch.tensor(self.data.target.tolist()) - 1 
        
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
        return img, target

model = resnet18(pretrained=True).to(device)

annotator_data_path = 'datasets/cub_dataset'
valid_path = 'datasets/cub_dataset/validation_df.csv'
root = 'datasets/cub_dataset' 

train_files = [f for f in os.listdir(annotator_data_path) if f.endswith(".csv") and "validation" not in f]
print(train_files)
train_file_infos = []
scores = []


data_transform =  transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])



for train_file in train_files:
    print(train_file)
    match = re.search(r"anno(\d+)", train_file)
    info = {"anno_id": int(match.group(1))}
    
    train_data = CUB200Annotator(root=root,
                                train=True,
                                transform=data_transform,
                                train_file=train_file,
                                sample_per_class=-1,
                                loader=default_loader,
                                seed=0) 
    train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=False)
    
    test_data = CUB200Annotator(root=root,
                                train=False,
                                transform= data_transform,
                                train_file=train_file,
                                valid_file=valid_path,
                                sample_per_class=-1,
                                loader=default_loader)

    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
    
    dual_sol = get_OT_dual_sol(model, 
                            train_loader, 
                            test_loader, 
                            training_size=len(train_data),
                            device=device,
                            dims=(3,224,224))
    lava_score = value(dual_sol, len(train_data))
    info["score"] = np.mean(lava_score)
    
    print(info)
    scores.append(info) 

score_df = pd.DataFrame(scores)
score_df.to_csv(f"annotator_cub_lava_score.csv")
