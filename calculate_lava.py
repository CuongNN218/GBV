import os
import torch
import re
import pickle as pkl 
import numpy as np 
import random 

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from LAVA.otdd.pytorch.distance_fast import DatasetDistance, FeatureCost
from models.resnet import resnet18

import time

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

class NoisyDataset(Dataset):
    def __init__(self, root_dir, pkl_file, transform=None, k=-1):
        
        self.file_dir = os.path.join(root_dir, pkl_file)
    
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
        self.targets = torch.tensor(self.labels)

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
         
        return img, label


model = resnet18(pretrained=True).to(device)

annotator_data_path = 'datasets/noisy_subsets/uniform_noise_annotators/training'
valid_path = 'datasets/noisy_subsets/uniform_noise_annotators/valid/buyer_validation_set.pkl'

train_files = [f for f in os.listdir(annotator_data_path) if os.path.isfile(os.path.join(annotator_data_path, f))]
print(train_files)
train_file_infos = []
scores = []


data_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

total_time = 0
for _ in range(5):


    for train_file in train_files:
        match = re.search(r"annotator_(\d+)_noise_(\d+.\d+)", train_file)
        info = {
                "anno_id": int(match.group(1)),
                "noise": float(match.group(2)),
                }
        train_data = NoisyDataset(annotator_data_path, 
                                  train_file, 
                                  data_transform, 
                                  k=100)
        train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=False)

        test_data = NoisyDataset('',
                                 valid_path,
                                 data_transform)
        test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
        start = time.time()
        dual_sol = get_OT_dual_sol(model, 
                                train_loader, 
                                test_loader, 
                                training_size=len(train_data),
                                device=device)
        lava_score = value(dual_sol, len(train_data))
        info["score"] = np.mean(lava_score)
        end = time.time() - start
        print("end time: ",  end)
        print(info)
        scores.append(info) 

#import pandas as pd
#
#score_df = pd.DataFrame(scores)
#score_df.to_csv(f"annotator_lava_score.csv")
