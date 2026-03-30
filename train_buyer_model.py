import torch
import torch.nn as nn
import argparse
import pandas as pd
import os
import torch.optim as optim
import numpy as np
import time

from models import resnet18
from torch.utils.data import Dataset, DataLoader
from davinz.davinz import get_davinz 
from etran import LDA_Score, Energy_Score
from gbc import get_gbc_score
from pathlib import Path
from torchvision import transforms
from glob import glob
from quinine import Quinfig
from utils import softmax, default_setup, neg_softmax, LEEP, seed_all
from tqdm import tqdm
from datasets import build_buyer_dataloader
from train_single_model import train_model, compute_features
from models import get_model
from metrics import LogME 
from datasets import NoisyDataset
import torch.utils.data as torchdata
from torch.optim import lr_scheduler
from mmd_rbf import batched_rbf_mmd2, get_MMD_values_uneven, rbf_mmd2
from cub import CUB200Annotator, build_buyer_cub_dataloader
from torchvision.datasets.folder import default_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UnionRefererenceDataset(Dataset):
    def __init__(self, imgs, labels, transform):
        
        self.imgs = imgs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return self.imgs.shape[0]
    
    def __getitem__(self, idx):
        img, label = self.imgs[idx], self.labels[idx] 
        
        if self.transform:
            img = self.transform(img)
        return idx, img, label, 1.0

def get_buyer_valid_loader(valid_file, dataset_name, transform, cfg):
    if dataset_name == 'cifar10':    
        valid_dataset = NoisyDataset(root_dir='',
                                     pkl_file=valid_file,
                                     transform=transform)

        valid_loader = torchdata.DataLoader(valid_dataset,
                                            batch_size=8,
                                            shuffle=False,
                                            num_workers=2)
    elif dataset_name == 'cub200':
        root = cfg.dataset.path
        valid_dataset = CUB200Annotator(root=root,
                train=False,
                transform=transform,
                train_file=None,
                valid_file=valid_file,
                loader=default_loader)
        valid_loader = torchdata.DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=2)
    return valid_loader

def train_buyer_model(cfg, args, logger):
    
    # get all models from annotators
    
    dataset_name = cfg.dataset.name 

    valid_file = cfg.dataset.valid_path
    if dataset_name == 'cifar10':
        data_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    elif dataset_name == 'cub200':
        data_transform =  transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
    buyer_valid_loader = get_buyer_valid_loader(valid_file, dataset_name, data_transform, cfg)
    
    folder_path = Path(args.annotator_model_path)
    annotator_model_paths = list(folder_path.glob('*.pth'))
    
    print("annotator model path: ", annotator_model_paths)
    
    if "no_reference" in cfg.exp_name:
        print("Combining reference set")
        scores = []
        
        # create reference set

        annotator_data_path = cfg.dataset.path 
        train_files = [f for f in os.listdir(annotator_data_path) if os.path.isfile(os.path.join(annotator_data_path, f))]
        
        train_file_infos = []
        anno_img_subsets = []
        anno_label_subsets = []

        for train_file in train_files: 
            train_data = NoisyDataset(cfg.dataset.path, train_file, data_transform, k=100)
            anno_img_subsets.append(train_data.imgs)
            anno_label_subsets.append(train_data.labels)
        ref_imgs = np.concatenate(anno_img_subsets)
        ref_labels = np.concatenate(anno_label_subsets)
        print("number of images: ", np.shape(ref_imgs))
        print(f"number of labels: {np.shape(ref_labels)}.")
        # create dataloader for reference set
        ref_dataset = UnionRefererenceDataset(imgs=ref_imgs, 
                                              labels=ref_labels, 
                                              transform=data_transform)        
        print(f"Number of sample in reference set: {len(ref_dataset)}.") 
        ref_loader = DataLoader(ref_dataset, batch_size=16, shuffle=False)
        
        # create model
    
        for model_path in annotator_model_paths:
            model_path = os.path.normpath(model_path)
            import re
            match = re.search(r"anno_(\d+)", model_path)        
            anno_id = int(match.group(1))        
            anno_model = get_model(cfg)
            anno_model.load_state_dict(torch.load(model_path))
            
            anno_model = anno_model.to(device)
            anno_model = anno_model.to(memory_format=torch.channels_last)

            # calculate score
        
            if args.measure == 'leep':
                
                features, labels = compute_features(model=anno_model, 
                                                    loader=ref_loader,
                                                    device=device,
                                                    logits=True)
                anno_score = LEEP(features, labels)
                 
            elif args.measure == 'mmd':
                if "label_free" in cfg.exp_name:
                    features, _ = compute_features(model=anno_model, 
                                                   loader=buyer_valid_loader,
                                                   device=device,
                                                   logits=True)
                else:
                    features, _ = compute_features(model=anno_model, 
                                                        loader=ref_loader,
                                                        device=device,
                                                        logits=True)
                
                curr_anno = ""
                for train_file in train_files:
                    if f"annotator_{anno_id}" in train_file:
                        curr_anno = train_file
                        break
                
                anno_sample_data = NoisyDataset(cfg.dataset.path, curr_anno, data_transform, k=100)
                anno_sample_loader = DataLoader(anno_sample_data, batch_size=16, shuffle=False)
                
                anno_features, _ = compute_features(anno_model,
                                                    loader=anno_sample_loader,
                                                    device=device,
                                                    logits=True)
                
                features, anno_features = torch.Tensor(features), torch.Tensor(anno_features)
                anno_score = rbf_mmd2(features, anno_features)               
                print(anno_score)
                
            elif args.measure == 'etran':
                features, labels = compute_features(model=anno_model,
                                                    loader=ref_loader,
                                                    device=device,
                                                    logits=True)
                lda_score = LDA_Score(features, labels)
                energy_score = Energy_Score(features, 1.0, "tail")
                if "no_eda" in cfg.exp_name:
                    print("Using no eda")
                    anno_score = energy_score
                else:
                    anno_score = 0.85 * lda_score + 0.15 * energy_score
                
            scores.append({"anno_id": anno_id, "score": anno_score})
        
        print(scores)
        exit(0)
        
    scores = []
    for model_path in annotator_model_paths:
        
        model_path = os.path.normpath(model_path)
        print(model_path)

        import re
#        match = re.search(r"annotator_(\d+)_noise_(\d+.\d+)",model_path) 
        match = re.search(r"anno_(\d+)", model_path)        
#        match = re.search(r"annotator_(\d+)", model_path)        
        anno_id = int(match.group(1))        
#        noise_scale = float(match.group(2))

        print(f"Evaluating annotator {anno_id} using model at: {model_path}") 
        anno_model = get_model(cfg)
        anno_model.load_state_dict(torch.load(model_path))
        
        anno_model = anno_model.to(device)
        anno_model = anno_model.to(memory_format=torch.channels_last)
        
        if args.measure == 'davinz':
            continue
        
        elif args.measure == 'lava':
            continue
        
        elif args.measure == 'logme': 
            import time 
            start = time.time()
            features, labels = compute_features(model=anno_model, 
                                                loader=buyer_valid_loader,
                                                device=device,
                                                logits=False)
            feature_time = time.time() - start
            logme = LogME(regression=False)
            score_time = time.time() - feature_time
            total_time = time.time() - start
            print("total time: ", total_time)
            print("feature time: ", feature_time)
            anno_score = logme.fit(features, labels) 
            
        elif args.measure == 'mmd':
            print("rebuttal mmd")
            import time 
            start = time.time()
            features, labels = compute_features(model=anno_model, 
                                                loader=buyer_valid_loader,
                                                device=device,
                                                logits=True)
            
            feature_time = time.time() - start
            features, labels = torch.Tensor(features), torch.Tensor(labels).long()
            
            labels = torch.nn.functional.one_hot(labels, num_classes=cfg.dataset.num_classes).float() 
            
            print(features.shape, labels.shape)
            anno_score = batched_rbf_mmd2(features, labels).item()
            
            score_time = time.time() - feature_time
            total_time = time.time() - start
            print("total time: ", total_time)
            print("feature time: ", feature_time)
        
        elif args.measure == 'leep':
                                    
            import time 
            start = time.time()
            
            features, labels = compute_features(model=anno_model, 
                                                loader=buyer_valid_loader,
                                                device=device,
                                                logits=True)
            
            feature_time = time.time() - start
            anno_score = LEEP(features, labels)
                   
            score_time = time.time() - feature_time
            total_time = time.time() - start
            print("total time: ", total_time)
            print("feature time: ", feature_time)
        
        elif args.measure == 'etran':
            
            import time 
            start = time.time()

            features, labels = compute_features(model=anno_model,
                                                loader=buyer_valid_loader,
                                                device=device,
                                                logits=True)
            
            feature_time = time.time() - start
            
            lda_score = LDA_Score(features, labels)
            energy_score = Energy_Score(features, 1.0, "tail")
            if "no_eda" in cfg.exp_name:
                print("Using no eda")
                anno_score = energy_score
            else:
                anno_score = 0.85 * lda_score + 0.15 * energy_score
            
            score_time = time.time() - feature_time
            total_time = time.time() - start
            print("total time: ", total_time)
            print("feature time: ", feature_time)
        
        print({"anno_id": anno_id, "score": anno_score})
        logger.info(f"anno_id {anno_id} - score {anno_score}")   
        scores.append({"anno_id": anno_id, 
#                       "noise": noise_scale, 
                       "score": anno_score})
        del anno_model
    print(scores)
    exit(0)
    if len(scores) > 0:
        score_df = pd.DataFrame(scores)
    
    if args.measure == 'davinz':
        if dataset_name == 'cifar10':
            annotator_data_path = cfg.dataset.path 
            train_files = [f for f in os.listdir(annotator_data_path) if os.path.isfile(os.path.join(annotator_data_path, f))]
            train_file_infos = []
            for train_file in train_files:
                match = re.search(r"annotator_(\d+)_noise_(\d+.\d+)", train_file)
                info = {"anno_id": int(match.group(1)), "noise": float(match.group(2))}
                train_data = NoisyDataset(cfg.dataset.path, 
                                          train_file, 
                                          data_transform, 
                                          k=100)
                train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=False)
                train_images, train_labels, _ = next(iter(train_loader))
                train_images, train_labels = train_images.numpy(), train_labels.numpy()
                print("Train images davinz", train_images.shape)

                test_data = NoisyDataset('',
                                         cfg.dataset.valid_path,
                                         data_transform)
                test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

                test_images, test_labels, _  = next(iter(test_loader))
                test_images, test_labels = test_images.numpy(), test_labels.numpy()
                print("test_img davinz: ", test_images.shape)
                 
                model_paths = [os.path.normpath(model_path) for model_path in annotator_model_paths]
                
                curr_model_path = None
                
                for model_path in model_paths:
                    anno_id = info["anno_id"]
                    if f"anno_{anno_id}" in model_path:
                        curr_model_path = model_path
                print(curr_model_path)
                
                anno_model = get_model(cfg)
                anno_model.load_state_dict(torch.load(curr_model_path))
                anno_model = anno_model.to(device)
                anno_model.eval() 
                davinz_score, time = get_davinz(train_images, test_images, train_labels, model=anno_model)
                print("davinz time: ", time)
                print("davinz score: ", davinz_score[0]) 
                info["score"] = davinz_score[0]
                logger.info(info)
                scores.append(info)
                del anno_model
        
        elif 'cub' in dataset_name:
            
            root = cfg.dataset.path
            annotator_data_path = cfg.dataset.path 
            
            train_files = [f for f in os.listdir(annotator_data_path) if f.endswith(".csv") and "validation" not in f]
            train_file_infos = [] 
            for train_file in train_files:
                print(f"Calculating file: {train_file}")
                match = re.search(r"anno(\d+)", train_file)
                info = {"anno_id": int(match.group(1))}
                train_data = CUB200Annotator(root=root,
                                                train=True,
                                                transform=data_transform,
                                                train_file=train_file,
                                                sample_per_class=cfg.dataset.subset_size,
                                                loader=default_loader,
                                                seed=args.seed) 
                train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=False)
                train_images, train_labels, _ = next(iter(train_loader))
                train_images, train_labels = train_images.numpy(), train_labels.numpy()
                print(f"train images shape {train_images.shape}") 
                test_data = CUB200Annotator(root=root,
                                                train=False,
                                                transform= data_transform,
                                                train_file=train_file,
                                                valid_file=cfg.dataset.valid_path,
                                                sample_per_class=-1,
                                                loader=default_loader)

                test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

                test_images, test_labels, _  = next(iter(test_loader))
                test_images, test_labels = test_images.numpy(), test_labels.numpy()
                print("test_img davinz: ", test_images.shape)
                 
                model_paths = [os.path.normpath(model_path) for model_path in annotator_model_paths]
                
                curr_model_path = ""
                
                for model_path in model_paths:
                    anno_id = info["anno_id"]
                    if f"anno_{anno_id}" in model_path:
                        curr_model_path = model_path
                        print("calculating this model: ", curr_model_path)
                
                anno_model = get_model(cfg)
                anno_model.load_state_dict(torch.load(curr_model_path))
                anno_model = anno_model.to(device)
                anno_model.eval() 
                davinz_score, time = get_davinz(train_images, test_images, train_labels, 
                                                model=anno_model, dims=(3,224,224))
                print("davinz score: ", davinz_score[0]) 
                info["score"] = davinz_score[0]
                logger.info(info)
                scores.append(info)
                del anno_model
#    exit(0) 
    if len(scores) > 0:
        score_df = pd.DataFrame(scores)
    if args.measure == 'lava':
        score_df = pd.read_csv("annotator_lava_score.csv")
    

    if args.adaptive:
        n = score_df.shape[0]
        tau = np.log2(n)
        mean = score_df["score"].mean()  
        std = score_df["score"].std()
        score_df["new_score"] = (score_df["score"] - mean) / std 
        print(f"Using adaptive tau: tau = {tau}")
        score_df["probability"] = softmax(score_df["new_score"].values, tau=tau)
    else:
        score_df["probability"] = softmax(score_df["score"].values, tau=args.tau)
#    score_df["probability"] = softmax(score_df["score"].values, tau=args.tau) 
    print(score_df) 
    
    print("Annotator scores:")
    print(score_df)
    
    # create data loader for buyer
    
    ids = score_df["anno_id"].tolist()
    
    if "uniform" in cfg.dataset.weights:
        print("Using uniform weights")
        prob = 1.0 / len(ids)
        weights = [prob for _ in range(len(ids))]
#    elif "test" in cfg.dataset.weights:
#        print("Testing probability")
#        noise = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
#        weights = neg_softmax(noise) 
#    else:
#        weights = score_df["probability"].tolist()
#    
#    print(f"ids: {ids} with weights {weights}")
    
    annotator_data_path = cfg.dataset.path 
    if dataset_name == 'cifar10':
        train_files = [f for f in os.listdir(annotator_data_path) if os.path.isfile(os.path.join(annotator_data_path, f))]
        train_file_infos = []
        for train_file in train_files:
            print("Train file aistats: ", train_file)
            match = re.search(r"annotator_(\d+)_noise_(\d+.\d+)", train_file)
            info = {"anno_id": int(match.group(1)),
                    "noise": float(match.group(2)),
                    "data_path": train_file}
            train_file_infos.append(info)
            train_file_df = pd.DataFrame(train_file_infos)
#            print("Train file df: ", train_file_df)
#            print("Score df: ", score_df)
#            weight_path_df = pd.merge(train_file_df, score_df, on=["anno_id", "noise"], how='inner') 

            weight_path_df = pd.merge(train_file_df, score_df, on=["anno_id"], how='inner') 
            print(weight_path_df)
            train_files = weight_path_df["data_path"].tolist()
            weights = weight_path_df["probability"].tolist()
    else:
        train_files = [f for f in os.listdir(annotator_data_path) if f.endswith(".csv") and "validation" not in f]
        train_file_infos = []
        for train_file in train_files:    
            match = re.search(r"anno(\d+)", train_file)
            info = {"anno_id": int(match.group(1)),
                    "data_path": train_file}
            train_file_infos.append(info)
            train_file_df = pd.DataFrame(train_file_infos)
            print("Train file df")
            print(train_file_df)
            weight_path_df = pd.merge(train_file_df, score_df, on=["anno_id"], how='inner') 
            print(weight_path_df)
            
        train_files = weight_path_df["data_path"].tolist()
        weights = weight_path_df["probability"].tolist()
        for train_file_i, weight_i in zip(train_files, weights):
            print("corresponding weights: ", train_file_i, weight_i)
    
    print("Annotator files: ", train_files)
    
    if dataset_name == 'cifar10': 
        buyer_train_loader, buyer_valid_loader = build_buyer_dataloader(cfg, 
                                                                        train_files, 
                                                                        valid_file, 
                                                                        weights=weights, 
                                                                        idxs=ids, 
                                                                        transform=data_transform,
                                                                        use_test_set=True)
    
    elif dataset_name == 'cub200':
        buyer_train_loader, buyer_valid_loader = build_buyer_cub_dataloader(cfg, 
                                                                            train_files, 
                                                                            valid_file,
                                                                            weights,
                                                                            ids,
                                                                            transform=data_transform,
                                                                            use_test_set=True)
    # training buyer model
    buyer_model = get_model(cfg).to(device)
    buyer_model = buyer_model.to(memory_format=torch.channels_last) 
    # build optimizer
    

    optimizer = torch.optim.Adam(buyer_model.parameters(), lr=1e-4, betas=(0.9, 0.999))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    ce_loss = nn.CrossEntropyLoss()
    
    buyer_save_path = os.path.join(cfg.dirs.root, cfg.exp_name, cfg.dirs.weights)
    print(f"Model will be saved at {buyer_save_path}")
    best_val_acc_buyer = train_model(model=buyer_model,
                                     trainloader=buyer_train_loader,
                                     testloader=buyer_valid_loader,
                                     epochs=cfg.training.epochs,
                                     optimizer=optimizer,
                                     criterion=ce_loss,
                                     scheduler=scheduler,
                                     ckpt_name="buyer",
                                     save_path=buyer_save_path,
                                     cfg=cfg,
                                     logger=logger,
                                     print_freq=300,
                                     save_ckpt=False)
    
    return best_val_acc_buyer


if __name__ == "__main__":

    # define arguments for training buyer's model
    parser = argparse.ArgumentParser(description="Training buyer arguments")
    
    parser.add_argument("-c", "--config_path", type=str, help="path to buyer training config")
    parser.add_argument("--annotator_model_path", type=str, help="path to directory containing annotator models.")
    parser.add_argument("--measure", type=str, default="logme", help="data valuation measurement")
    parser.add_argument("--tau", type=float, default=0.1, help="tau for softmax")    
    parser.add_argument("--no_pretrained", action="store_true", help="using pretrained for buyer")
    parser.add_argument("--seed", type=int, default=0, help="seed for random." )
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--adaptive", action='store_true', default="using adaptive tau")
    
    args = parser.parse_args()
    
    cfg = Quinfig(config_path=args.config_path)
    
    if args.no_pretrained:
        cfg.model.pretrained = False
    cfg.exp_name += f"_{args.exp_name}"
    if args.adaptive:
        cfg.exp_name = cfg.exp_name + f"{args.measure}" + f"_adaptive_tau" + f"_seed_{args.seed}"
    else:
        cfg.exp_name = cfg.exp_name + f"{args.measure}" + f"_tau_{args.tau}" + f"_seed_{args.seed}"
    cfg.exp_name +=  "" if not args.no_pretrained else f"_no_pre_v2"
    
    cfg.exp_name += "uniform" if "uniform" in cfg.dataset.weights else ""
    logger, _, _ = default_setup(cfg, args)    
    seed_all(args.seed)

    train_buyer_model(cfg, args, logger)

