import torch
import torch.nn as nn
import argparse
import pandas as pd
import os
import torch.optim as optim
import numpy as np
import re

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
from mmd_rbf import batched_rbf_mmd2
from cub import CUB200Annotator, build_buyer_cub_dataloader
from torchvision.datasets.folder import default_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def get_batch_info(files, type="model"):
    
    infos = []
    if type == "model":
        for filename in files:
            match = re.search(r"batch_(\d+).*subset_(\d+).*noise_(\d+\.\d+)", filename)
            info = {"batch": int(match.group(1)),
                    "subset": int(match.group(2)),
                    "noise": float(match.group(3)),
                    "model_path": filename}
            print(info) 
            infos.append(info)

    elif type == "data":
        for filename in files:
            print(filename)
            match = re.search(r"batch_(\d+).*subset_(\d+).*noise_(\d+\.\d+)", filename)
            info = {"batch": int(match.group(1)),
                    "subset": int(match.group(2)),
                    "noise": float(match.group(3)),
                    "data_path": filename}
            print(info)
            infos.append(info)

    info_df = pd.DataFrame(infos)

    return info_df


def continual_train_buyer_model(cfg, args, logger):
    
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
    annotator_model_paths = list(map(str, folder_path.glob('*.pth')))
    
    model_info_df = get_batch_info(annotator_model_paths, "model")
    annotator_data_path = cfg.dataset.path 
    if dataset_name == 'cifar10':
        train_files = [f for f in os.listdir(annotator_data_path) if os.path.isfile(os.path.join(annotator_data_path, f))]
    else:
        train_files = [f for f in os.listdir(annotator_data_path) if f.endswith(".csv") and "validation" not in f]
    
    data_info_df = get_batch_info(train_files, type="data")
    model_data_df = pd.merge(model_info_df, data_info_df, on=['batch', 'subset', 'noise'], how='inner')
    
    model_data_df = model_data_df.sort_values(by=['batch', 'subset']).reset_index(drop=True)
    print("Full model data df")
    print(model_data_df)

    if len(args.score_path) > 0:
        score_df = pd.read_csv(args.score_path)
    else:
        scores = []
        for _, row in model_data_df.iterrows():

            model_path = os.path.normpath(row["model_path"])
            batch, subset, noise = row["batch"], row["subset"], row["noise"]
            file_name = os.path.split(model_path)[-1].split(".")[0]

            print(f"Evaluating batch {batch} annotator {subset} using model at: {model_path}") 
            
            anno_model = get_model(cfg)
            anno_model.load_state_dict(torch.load(model_path))
            
            anno_model = anno_model.to(device)
            anno_model = anno_model.to(memory_format=torch.channels_last)
            
            if args.measure == 'logme': 
                features, labels = compute_features(model=anno_model, 
                                                    loader=buyer_valid_loader,
                                                    device=device,
                                                    logits=False)
                logme = LogME(regression=False)
                anno_score = logme.fit(features, labels) 
            
            elif args.measure == 'mmd':
                
                features, labels = compute_features(model=anno_model, 
                                                    loader=buyer_valid_loader,
                                                    device=device,
                                                    logits=True)
                features, labels = torch.Tensor(features), torch.Tensor(labels).long()
                labels = torch.nn.functional.one_hot(labels, num_classes=cfg.dataset.num_classes).float() 
                anno_score = batched_rbf_mmd2(features, labels)
            
            elif args.measure == 'leep':
                            
                features, labels = compute_features(model=anno_model, 
                                                    loader=buyer_valid_loader,
                                                    device=device,
                                                    logits=True)
                anno_score = LEEP(features, labels)

            score_dict = {"batch": batch, "subset": subset, "noise": noise, "score": anno_score}

            scores.append(score_dict)
            
            del anno_model

        score_df = pd.DataFrame(scores)
        score_file_name = f"results_nips/scores/continual_score_100_{args.measure}.csv"
        print(f"Saving score at {score_file_name}")
        score_df.to_csv(score_file_name) 
    
    score_df = pd.merge(model_data_df, score_df,on=["batch", "subset", "noise"], how="inner")
    
    print("Annotator scores:", score_df)
    # training buyer model
    
    #setting the first prior
    batch_probabilities = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    unique_batches = score_df['batch'].unique()
    curr_batch_data = []

    strategy = args.strategy
    buyer_model = None 
    print(f"Using current strategy {strategy}") 

    for idx, batch_value in enumerate(unique_batches):
        
        print(f"Calculating batch {batch_value}.")
        
        batch_df = score_df[score_df['batch'] == batch_value]
        print(batch_df)
        data_paths = batch_df['data_path'].tolist()
        curr_batch_data = data_paths
#        curr_batch_data.extend(data_paths)
#        print("Data paths: ", data_paths)
#        print("Current batch data: ", curr_batch_data) 
        weights_curr_batch = np.array(batch_df["score"].tolist())
        ids = batch_df["subset"].tolist()
        
        curr_probability = softmax(weights_curr_batch, tau=args.tau)
        
        if "bayes" in strategy:
            actual_probability = curr_probability * batch_probabilities 
            # renormalise it
            curr_weights  = actual_probability / np.sum(actual_probability + 1e-7)
            batch_probabilities = curr_weights
        
        elif "last" in strategy:
            # using the weights of current step
            curr_weights = curr_probability / np.sum(curr_probability + 1e-7)
        
        elif "first" in strategy:
            # using the weights of first step
            if idx > 0:
                curr_weights = all_probabilities[0]
            else:
                all_probabilities = []
                curr_weights = curr_probability / np.sum(curr_probability + 1e-7)
                all_probabilities.append(curr_weights)
                print(all_probabilities)
        
        elif "avg" in strategy:
            # average probability from all steps. 
            curr_probability = curr_probability / np.sum(curr_probability + 1e-7)
            if idx == 0:
                all_probabilities = [curr_probability]
            else:
                all_probabilities.append(curr_probability)
            
            print(all_probabilities)
            curr_weights = np.mean(np.array(all_probabilities), axis=0)
        else:
            print(f"Do not support this strategy {strategy}")
            exit(0)
        
        if not isinstance(curr_weights, list):
            curr_weights = curr_weights.tolist() 
        
        print("curr_weights: ", curr_weights)
#        continue

        curr_weights = curr_weights * (len(curr_batch_data) // len(curr_weights))
        for path, weight in zip(curr_batch_data, curr_weights):
            logger.info(f"Strategy: {strategy} - data at {path} - weight {weight} ")
            
        buyer_train_loader, buyer_valid_loader = build_buyer_dataloader(cfg, curr_batch_data, valid_file, 
                                                                        weights=curr_weights, idxs=ids, 
                                                                        transform=data_transform, use_test_set=True)
        
        # training models by data batches.
        if buyer_model is None:
            buyer_model = get_model(cfg).to(device)
            buyer_model = buyer_model.to(memory_format=torch.channels_last)
#        optimizer = torch.optim.SGD(buyer_model.parameters(),lr=0.01)     
        optimizer = torch.optim.Adam(buyer_model.parameters(), lr=1e-4, betas=(0.9, 0.999))
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        ce_loss = nn.CrossEntropyLoss()
        
        buyer_save_path = os.path.join(cfg.dirs.root, cfg.exp_name, cfg.dirs.weights)
        print(f"Model will be saved at {buyer_save_path}")
        ckpt_name = f"_batch_{batch_value}"
        print(f"Checkpoint name: {ckpt_name}")
        
        buyer_model, _ = train_model(model=buyer_model, trainloader=buyer_train_loader, testloader=buyer_valid_loader, 
                    epochs=cfg.training.epochs,optimizer=optimizer, criterion=ce_loss, ckpt_name=ckpt_name,
                    scheduler=scheduler, save_path=buyer_save_path, cfg=cfg,logger=logger, 
                    print_freq=300, save_ckpt=False)


if __name__ == "__main__":

    # define arguments for training buyer's model
    parser = argparse.ArgumentParser(description="Training buyer arguments")
    
    parser.add_argument("-c", "--config_path", type=str, help="path to buyer training config")
    parser.add_argument("--annotator_model_path", type=str, help="path to directory containing annotator models.")
    parser.add_argument("--score_path", type=str, default="", help="path to score csv.")
    parser.add_argument("--measure", type=str, default="logme", help="data valuation measurement")
    parser.add_argument("--tau", type=float, default=1.0, help="tau for softmax")    
    parser.add_argument("--no_pretrained", action="store_true", help="using pretrained for buyer")
    parser.add_argument("--seed", type=int, default=0, help="seed for random." )
    parser.add_argument("--strategy", type=str, default="bayes", help="strategy for weight calculation at the next batches")
    parser.add_argument("--exp_name", type=str, default="")
        
    args = parser.parse_args()
    
    cfg = Quinfig(config_path=args.config_path)
    cfg.training.tau = args.tau
    
    if args.no_pretrained:
        cfg.model.pretrained = False
    
    cfg.exp_name += f"_strategy_{args.strategy}"
    print(len(args.exp_name))
    cfg.exp_name += "" if len(args.exp_name) ==  0 else f"_{args.exp_name}"   
    if "bayes" in args.strategy:
        cfg.exp_name += f"_{args.measure}" + f"_tau_{args.tau}" 
    cfg.exp_name += f"_seed_{args.seed}"
    cfg.exp_name +=  "" if not args.no_pretrained else f"_no_pre_v2"
    
    cfg.exp_name += "uniform" if "uniform" in cfg.dataset.weights else ""
    logger, _, _ = default_setup(cfg, args)    
    seed_all(args.seed)

    continual_train_buyer_model(cfg, args, logger)

