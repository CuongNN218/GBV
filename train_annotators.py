import torch
import torch.nn as nn
import numpy as np
import os 
import sys
import torch.optim as optim
import torchvision
import math
import argparse
import torch.nn as nn
import re

from utils import default_setup, seed_all
from models import get_model
from quinine import Quinfig
from datasets import build_single_dataloader
from torchvision import models, transforms
from train_single_model import train_model, inference
from bisect import bisect_right
from torch.optim.lr_scheduler import LambdaLR, StepLR
from utils import get_constant_schedule, get_linear_schedule_with_warmup
from torch.optim import lr_scheduler
from cub import build_single_cub_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_annotators(cfg, args, logger):
    
    annotator_data_path = args.annotator_train_file_path
    buyer_valid_file = cfg.dataset.valid_path
    epochs = cfg.training.epochs
    print("Train file path: ", annotator_data_path) 
    data_name = cfg.dataset.name
    # get all annotator data files
    if data_name == 'cifar10':
        anno_data_files = [f for f in os.listdir(annotator_data_path) if os.path.isfile(os.path.join(annotator_data_path, f))]
    else:
        anno_data_files = [f for f in os.listdir(annotator_data_path) if f.endswith(".csv") and "validation" not in f]
    print("Loading data from the following files:", anno_data_files) 
    annotator_loaders = []
    
    # build datasets for all annotators
    if data_name == 'cifar10':
        data_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])    
    elif data_name == "cub200": 
        data_transform = transforms.Compose([transforms.Resize(256),
                                            transforms.RandomResizedCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])     
    for train_file in anno_data_files:
        print("Using file", train_file)
        if cfg.dataset.name == 'cifar10':
            train_loader, valid_loader = build_single_dataloader(cfg,
                                                                 train_file, 
                                                                 buyer_valid_file, 
                                                                 transform=data_transform)
        elif cfg.dataset.name == 'cub200':
            train_loader, valid_loader = build_single_cub_dataloader(cfg, 
                                                                     train_file,
                                                                     buyer_valid_file,
                                                                     data_transform) 
        model = get_model(cfg)        
        model = model.to(device)
        model = model.to(memory_format=torch.channels_last)   
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        criterion = nn.CrossEntropyLoss()   

        # print data information
        anno_file_name = train_file
        print(f"Anno file name: {anno_file_name}")
        if data_name == 'cifar10':
            annotator_number = anno_file_name.split("_")[1]
            post_fix = anno_file_name.split("_")[3]
            noise_scale = ".".join(post_fix.strip().split('.')[:2])
            if "continual" in cfg.exp_name:
                match = re.search(r"batch_(\d+).*subset_(\d+).*noise_(\d+\.\d+)", train_file)
                info = {
                    "batch": int(match.group(1)),
                    "subset": int(match.group(2)),
                    "noise": float(match.group(3))
                }
                print(f"info: {info}")

                items = []
                for key, value in info.items():
                    items.append(str(key))
                    items.append(str(value))
                ckpt_name = "_".join(items)
                ckpt_name = "_".join([args.exp_name, ckpt_name])
                print(f"checkpoint name: {ckpt_name}")
            else:
                # use match to get infos
                print(train_file)
                match = re.search(r"annotator_(\d+)_noise_(\d+.\d+)", train_file)
                info = {"annotator":int(match.group(1)),
                        "noise": float(match.group(2))}
                items = []
                for k,v in info.items():
                    items.append(str(k))
                    items.append(str(v))
                ckpt_name = "_".join(items)
                ckpt_name += f"_size_{args.k}"
                ckpt_name = "_".join([args.exp_name, ckpt_name])

                print(f"checkpoint name: {ckpt_name}")

        elif data_name == 'cub200':
            
            cub_noises = {'0': 0.0,
                      '1': 0.33,
                      '2': 0.66}

            annotator_number = anno_file_name.split("_")[0][-1]
            noise_scale = cub_noises[annotator_number]
            print(f"Current annotator {annotator_number}.")
            ckpt_name = f"anno_{annotator_number}.pth"
        save_path = os.path.join(cfg.dirs.root, cfg.exp_name, cfg.dirs.weights)
        print(f"Saving at: {save_path}")
        best_acc_val = train_model(model=model,
                                   trainloader=train_loader,
                                   testloader=valid_loader,
                                   epochs=epochs,
                                   optimizer=optimizer,
                                   criterion=criterion,
                                   ckpt_name=ckpt_name,
                                   evaluation_freq=2,
                                   print_freq=500,
                                   scheduler=scheduler,
                                   save_path=save_path,
                                   cfg=cfg,
                                   device=device,
                                   grad_clip=0.0,
                                   logger=logger,
                                   save_ckpt=True)
        
        print(f"Training current annotator {annotator_number} with noise {noise_scale}, best accuracy = {best_acc_val}")
        
        del model, optimizer, scheduler
    
if __name__ == '__main__':
    
    # load configs
    parser=argparse.ArgumentParser(description="sample argument parser")
    
    parser.add_argument("--config", type=str, help="path to config file")
    parser.add_argument("--seed", type=int, default=0, help="seed for reproducing results")
    parser.add_argument("--annotator_train_file_path", type=str, help="path to config file")
    parser.add_argument("--exp_name", type=str, help="name of exp")
    parser.add_argument("--k", type=int, default=-1, help="number of samples use for training annotators")
    args=parser.parse_args()
    
    cfg = Quinfig(args.config) 
    if args.k == -1:
        cfg.exp_name += f"_seed_{args.seed}" 
    else:
        cfg.exp_name += f"_seed_{args.seed}_size_{args.k}" 
    # assign for selecting subset
    cfg.dataset.subset_size = args.k

    seed_all(args.seed)   
    
    print("Using configs:")
    print(cfg)
    
    logger, _, _ = default_setup(cfg, args)
    train_annotators(cfg, args, logger)
    
