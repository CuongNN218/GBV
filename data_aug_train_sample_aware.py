from train_single_model import train_model
from train_buyer_model import compute_features
from cub_aug_cls import Cub2011, _augmentation_space
from torch.utils.data import DataLoader
from dogs import StanfordDogs
import os
import torch.optim as optim
import math

from typing import Dict, List, Optional, Tuple
from torch import Tensor
from utils import accuracy
from tqdm import tqdm
import torch.nn.functional as F 
from torch.optim import lr_scheduler
from utils import default_setup, softmax, seed_all
from quinine import Quinfig
from torchvision import transforms
import torch 
import pandas as pd
import numpy as np
import argparse
from models import get_model
from metrics import LogME
from utils import LEEP, seed_all
from etran import LDA_Score, Energy_Score
from mmd_rbf import batched_rbf_mmd2
from cub_aug_cls import _apply_op, _augmentation_space, gaussian_noise_image
import random
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def _sra_apply_op(
    img: Tensor, op_name: str, magnitude: float, interpolation: InterpolationMode, fill: Optional[List[float]]
):
    prob = np.random.uniform(low=0.2, high=0.8) 
    p = np.random.uniform(low=0., high=1.)
    if p >= prob:
        return img

    if op_name == "ShearX":
        img = TF.affine(
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
        img = TF.affine(
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
        img = TF.affine(
            img,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        img = TF.affine(
            img,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "Rotate":
        img = TF.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
#        img = F.adjust_brightness(img, 1.0 + magnitude)
        img = TF.adjust_brightness(img, magnitude)
    elif op_name == "Color":
#        img = F.adjust_saturation(img, 1.0 + magnitude)
        img = TF.adjust_saturation(img, magnitude)
    elif op_name == "Contrast":
        img = TF.adjust_contrast(img, magnitude)
#        img = F.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
#        img = F.adjust_sharpness(img, 1.0 + magnitude)
        img = TF.adjust_sharpness(img, magnitude)
    elif op_name == "Posterize":
        img = TF.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = TF.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = TF.autocontrast(img)
    elif op_name == "Equalize":
        img = TF.equalize(img)
    elif op_name == "Invert":
        img = TF.invert(img)
    elif op_name == "GaussianBlur":
        img = TF.gaussian_blur(img, 11, magnitude)
    elif op_name == "GaussianNoise":
        img = gaussian_noise_image(img, 0.0, sigma=magnitude)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    return img


def _sample_aware_augment(img, target, mag=None, base_transform=None, num_bins=5, img_size=None, num_augs=2):
    op_meta = _augmentation_space(num_bins, img_size)
    op_names = random.sample(list(op_meta.keys()), k=num_augs)
    
    img = TF.to_pil_image(img)

    # always apply GaussianNoise after
    level_max = num_bins - 1
    if "GaussianNoise" in op_names[0]:
        op_names = [op_names[1], op_names[0]]
    
    for op_name in op_names:
        # get op_meta
        magnitudes, signed = op_meta[op_name]
        
        if mag is not None: 
            level = min(int(level_max * mag) + 1, level_max)       
            magnitude = (float(magnitudes[level].item()) if magnitudes.ndim > 0 else 0.0)
        else:
            magnitude = (
                float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item())
                if magnitudes.ndim > 0
                else 0.0
            )

        if signed and torch.randint(2, (1,)):
            magnitude *= -1.0
        
        if "GaussianNoise" in op_name:
            img = base_transform(img)               
            img = _sra_apply_op(img, op_name, magnitude, interpolation=InterpolationMode.NEAREST, fill=None)
        else: 
            img = _sra_apply_op(img, op_name, magnitude, interpolation=InterpolationMode.NEAREST, fill=None)
    
    if not "GaussianNoise" in op_names[1]:
        img = base_transform(img)

    return img, target


def train_one_epoch(model, 
                    trainloader, 
                    optimizer, 
                    epoch,
                    scheduler=None,
                    print_freq=100,
                    scaler=None, 
                    grad_clip=None,
                    device="cuda:0",
                    cfg=None,
                    logger=None,
                    num_bins=5,
                    img_size=None, 
                    base_transform=None,
                    num_augs=2):
    
    
    ''' training model for one epoch '''

    # model training
    model.train() # model in channel last mode
    loss_hist = []
    running_loss = 0.0
    running_corrects = 0.0 

    # no need to device since we use the prefetcher from timm
#    for i, (ori_inputs, inputs, labels, weights) in enumerate(trainloader):
    for i, (inputs, labels, weights) in enumerate(trainloader):        
#        ori_inputs = ori_inputs.to(device)
        inputs = inputs.to(device)
        labels = labels.to(device)
        weights = weights.to(device)

        # get number of batch size
        N = inputs.shape[0]

        # split input into 2 splits

        inputs_b1 = inputs[:N//2] # first half is augmented by rand aug
        inputs_b2 = inputs[N//2:] # second half is original augmented later
        
        labels_b1 = labels[:N//2]
        labels_b2 = labels[N//2:]
        
        weights_b1 = weights[:N//2]
        weights_b2 = weights[N//2:]
        
        # augmentation for the first half
        augs_b1 = []
        new_labels_b1 = []

        for input_b1, label_b1 in zip(inputs_b1, labels_b1):
            aug_b1, label_b1 = _sample_aware_augment(input_b1, label_b1, None, base_transform, num_bins, img_size, num_augs=num_augs)
            augs_b1.append(aug_b1)
            new_labels_b1.append(label_b1)

        augs_b1 = torch.stack(augs_b1, dim=0).to(device)
        new_labels_b1 = torch.stack(new_labels_b1, dim=0).to(device)

        optimizer.zero_grad()
        outputs, logits = model(augs_b1.contiguous(memory_format=torch.channels_last), labels_b1, weights=None)
        
        loss_value = outputs["loss_cls"].item()        
        losses = sum(loss for loss in outputs.values())
        
        loss_value = losses.item()        
        
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stop training.")
            sys.exit(1)
        
        running_loss += loss_value
        
        losses.backward()        
        optimizer.step()
        
        # calculate similarity for the second half

        with torch.no_grad():
            outputs, logits = model(inputs_b2.contiguous(memory_format=torch.channels_last), labels_b2, None)
        
        y = labels_b2
        ori = logits

        if y.shape[-1] != ori.shape[-1]:
            y = torch.zeros(ori.shape).to(y.device).scatter_(-1, y[..., None], 1.0)
            cos_sim = F.cosine_similarity(F.softmax(ori.detach(), dim=-1), y) ** (2 / np.log(y.shape[-1]))  # (N,)
        
        # adaptation for the second half
        optimizer.zero_grad()
        augs_b2 = []
        new_labels_b2 = []

        for input_b2, label_b2, mag in zip(inputs_b2, labels_b2, cos_sim):
            aug_b2, label_b2 = _sample_aware_augment(input_b2, label_b2, mag, base_transform, num_bins, img_size, num_augs=num_augs)
            augs_b2.append(aug_b2)
            new_labels_b2.append(label_b2)

        augs_b2 = torch.stack(augs_b2, dim=0).to(device)
        new_labels_b2 = torch.stack(new_labels_b2, dim=0).to(device)
        outputs, logits = model(augs_b2.contiguous(memory_format=torch.channels_last), new_labels_b2, None)

        losses = sum(loss for loss in outputs.values())
        loss_value = losses.item()

        running_loss += losses.item()

        losses.backward()
        optimizer.step()

        running_loss += loss_value
                
        if i > 0 and i % print_freq == 0:
            curr_lr = optimizer.param_groups[0]["lr"] 
            print(f"Epoch: {epoch} - Iter {i}: Loss = {running_loss/(print_freq * 8)}, Lr = {curr_lr}")
            running_loss = 0.0 # reset running_loss
    
    epoch_loss = running_loss / (i + 1) 
    curr_lr = optimizer.param_groups[0]["lr"]
    logger.info(f'Epoch: {epoch} - Current Lr: {curr_lr} - Train  Loss: {epoch_loss:.4f}')
    print(f"Epoch: {epoch} - Current Lr: {curr_lr} - Train  Loss: {epoch_loss:.4f}")
    return model 


@torch.no_grad()
def inference(model, testloader, device="cuda:0"):
    '''
        model inference on test set.
    '''
    # change the mode to training
    model.eval()
    all_ground_truths = []
    all_predictions = []
    print(f"Inferencing......")
    for i, (inputs, labels, _) in enumerate(tqdm(testloader)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs.contiguous(memory_format=torch.channels_last))
        _, preds = torch.max(outputs, 1)

        all_predictions.append(preds.cpu().numpy())
        all_ground_truths.append(labels.cpu().numpy())
    
    predictions = np.concatenate(all_predictions)
    ground_truths = np.concatenate(all_ground_truths)

    acc = accuracy(predictions, ground_truths)
    return acc 


def main(cfg, args, logger):
    
    img_size = (cfg.dataset.img_size, cfg.dataset.img_size)
    print(f"resizing img to {img_size}")
    dataset_name = cfg.dataset.name
    print("Training dataset: ", dataset_name)
    data_transform_0 = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(cfg.dataset.img_size)])
                                        
    data_transform_1 = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    base_transforms = (data_transform_0, data_transform_1)
    
    aug_space = _augmentation_space(args.num_bins, img_size)  
    print("Augmentation space", aug_space) 
    
    score_df = None 

    if 'cub' in dataset_name:
        print("Strategy: ", args.aug)
        train_dataset = Cub2011(cfg.dataset.path, 
                                    train=True, 
                                    weights_df=score_df,
                                    base_transform=base_transforms,
                                    img_size=img_size,
                                    strategy=args.aug)
        
        test_dataset = Cub2011(cfg.dataset.path, 
                               train=False,
                               weights_df=None,
                               img_size=img_size,
                               base_transform=base_transforms)
    
    elif 'dog' in dataset_name:
        train_dataset = StanfordDogs(cfg.dataset.path,
                                     weight_df=score_df,
                                     base_transform=base_transforms,
                                     img_size=img_size,
                                     num_bins=args.num_bins,
                                     strategy=args.aug,
                                     download=False)

        test_dataset = StanfordDogs(cfg.dataset.path,
                                    train=False,
                                    weight_df=None,
                                    img_size=img_size,
                                    num_bins=args.num_bins,
                                    strategy=args.aug,
                                    base_transform=base_transforms)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)
   
    # train model from scratch
    
    cfg.model.pretrained = False
    model = get_model(cfg).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    ce_loss = torch.nn.CrossEntropyLoss()
    
    save_path = os.path.join(cfg.dirs.root, cfg.exp_name, cfg.dirs.weights)
    
    # rewrite train_model
    
    best_acc = 0.0

    for epoch in range(100):
        _ = train_one_epoch(model, 
                            train_loader, 
                            optimizer, 
                            epoch, 
                            exp_lr_scheduler, 
                            base_transform=data_transform_1,
                            num_bins=args.num_bins,
                            cfg=cfg,
                            img_size=img_size,
                            logger=logger)
        
        if epoch >= 20:
            exp_lr_scheduler.step()

        if epoch % 2 == 0:
            curr_acc = inference(model, test_loader)
            print(f"Epoch {epoch}: {curr_acc}.")
            if curr_acc > best_acc:
                logger.info(f"Save best accurcary at {epoch}: {curr_acc} ")
                best_acc = curr_acc

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, help="path to buyer training config")
    parser.add_argument("--uniform", type=int, default=0, help="using uniform weights")
    parser.add_argument("--tau", type=float, default=-1, help="Tau for augment weights")
    parser.add_argument("--aug", type=str, help="augmentation strategies: aa, ra, ta, ent")
    parser.add_argument("--seed", type=int, help="seed for random")
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--measure", type=str, default="logme", help="measure for scoring")
    parser.add_argument("--weight_path", type=str, default="", help="pre load csv augmentation weights.")
    parser.add_argument("--num_augs", type=int, default=2)
    parser.add_argument("--weighted_loss", action="store_true", help="Using put weights on loss calculation")
    parser.add_argument("--num_bins", type=int, default=5, help="number of bins for augmentation space.")
    args = parser.parse_args()
    
    cfg = Quinfig(config_path=args.config_path)
    
    cfg.exp_name += f"_{args.exp_name}"

    if args.uniform == 1:
        cfg.exp_name += "_uniform"
    
    if "weight" in args.aug:
        cfg.exp_name += f"_{args.measure}"
    
    if args.tau > 0: 
        cfg.exp_name += f"_tau_{args.tau}"
    
    cfg.exp_name += f"_stra_{args.aug}"
    cfg.exp_name += f"_naug_{args.num_augs}"
    cfg.exp_name += f"_n_bin_{args.num_bins}"  
    
    # assign weighted loss
    
    cfg.training.weighted_loss = args.weighted_loss
    
    if args.weighted_loss:
        cfg.exp_name += "weighted_loss"

    print(f"Using the weighted loss: {cfg.training.weighted_loss}") 

    cfg.exp_name += f"_seed_{args.seed}"
    logger, _, _ = default_setup(cfg, args)    
    
    seed_all(args.seed) 
    main(cfg, args, logger)
