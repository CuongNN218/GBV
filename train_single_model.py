import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import time
import os
import glob
import math
import sys
import itertools
import torch.nn.functional as F
 
from datasets import NoisyDataset
from PIL import Image
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import accuracy
#from data_aug_train import calculate_weights


def train_one_epoch(model, 
                    trainloader, 
                    optimizer, 
                    epoch,
                    criterion,
                    scheduler=None,
                    print_freq=100,
                    scaler=None, 
                    grad_clip=None,
                    device="cuda:0",
                    cfg=None,
                    logger=None,
                    args=None,
                    train_dataset=None):
    
    
    '''
        training model for one epoch 
    '''

    # model training
    model.train() # model in channel last mode
    loss_hist = []
    running_loss = 0.0
    running_corrects = 0.0 

    # no need to device since we use the prefetcher from timm
    for i, (idx, inputs, labels, weights) in enumerate(trainloader):
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        weights = weights.to(device)
        
        optimizer.zero_grad()
        if cfg.training.weighted_loss:
            outputs, logits = model(inputs.contiguous(memory_format=torch.channels_last), labels, weights=weights)
        else:
            outputs, logits = model(inputs.contiguous(memory_format=torch.channels_last), labels, None)
        
        if getattr(args, 'aug', None) is not None and  args.aug == 'ent':
            probability = F.softmax(logits,dim=1)
            entropy = -torch.sum(probability * torch.log(probability + 1e-8), dim=1)
            num_classes = 200 if cfg.dataset.name == 'cub' else 120
            magnitude = entropy / np.log(num_classes)
            train_dataset._set_magnitude(idx, 1 - magnitude.detach().cpu())  

        loss_value = outputs["loss_cls"].item()        
        losses = sum(loss for loss in outputs.values())
        
        loss_value = losses.item()        
        
        # loss is nan, then stop training
        
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stop training.")
            sys.exit(1)
        
        running_loss += loss_value
        
        if i > 0 and i % print_freq == 0:
            curr_lr = optimizer.param_groups[0]["lr"] 
            print(f"Epoch: {epoch} - Iter {i}: Loss = {running_loss/(print_freq * 8)}, Lr = {curr_lr}")
            running_loss = 0.0 # reset running_loss
        
        losses.backward()        
        optimizer.step()
        running_loss += losses.item() * inputs.size(0)
        
        if grad_clip > 0.0:
            nn.utils.clip_grad_norm(model.parameters, grad_clip)
    epoch_loss = running_loss / 5000
    
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

def train_model(model, 
                trainloader, 
                testloader, 
                epochs, 
                optimizer, 
                criterion,
                evaluation_freq=1,
                print_freq=100,
                scheduler=None,
                save_path=None,
                grad_clip=0.0,
                device="cuda:0",
                logger=None,
                cfg=None, 
                save_ckpt=True,
                ckpt_name="",
                update=False,
                args=None,
                train_dataset=None):
    best_model = None 
    best_acc_eval = 0.0
    for epoch in range(epochs):
        # do training for the first epoch
#        print("MAGNITUDE", trainloader.dataset.MAGNITUDE[:20])
        updated_model = train_one_epoch(model=model, 
                                    trainloader=trainloader, 
                                    optimizer=optimizer, 
                                    epoch=epoch,
                                    criterion=criterion,
                                    scheduler=scheduler, 
                                    grad_clip=grad_clip,
                                    device=device,
                                    cfg=cfg,
                                    print_freq=print_freq,
                                    logger=logger,
                                    args=args,
                                    train_dataset=train_dataset)
        # if need to update weights of the augmentation
#        if update:
#            # do recalculate weights
#        
#            new_weight_df = calculate_weights(updated_models, args=args, cfg=cfg, img_size=224)       
#            trainloader.dataset.weight_df = new_weight_df
#        if cfg.opt.name == "step":
#        if epoch > 19:
        scheduler.step()

        end_loss_value = 0.0
        
        # evaluation
        if (epoch + 1) % evaluation_freq == 0:
            curr_acc = inference(model, testloader)

            logger.info(f"Accuracy at epoch {epoch} : {curr_acc}.")
            
            if curr_acc > best_acc_eval:
                annotator_save_path = os.path.join(save_path, f"best_model_{ckpt_name}.pth")
                best_model = updated_model
                if save_ckpt:
                    torch.save(model.state_dict(), annotator_save_path)
                    print(f"Save best model of annotator {ckpt_name} with acc {curr_acc:.4f}.")
                logger.info(f"Save best model of annotator {ckpt_name} with acc {curr_acc:.4f}.")
                best_acc_eval = curr_acc
                print(f"Best acc: {best_acc_eval:.4f}")
    return best_model, best_acc_eval

@torch.no_grad()
def compute_features(model, loader, device, use_timm=False, logits=False):
    model.eval()
    features = []
    targets = []

    for (_, inputs, labels, _) in tqdm(loader):
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        if not logits:
            if not use_timm:
                output_features = model.forward_features(inputs.contiguous(memory_format=torch.channels_last))
            else:
                output = model.forward_features(inputs.contiguous(memory_format=torch.channels_last))
                output_features = model.forward_head(output, pre_logits=True)
        else:
            output_features = model(inputs.contiguous(memory_format=torch.channels_last))
            output_features = torch.softmax(output_features, dim=1)
        features.append(output_features.detach().cpu().numpy())
        targets.append(labels.detach().cpu().numpy())
    
    features = np.concatenate(features, axis=0)
    targets = np.concatenate(targets, axis=0)
    return features, targets
