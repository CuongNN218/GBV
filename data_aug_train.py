from train_single_model import train_model
from train_buyer_model import compute_features
from cub_aug_cls import Cub2011, _augmentation_space
from torch.utils.data import DataLoader
from dogs import StanfordDogs
import os
import torch.optim as optim

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_top_k_and_single(df, k):
    grouped = df.groupby("op_name")
    result = grouped.apply(lambda x: x if len(x) == 1 else x.nlargest(k, "score"))
    return result.reset_index(drop=True)

def data_aug_train_loader(op_name, mag, signed, cfg, base_transform, img_size, dataset_name, num_bins):
    root = cfg.dataset.path
    weight_df = pd.DataFrame([{"op_name": op_name,
                               "magnitude": mag,
                               "signed": signed,
                               "score": 1.0}])
    
    if 'cub' in dataset_name:
        print("Loading Cub....")
        dataset = Cub2011(root,
                          train=True, 
                          weights_df= weight_df,
                          base_transform=base_transform,
                          img_size=img_size,
                          num_bins=num_bins)

    elif 'dog' in dataset_name:
        print("Loading Dogs...")
        dataset = StanfordDogs(root, 
                               train=True,
                               weight_df=weight_df,
                               base_transform=base_transform,
                               img_size=img_size, 
                               num_bins=num_bins)
       
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)
    return dataloader


def calculate_weights(model, args, cfg, img_size, timm=True):
    scores = []
    for op_name in list(aug_space.keys()):
        if isinstance(aug_space[op_name][0].tolist(), float):
            magnitudes = [aug_space[op_name][0].tolist()]
        else:
            magnitudes = aug_space[op_name][0].tolist()

        for mag in magnitudes:
            signed = aug_space[op_name][1]
            dataloader = data_aug_train_loader(op_name=op_name,
                                               mag=mag,
                                               signed=signed,
                                               cfg=cfg,
                                               base_transform=base_transforms, 
                                               img_size=img_size, 
                                               dataset_name=cfg.dataset.name,
                                               num_bins=args.num_bins)
            
            if "logme" in args.measure:
                features, labels = compute_features(model, dataloader, device, timm)
                
                logme = LogME(regression=False)
                score = logme.fit(features, labels)

            scores.append({"op_name": op_name, 
                           "magnitude": mag,
                           "signed": signed,
                           "score": score})

    score_df = pd.DataFrame(scores)
    score_df = get_top_k_and_single(score_df, k=args.num_augs)

    print(f"after filtered augs")
    print(score_df)

    score_df["probability"] = softmax(score_df["score"].values, tau=0.01)
    return score_df


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
    
    timm_names = ["resnetaa50d.d_in12k","vit_small_patch16_dinov3.lvd1689m"]
    if "timm" in args.uni_model_name:
        
        import timm
        use_timm = True
        print("Using timm models")

        if args.uni_model_name == "resnet50_timm":
            model = timm.create_model('resnetaa50d.d_in12k',
                                        pretrained=True,
                                        num_classes=0)
        elif args.uni_model_name == "vit_timm":
            model = timm.create_model("vit_small_patch16_dinov3.lvd1689m",
                                        pretrained=True,
                                        num_classes=0)
        elif args.uni_model_name == "vit_timm_openai":
            model = timm.create_model("timm/vit_base_patch16_clip_224.openai",
                                      pretrained=True,
                                      num_classes=0)
        model = model.to(device)
    else:
        print("Using models from our code")
        use_timm = False 
        cfg.model.pretrained = True
        model = get_model(cfg, args.uni_model_name).to(device)
    
    if os.path.exists(args.weight_path):
        print(f"Using scores from csv file: {args.weight_path}")
        score_df = pd.read_csv(args.weight_path)
        print(score_df)
    else:
        print(f"Calculating the current aug weights")
        scores = []
        for op_name in list(aug_space.keys()):
            if isinstance(aug_space[op_name][0].tolist(), float):
                magnitudes = [aug_space[op_name][0].tolist()]
            else:
                magnitudes = aug_space[op_name][0].tolist()

            for mag in magnitudes:
                signed = aug_space[op_name][1]
                dataloader = data_aug_train_loader(op_name=op_name,
                                                   mag=mag,
                                                   signed=signed,
                                                   cfg=cfg,
                                                   base_transform=base_transforms, 
                                                   img_size=img_size, 
                                                   dataset_name=dataset_name,
                                                   num_bins=args.num_bins)
                
                if "logme" in args.measure:
                    features, labels = compute_features(model, dataloader, device, use_timm=use_timm)
                    print(features.shape)
                    logme = LogME(regression=False)
                    score = logme.fit(features, labels)
                
                elif "mmd" in args.measure:
                    features, labels = compute_features(model, dataloader, device, logits=True)
                    features, labels = torch.Tensor(features), torch.Tensor(labels).long()
                    
                    labels = torch.nn.functional.one_hot(labels, num_classes=cfg.dataset.num_classes).float() 
                    score = batched_rbf_mmd2(features, labels)
                
                elif "leep" in args.measure:
                    features, labels = compute_features(model, dataloader, device, logits=True)
                    score = LEEP(features, labels)

                elif "etran" in args.measure:
                    features, labels = compute_features(model, dataloader, device)
                    lda_score = LDA_Score(features, labels)
                    energy_score = Energy_Score(features, 0.5, 'bot')
                    score = 0.85 * lda_score + 0.15 * energy_score

                scores.append({"op_name": op_name, 
                               "magnitude": mag,
                               "signed": signed,
                               "score": score})
                
                print(f"op {op_name} - mag {mag}: {score}") 
    
        score_df = pd.DataFrame(scores)
#        csv_file_name = f"results_aistats/scores/data_aug_{args.uni_model_name}_{dataset_name}_{args.measure}.csv"
#        score_df.to_csv(csv_file_name, index=False)
#    exit(0) 
    score_df = get_top_k_and_single(score_df, k=args.num_augs)

    if args.adaptive_tau:
        n = score_df.shape[0]
        tau = np.log2(n)
        mean = score_df["score"].mean()  
        std = score_df["score"].std()
        score_df["new_score"] = (score_df["score"] - mean) / std 
        print(f"Using adaptive tau: tau = {tau}")
        score_df["probability"] = softmax(score_df["new_score"].values, tau=tau)
    else:
        score_df["probability"] = softmax(score_df["score"].values, tau=args.tau)
    
    print("Augmentation score: ", score_df)

    if args.uniform == 1:
        weights = [float(1/len(test_aug_list)) for _ in test_aug_list]
    else:    
        weights = score_df["probability"].to_list()
    print("weights: ", weights)

    global train_dataset

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
    
    cfg.model.pretrained = True
    model = get_model(cfg).to(device)
#
#    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,weight_decay=0.0001)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    ce_loss = torch.nn.CrossEntropyLoss()
    cfg.exp_name = f"{cfg.exp_name}_{args.uni_model_name}" 
    save_path = os.path.join(cfg.dirs.root, cfg.exp_name, cfg.dirs.weights)
    
    best_acc_val = train_model(model, 
                               train_loader, 
                               test_loader, 
                               epochs=cfg.training.epochs, 
                               optimizer=optimizer,
                               criterion=ce_loss, 
                               ckpt_name="data_aug",
                               scheduler=exp_lr_scheduler,
                               save_path=save_path, 
                               save_ckpt=False,
                               logger=logger,
                               cfg=cfg,
                               args=args,
                               train_dataset=train_dataset)

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
    parser.add_argument("--adaptive_tau", action="store_true", help="Using adaptive tau.")
    parser.add_argument("--uni_model_name", type=str, default="resnet50", help="name of the universal model")
    args = parser.parse_args()

    cfg = Quinfig(config_path=args.config_path)
    
    cfg.exp_name += f"_{args.exp_name}"

    if args.uniform == 1:
        cfg.exp_name += "_uniform"
    
    if "weight" in args.aug:
        cfg.exp_name += f"_{args.measure}"
    adaptive_tau = False
    if args.adaptive_tau:
        adaptive_tau = True
    if args.tau > 0 and adaptive_tau is False: 
        cfg.exp_name += f"_tau_{args.tau}"
    else:
        cfg.exp_name += f"_adaptive_tau"
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
