import torch
import os 
import sys
import numpy as np
import random 
import torch
import itertools
import functools
import logging

from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from datetime import datetime

def extract_features(model, loader):
    
    model.eval()
    features = []
    labels = []
    
    for batch_inputs in tqdm(loader):
        output_features = model.forward_features(
            batch_inputs[0].contiguous(memory_format=torch.channels_last))
        features.append(output_features.cpu().numpy().tolist())
        labels.append(batch_inputs[1].cpu().numpy().tolist())
    
    features = np.array(list(itertools.chain(*features)))
    labels = np.array(list(itertools.chain(*labels)))
    
    return features, labels


def seed_all(seed=None):
    if seed is None:
        seed = (os.getpid() + int(datetime.now().strftime('%S%f')) +
                int.from_bytes(os.urandom(2), 'big'))
        print('Using a generated random seed {}'.format(seed))

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def worker_init_reset_seed(worker_id):
    initial_seed = torch.initial_seed() % 2**31
    seed_all(initial_seed + worker_id)


@functools.lru_cache()
def setup_logger(name, save_dir, filename='log.txt'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
#    if distributed_rank > 0:
#        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s %(name)s %(levelname)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def default_setup(cfg, args):
    
    root = cfg.dirs.root
    experiment = cfg.exp_name
    root = os.path.join(root, experiment)

    log_dir = os.path.join(root, cfg.dirs.logs)
    weight_dir = os.path.join(root, cfg.dirs.weights)
#    figure_dir = os.path.join(root, cfg.dirs.figure)
    
    print("weight dir: ", weight_dir)
    print("log_dir: ", log_dir)

#    if comm.is_main_process():
    os.makedirs(weight_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
#    os.makedirs(figure_dir, exist_ok=True)

#    rank = comm.get_rank()
    logger_name = experiment
    logger = setup_logger(name=logger_name,
                          save_dir=log_dir,
#                          distributed_rank=rank,
                          filename=f'{experiment}.txt')

#    logger.info('Rank of current process: {}. World size: {}'.format(
#        rank, comm.get_world_size()))
    # TODO: environment info

    logger.info('Command line arguments: ' + str(args))

#    setup_print(comm.is_main_process())
    logger.info(f"Saving weights at: {weight_dir}")
    logger.info(f"Logging file is saving at {log_dir}")
    
    return logger, weight_dir, log_dir

def get_constant_schedule(optimizer, last_epoch=-1):
    """Create a schedule with a constant learning rate."""
    return LambdaLR(optimizer, lambda _: 1, last_epoch=last_epoch)


def get_linear_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    last_epoch=-1):
    """Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)),)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def softmax(x, tau=1.0):
    x = x / tau
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / exp_x.sum()


def neg_softmax(x):
    
    neg_exp = np.exp(-1.0 * x)
    prob = neg_exp / np.sum(neg_exp, axis=0)
    return prob


def accuracy(preds, labels):
    return np.sum(preds == labels) / preds.shape[0]

def LEEP(pseudo_source_label: np.ndarray, target_label: np.ndarray):
    N, C_s = pseudo_source_label.shape
    target_label = target_label.reshape(-1)
    C_t = int(np.max(target_label) + 1)   # the number of target classes
    normalized_prob = pseudo_source_label / float(N)  # sum(normalized_prob) = 1
    joint = np.zeros((C_t, C_s), dtype=float)  # placeholder for joint distribution over (y, z)
    
    for i in range(C_t):
        this_class = normalized_prob[target_label == i]
        row = np.sum(this_class, axis=0)
        joint[i] = row

    p_target_given_source = (joint / joint.sum(axis=0, keepdims=True)).T  # P(y | z)
    empirical_prediction = pseudo_source_label @ p_target_given_source
    empirical_prob = np.array([predict[label] for predict, label in zip(empirical_prediction, target_label)])
    leep_score = np.mean(np.log(empirical_prob))
   
    return leep_score






