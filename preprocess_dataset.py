import os
import torch 
import torch.nn as nn
import numpy as np
import random
import pickle 

from torchvision.datasets import CIFAR10

random.seed(10)
np.random.seed(10)
PATH = "./datasets/noisy_subsets/training"
VALID_PATH= "./datasets/noisy_subsets/valid"


def change_label(curr_label):
    labels = set([i for i in range(10)])
    remainer = labels - set([curr_label])
    # random sample from set
    new_label = random.sample(remainer, 1)
    return new_label[0]

if not os.path.exists(PATH):
    os.makedirs(PATH)

if not os.path.exists(VALID_PATH):
    os.makedirs(VALID_PATH)

noise_scales = np.arange(0, 0.5, 0.05) 
print(f"number of noise {len(noise_scales)} and {noise_scales}.")

training_data = CIFAR10("cifar_10", 
                        train=True, 
                        download=True)

testing_data = CIFAR10("datasets/cifar_10/",
                        train=False,
                        transform=None)

training_data.targets = np.array(training_data.targets)
testing_data.targets = np.array(testing_data.targets)

print("Number of images in test set:", testing_data.targets.shape[0])

labels = [i for i in range(0,10)]
print(labels, len(labels))

sub_datasets = []

# get subdataset by label
for label in labels:
    subdata_idxs = np.where(training_data.targets == label)[0].tolist()
#    print(type(subdata_idxs))
#    print(type(training_data.data))
#    print(type(training_data.targets))
    sub_dataset = (training_data.data[subdata_idxs], training_data.targets[subdata_idxs])
    sub_datasets.append(sub_dataset)

# get sub validation dataset by label

# select 100 samples per class for buyer

num_test_samples = 100
sub_test_datasets = []

test_imgs = []
test_labels = []
for label in labels:
    
    print("Using labels:", label)
#    print("where: ", np.where(testing_data.targets == label)[0])
    subdata_idxs = np.where(testing_data.targets == label)[0].tolist()
    sub_test_dataset = (testing_data.data[subdata_idxs], testing_data.targets[subdata_idxs])
    
    validation_indexes = random.sample(list(range(sub_test_dataset[0].shape[0])), k=100) 
    imgs = sub_test_dataset[0][validation_indexes]
    labels = sub_test_dataset[1][validation_indexes] 
    
    test_imgs.append(imgs)
    test_labels.append(labels)
    
test_imgs = np.concatenate(test_imgs)
test_labels = np.concatenate(test_labels)

print(f"# of testing images: {test_imgs.shape}")
print(f"# of test labels: {test_labels.shape}")


file_name = f"buyer_validation_set.pkl"
validation_path = os.path.join(VALID_PATH, file_name)
with open(validation_path, "wb") as f:
    pickle.dump((test_imgs, test_labels), f)

print(f"Saved buyer's data at {validation_path}.")


# divide into 10 sub datasets 
len_a_data = len(sub_dataset[0]) // 10
print("len of each dataset", )


# get data for annotator
annotator_sets = []

for i in range(10):
    imgs = []
    labels = []
    for sub_dataset in sub_datasets:
        idxs = [i for i in range(i * len_a_data, (i+1) * len_a_data)]
        print(f"Loading indexs from {min(idxs)} to {max(idxs)}.") 
        sub_imgs = sub_dataset[0][idxs]
        sub_labels = sub_dataset[1][idxs]
        
        imgs.append(sub_imgs)
        labels.append(sub_labels)
    
    imgs = np.concatenate(imgs)
    labels = np.concatenate(labels)
    print("n _ images:", len(imgs))
    print("n - labels: ", len(labels))
    annotator_sets.append((imgs, labels))


for i, (annotator, noise) in enumerate(zip(annotator_sets, noise_scales)):
    changed = 0 
    for idx, (_, label) in enumerate(zip(annotator[0], annotator[1])):
        if np.random.rand() < noise:
            annotator[1][idx] = change_label(label)
            changed += 1
    print("number of labels changed", changed)
    print("rating = ", changed / 5000) 
    # dump this annotator to pkl

    file_name = f"annotator_{i}_noise_{noise:.2f}.pkl"
    training_path = os.path.join(PATH, file_name)
    with open(os.path.join(PATH, file_name), "wb") as f:
        pickle.dump(annotator, f)
    print(f"Saved data of annotator {i} at {training_path}.")
