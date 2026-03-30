import os
import torch 
import torch.nn as nn
import numpy as np
import random
import pickle 

from torchvision.datasets import CIFAR10

random.seed(10)
np.random.seed(10)

exp_name = "increase_noise_annotators"
PATH = f"./datasets/noisy_subsets_rebuttal/{exp_name}/training"
VALID_PATH= f"./datasets/noisy_subsets_rebuttal/{exp_name}/valid"
seed = 0
num_test_samples = 100

random.seed(10)
np.random.seed(10)
#noise_scales = np.arange(0, 0.5, 0.05)

num_annotators = 10
#noise_scales = (0, 0.05, 0.1, 0.4, 0.5)
noise_scales = np.array([(float(i / num_annotators)) for i in range(num_annotators)])
print(f"number of noise: {len(noise_scales)} and values {noise_scales}.")

if not os.path.exists(PATH):
    os.makedirs(PATH)

if not os.path.exists(VALID_PATH):
    os.makedirs(VALID_PATH)

def change_label(curr_label):
    labels = set([i for i in range(10)])
    remainer = labels - set([curr_label])
    # random sample from set
    new_label = random.sample(remainer, 1)
    return new_label[0]


training_data = CIFAR10("./datasets/cifar_10",
                        train=True,
                        download=True)

testing_data = CIFAR10("./datasets/cifar_10",
                        train=False,
                        transform=None)


training_data.targets = np.array(training_data.targets)
testing_data.targets = np.array(testing_data.targets)

print("Number of images in test set:", testing_data.targets.shape[0])
print("Number of images in training set:", training_data.targets.shape[0])

train_labels = np.unique(training_data.targets)
print(train_labels, len(train_labels))

test_labels = np.unique(testing_data.targets)
print(test_labels, len(test_labels))


sub_datasets = []

for train_label in train_labels:
    subdata_idxs = np.where(training_data.targets == train_label)[0].tolist()
    sub_dataset = (training_data.data[subdata_idxs], training_data.targets[subdata_idxs])
    sub_datasets.append(sub_dataset)


test_imgs = []
test_targets = []

picked_idxs = []

ploted_idxs = []

for test_label in test_labels:

    print("Using labels:", test_label)

    subdata_idxs = np.where(testing_data.targets == test_label)[0].tolist()
    validation_idxs = random.sample(subdata_idxs, num_test_samples)

    picked_idxs.extend(validation_idxs)
    ploted_idxs.extend(validation_idxs[:4])

    test_imgs.append(testing_data.data[validation_idxs])
    test_targets.append(testing_data.targets[validation_idxs])

test_imgs = np.concatenate(test_imgs)
test_targets = np.concatenate(test_targets)

print(f"# of testing images: {test_imgs.shape}")
print(f"# of test labels: {test_targets.shape}")

print("number of picked idxs: ", len(np.unique(np.array(picked_idxs))))
print(len(ploted_idxs), ploted_idxs)


file_name = f"buyer_validation_set.pkl"
validation_path = os.path.join(VALID_PATH, file_name)
with open(validation_path, "wb") as f:
    pickle.dump((test_imgs, test_targets), f)

print(f"Saved buyer's data at {validation_path}.")


annotator_len = sub_datasets[0][0].shape[0] // num_annotators
print("annotator len: ", annotator_len)


annotators_set = []

for sub_dataset in sub_datasets:
    curr_label_anno_set = []
    ori_idxs = list(range(sub_dataset[0].shape[0]))
    print("ori idxs: ", len(ori_idxs))
    random.shuffle(ori_idxs)
    sub_idxs = [ori_idxs[i:i+annotator_len] for i in range(0, len(ori_idxs), annotator_len)]
    for subset in sub_idxs:
      print("subset len: ", len(subset))
      curr_label_anno_set.append((sub_dataset[0][subset], sub_dataset[1][subset]))
    annotators_set.append(curr_label_anno_set)


n_annotators = num_annotators
print("number of annotators: ", n_annotators)


data_annotators = []
for i in range(n_annotators):
  anno_imgs = []
  anno_targets = []
  for j in range(len(sub_datasets)):
    data = annotators_set[j][i]
    anno_imgs.append(data[0])
    anno_targets.append(data[1])
  anno_imgs = np.concatenate(anno_imgs)
  anno_targets = np.concatenate(anno_targets)
  print("n - images:", len(anno_imgs))
  print("n - labels: ", len(anno_targets))
  print("unique labels", len(np.unique(anno_targets)))
  data_annotators.append((anno_imgs, anno_targets))


for i, (annotator, noise) in enumerate(zip(data_annotators, noise_scales)):
    changed = 0
    for idx, (_, label) in enumerate(zip(annotator[0], annotator[1])):
        if np.random.rand() < noise:
            new_label = change_label(label)
            annotator[1][idx] = new_label
            # print(f"change label from {label} to {new_label}")
            changed += 1
    print("number of labels changed", changed)
    print("rating = ", changed / 5000)
    # dump this annotator to pkl

    file_name = f"annotator_{i}_noise_{noise:.2f}.pkl"
    training_path = os.path.join(PATH, file_name)
    with open(os.path.join(PATH, file_name), "wb") as f:
        pickle.dump(annotator, f)
    print(f"Saved data of annotator {i} at {training_path}.")
