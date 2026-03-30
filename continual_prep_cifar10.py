import torch
from torchvision import datasets, transforms
import numpy as np
import pickle
import os
import itertools

def create_non_overlapping_balanced_noisy_cifar10_subsets_v3(data_dir='./data', num_initial_batches=4, num_subsets=5, samples_per_class=-1, seed=42):
    """
    Randomly splits the CIFAR-10 training set into smaller batches and then further
    splits each batch into completely non-overlapping subsets with equal samples
    per class before flipping labels. Checks the number of samples and non-overlapping
    indices for each set before applying noise. Then applies random label noise.

    Args:
        data_dir (str): Directory to store the CIFAR-10 dataset.
        num_initial_batches (int): Number of smaller batches to split the training set into.
        num_subsets (int): Number of subsets to split each initial batch into.
        seed (int): Random seed for reproducibility.
    """

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load the CIFAR-10 training dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    train_data = trainset.data
    train_labels = np.array(trainset.targets)
    num_classes = 10
    num_train_samples = len(train_data)
    
    noises = [0.0, 0.2, 0.4, 0.6, 0.8]
    num_subsets = len(noises)
    
    # Calculate the number of samples per class
#    total_samples_per_class = int(num_train_samples // num_classes) if samples_per_class == -1 else samples_per_class 
#    print(f"Using {samples_per_class} for each class")
##    initial_batch_size_per_class = samples_per_class // num_initial_batches
#    initial_batch_size_per_class = int(total_samples_per_class // num_initial_batches) if samples_per_class == -1 else sample_per_class
#    subset_size_per_class = initial_batch_size_per_class // num_subsets
    subset_size_per_class = samples_per_class
    initial_batch_size_per_class = subset_size_per_class * num_subsets
    print("information about data: ",subset_size_per_class, initial_batch_size_per_class)
    # Organize data by class
    class_indices = [np.where(train_labels == i)[0] for i in range(num_classes)]

    all_batches_indices_by_class = []
    for c in range(num_classes):
        shuffled_class_indices = np.random.permutation(class_indices[c])
        batches_indices = np.split(shuffled_class_indices[:initial_batch_size_per_class * num_initial_batches], num_initial_batches)
        all_batches_indices_by_class.append(batches_indices)

    for i in range(num_initial_batches):
        batch_indices = np.concatenate([all_batches_indices_by_class[c][i] for c in range(num_classes)])
        batch_data = train_data[batch_indices]
        batch_labels = train_labels[batch_indices]

        subset_indices_by_class = []
        for c in range(num_classes):
            class_batch_indices = np.arange(initial_batch_size_per_class)
            shuffled_class_batch_indices = np.random.permutation(class_batch_indices)
            subsets_indices = np.split(shuffled_class_batch_indices[:subset_size_per_class * num_subsets], num_subsets)
            subset_indices_by_class.append(subsets_indices)

        all_subset_indices = [[] for _ in range(num_subsets)]
        for j in range(num_subsets):
            for c in range(num_classes):
                start_index_in_batch = c * initial_batch_size_per_class
                subset_indices = subset_indices_by_class[c][j]
                global_indices = batch_indices[start_index_in_batch + subset_indices]
                all_subset_indices[j].extend(global_indices)
            all_subset_indices[j] = np.array(all_subset_indices[j])
        
        curr_noise = np.random.permutation(noises)
        print("Curr Noises: ", curr_noise)

        for j, noise_level in enumerate(curr_noise):
            subset_data_list = []
            subset_labels_list = []
            original_subset_labels_list = []
            current_subset_indices = all_subset_indices[j]

            for c in range(num_classes):
                start_index_in_batch = c * initial_batch_size_per_class
                data_for_class = batch_data[start_index_in_batch : start_index_in_batch + initial_batch_size_per_class]
                labels_for_class = batch_labels[start_index_in_batch : start_index_in_batch + initial_batch_size_per_class]
                subset_local_indices = subset_indices_by_class[c][j]

                subset_data_list.append(data_for_class[subset_local_indices])
                subset_labels_list.append(labels_for_class[subset_local_indices])

            subset_data = np.concatenate(subset_data_list)
            subset_labels = np.concatenate(subset_labels_list)
            original_subset_labels = subset_labels.copy()
             
            # Randomly flip labels based on the noise level
            num_noise = int(len(subset_labels) * noise_level)
            corrupted_indices = np.random.choice(len(subset_labels), num_noise, replace=False)

            for k in corrupted_indices:
                current_label = subset_labels[k]
                possible_labels = [l for l in range(num_classes) if l != current_label]
                if possible_labels:
                    subset_labels[k] = np.random.choice(possible_labels)

            # Save the subset to a pickle file
            filename = f'cifar10_non_overlap_permutation_size_{samples_per_class}_batch_{i+1}_seed_{seed}_subset_{j+1}_noise_{noise_level:.2f}.pkl'
            filepath = os.path.join(data_dir, 'noisy_cifar10_subsets_permutation_v2', filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            print(f"Data size: {subset_data.shape}")
            data = (subset_data, subset_labels)
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)

            print(f"Saved subset: {filename}")


create_non_overlapping_balanced_noisy_cifar10_subsets_v3(data_dir='./datasets/continual_cifar10', seed=100, samples_per_class=100)
print("Finished creating non-overlapping balanced noisy CIFAR-10 subsets.")
