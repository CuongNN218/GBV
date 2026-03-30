# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of Gaussian Bhattacharyya Coefficient (GBC).

Pándy, Michal, et al. "Transferability Estimation using Bhattacharyya Class
Separability." https://arxiv.org/abs/2111.12780.
"""

import numpy as np


def compute_bhattacharyya_distance(mu1, mu2, sigma1, sigma2):
  """Compute Bhattacharyya distance between diagonal or spherical Gaussians."""
  avg_sigma = (sigma1 + sigma2) / 2
  first_part = np.sum((mu1 - mu2)**2 / avg_sigma) / 8
  second_part = np.sum(np.log(avg_sigma))
  second_part -= 0.5 * (np.sum(np.log(sigma1)))
  second_part -= 0.5 * (np.sum(np.log(sigma2)))
  return first_part + 0.5 * second_part


def get_bhattacharyya_distance(per_class_stats, c1, c2, gaussian_type):
  """Return Bhattacharyya distance between 2 diagonal or spherical gaussians."""
  mu1 = per_class_stats[c1]['mean']
  mu2 = per_class_stats[c2]['mean']
  sigma1 = per_class_stats[c1]['variance']
  sigma2 = per_class_stats[c2]['variance']
  if gaussian_type == 'spherical':
    # sigma_mean(sigma2)
    sigma1 = np.mean(sigma1)
    sigma2 = np.mean(sigma2)
    
  return compute_bhattacharyya_distance(mu1, mu2, sigma1, sigma2)


def compute_per_class_mean_and_variance(features, target_labels, unique_labels):
  """Compute features mean and variance for each class."""
  per_class_stats = {}
  for label in unique_labels:
    label = int(label)  # For correct indexing
    per_class_stats[label] = {}
    class_ids = np.equal(target_labels, label)
    class_features = np.take(features, np.nonzero(class_ids)[0], axis=0)
    mean = np.mean(class_features, axis=0)
    variance = np.var(class_features, axis=0)
    per_class_stats[label]['mean'] = mean
    # Avoid 0 variance in cases of constant features with tf.maximum
    per_class_stats[label]['variance'] = np.maximum(variance, 1e-4)
  return per_class_stats


def get_gbc_score(features, target_labels, gaussian_type):
  """Compute Gaussian Bhattacharyya Coefficient (GBC).

  Args:
    features: source features from the target data.
    target_labels: ground truth labels in the target label space.
    gaussian_type: type of gaussian used to represent class features. The
      possibilities are spherical (default) or diagonal.

  Returns:
    gbc: transferability metric score.
  """
  assert gaussian_type in ('diagonal', 'spherical')
  unique_labels = np.unique(target_labels)
  unique_labels = list(unique_labels)
  per_class_stats = compute_per_class_mean_and_variance(
      features, target_labels, unique_labels)

  per_class_bhattacharyya_distance = []
  for c1 in unique_labels:
    temp_metric = []
    for c2 in unique_labels:
      if c1 != c2:
        bhattacharyya_distance = get_bhattacharyya_distance(
            per_class_stats, int(c1), int(c2), gaussian_type)
        # temp_metric.append(tf.exp(-bhattacharyya_distance))
        temp_metric.append(np.exp(-bhattacharyya_distance))
    # per_class_bhattacharyya_distance.append(tf.reduce_sum(temp_metric))
    per_class_bhattacharyya_distance.append(np.sum(temp_metric))
  gbc = -np.sum(per_class_bhattacharyya_distance)

  return gbc
