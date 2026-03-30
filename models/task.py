import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast


def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)


class Task(nn.Module):
    def __init__(self, feature_extractor):
        super(Task, self).__init__()

        self.feature_extractor = feature_extractor
#
#    @autocast()
    def forward_features(self, input):
        features = self.feature_extractor(input)
        return features

#    @autocast()
    def forward(self, input, labels=None, weights=None):
        if not hasattr(self, 'head'):
            raise NotImplementedError('Model head has not been implemented')
        feature = self.forward_features(input)
        logits = self.head(feature)
        if self.training:
            losses = self.loss(logits, labels, weights)
            return losses, logits
        else:
            return logits

    def loss(self):
        pass


class Classification(Task):
    def __init__(self, feature_extractor, num_classes, cfg=None):
        super(Classification, self).__init__(feature_extractor)
        self.cfg = cfg
        self.head = nn.Sequential(
            nn.Linear(self.feature_extractor.feature_size, num_classes))
        if self.cfg.training.label_smoothing > 0:
            self.criterion = LabelSmoothingCrossEntropy(
                self.cfg.training.label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss(reduction='none')

    def loss(self, logits, labels, weights):
        loss_dict = {}
        loss = self.criterion(logits, labels)
#        print(weights)
        if weights is not None:
            loss = loss * weights
            loss = torch.sum(loss)
        else: 
            loss = torch.mean(loss)
        loss_dict["loss_cls"] = loss
        return loss_dict

class Regression(Task):
    def __init__(self, feature_extractor, num_out, cfg=None):
        super(Classification, self).__init__(feature_extractor)
        self.cfg = cfg
        self.head = nn.Sequential(
            nn.Linear(self.feature_extractor.feature_size, num_out))
        if self.cfg.training.label_smoothing > 0:
            self.criterion = LabelSmoothingCrossEntropy(
                self.cfg.training.label_smoothing)
        else:
            self.criterion = nn.MSELoss(reduction='none')

    def loss(self, logits, labels, weights):
        loss_dict = {}
        loss = self.criterion(logits, labels)
        if weights is not None:
            loss = loss * weights
            loss = torch.mean(loss)
        else: 
            loss = torch.mean(loss)
        loss_dict["loss_cls"] = loss
        return loss_dict

