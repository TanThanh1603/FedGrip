"""Native StableFDG: style sharing, exploration/oversampling and AFH.

Adapted from savertm/StableFDG_github (77eac350f851), ops/oma.py,
ops/style_insert.py, ops/cross_attn.py and backbone/resnet.py.
Integration changes: explicit label lookup for missing local classes, safe
short batches, normalized attention with epsilon, spatial-size-aware pooling.
No dependency on the vendored Dassl runtime.
"""
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from sklearn.cluster import KMeans
from model.models import NUM_CLASSES


def share_style(x, peer, probability=0.5):
    if peer is None or len(x) < 2 or random.random() > probability:
        return x
    mean = x.mean((2, 3), keepdim=True).detach()
    std = (x.var((2, 3), keepdim=True) + 1e-6).sqrt().detach()
    channels = x.shape[1]
    number = min(16, len(x) // 2)
    stats = torch.cat((mean, std), 1).flatten(1)
    centers = KMeans(n_clusters=len(x) - number, init='k-means++',
                     max_iter=1, n_init=1).fit(stats.cpu().numpy()).cluster_centers_
    centers = torch.as_tensor(centers, device=x.device, dtype=x.dtype)
    peer_mean, peer_std = (p.to(x) for p in peer)
    sampled = peer_mean + torch.randn(number, 2 * channels, device=x.device,
                                    dtype=x.dtype) * peer_std
    sampled = torch.cat((sampled[:, :channels], sampled[:, channels:].clamp_min(0)), 1)
    styles = torch.cat((centers, sampled))[:, :, None, None]
    return (x - mean) / std * styles[:, channels:] + styles[:, :channels]


class StyleExploration(nn.Module):
    def __init__(self, level=3.0, oversampling=32):
        super().__init__()
        self.level, self.oversampling = level, oversampling
        self.beta = torch.distributions.Beta(0.1, 0.1)

    def forward(self, x, labels, supplemental, supplemental_labels, first):
        if not self.training or random.random() > 0.5:
            return x, labels, first
        added = []
        if first:
            classes, counts = labels.unique(sorted=True, return_counts=True)
            singletons = classes[counts == 1].tolist()
            # One additional same-class training example for singleton labels.
            selected = singletons
            if len(selected) > self.oversampling:
                selected = np.random.choice(selected, self.oversampling, replace=False).tolist()
            if selected:
                indices = [(supplemental_labels == c).nonzero().flatten()[0].item() for c in selected]
                x = torch.cat((x, supplemental[indices]))
                labels = torch.cat((labels, supplemental_labels[indices]))
            remaining = max(0, self.oversampling - len(selected))
            # Balance extra examples among the currently least represented classes.
            classes, counts = labels.unique(sorted=True, return_counts=True)
            counts = counts.cpu().numpy()
            for _ in range(remaining):
                candidates = np.flatnonzero(counts == counts.min())
                position = int(np.random.choice(candidates))
                indices = (labels == classes[position]).nonzero().flatten()
                added.append(int(indices[int(np.random.randint(len(indices)))]))
                counts[position] += 1
        mean = x.mean((2, 3), keepdim=True)
        std = (x.var((2, 3), keepdim=True) + 1e-6).sqrt()
        normalized = (x - mean) / std
        if added:
            normalized = torch.cat((normalized, normalized[added]))
            labels = torch.cat((labels, labels[added]))
            mean = torch.cat((mean, mean[added] + self.level * (mean[added] - mean.mean(0, keepdim=True))))
            std = torch.cat((std, std[added] + self.level * (std[added] - std.mean(0, keepdim=True))))
        mean, std = mean.detach(), std.detach()
        weight = self.beta.sample((len(labels), 1, 1, 1)).to(x)
        perm = torch.randperm(len(labels), device=x.device)
        return (normalized * (weight * std + (1 - weight) * std[perm])
                + weight * mean + (1 - weight) * mean[perm]), labels, False


class FeatureHighlighter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Original AFH uses dk=dv=30, Nh=1; value projection is unused upstream.
        self.qkv_conv = nn.Conv2d(channels, 90, 1)

    def query_key(self, x):
        q, k, _ = self.qkv_conv(x).chunk(3, dim=1)
        return q.flatten(2) * (30 ** -0.5), k.flatten(2)

    def forward(self, x, labels=None, supplemental_query=None, supplemental_labels=None):
        q, k = self.query_key(x)
        pairs = torch.arange(len(x), device=x.device)
        if self.training:
            replacements = q.clone()
            for label in labels.unique():
                indices = (labels == label).nonzero().flatten()
                if len(indices) > 1:
                    pairs[indices] = indices.roll(-1)
                else:
                    index = (supplemental_labels == label).nonzero().flatten()[0]
                    replacements[indices[0]] = supplemental_query[index]
            q = replacements
        q, k = F.normalize(q, dim=1, eps=1e-12), F.normalize(k, dim=1, eps=1e-12)
        scores = torch.bmm(((q[pairs] + q) * 0.5).transpose(1, 2), k).mean(1)
        attention = scores.softmax(-1).reshape(len(x), 1, *x.shape[2:])
        # Equals upstream x/49 at 224x224 ResNet input (7x7 final feature map).
        return torch.cat((x.mean((2, 3)), (x * attention).sum((2, 3))), dim=1)


class StableFDGModel(nn.Module):
    def __init__(self, dataset, version='res18', level=3.0, oversampling=32, pretrained=True):
        super().__init__()
        if version == 'res18':
            self.base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        elif version == 'res50':
            self.base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            raise ValueError('StableFDG supports res18/res50; MobileNet mapping is not implemented')
        channels = self.base.fc.in_features
        self.base.fc = nn.Identity()
        self.exploration = StyleExploration(level, oversampling)
        self.highlighter = FeatureHighlighter(channels)
        self.classifier = nn.Linear(channels * 2, NUM_CLASSES[dataset])

    def early(self, x):
        b = self.base
        return b.layer1(b.maxpool(b.relu(b.bn1(b.conv1(x)))))

    def supplemental(self, x):
        first = self.early(x)
        second = self.base.layer2(first)
        third = self.base.layer3(second)
        last = self.base.layer4(third)
        query, _ = self.highlighter.query_key(last)
        return [first, second, third], query

    def forward(self, x, labels=None, supplemental=None, peer=None):
        x = self.early(x)
        if self.training:
            x = share_style(x, peer)
            features, queries, extra_labels = supplemental
        first = True
        for index, layer in enumerate((self.base.layer2, self.base.layer3, self.base.layer4)):
            if self.training:
                x, labels, first = self.exploration(x, labels, features[index], extra_labels, first)
            x = layer(x)
        representation = self.highlighter(x, labels,
                                         queries if self.training else None,
                                         extra_labels if self.training else None)
        logits = self.classifier(representation)
        return (logits, labels) if self.training else logits
