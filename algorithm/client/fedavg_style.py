"""Controlled style-augmentation clients for the FedAvg mechanism study."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from algorithm.client.fedavg import FedAvgClient
from utils.tools import local_time


FEATURE_SPLIT = 7


class FedAvgStyleClient(FedAvgClient):
    """Compare local MixStyle and peer quantile transport."""

    def __init__(self, args, dataset, client_id, logger):
        super().__init__(args, dataset, client_id, logger)
        self.variant = args.style_variant
        self.transport_probability = args.style_transport_probability
        self.consistency_weight = args.style_consistency_weight
        self.quantiles = args.style_quantiles
        self.num_sketches = args.style_sketches
        if self.variant not in {"mixstyle", "cqt"}:
            raise ValueError(f"Unknown FedAvg style variant: {self.variant}")
        if not hasattr(self.classification_model.base, "features"):
            raise ValueError("FedAvg style comparison requires a MobileNet backbone")
        if not 0 <= self.transport_probability <= 1:
            raise ValueError("style_transport_probability must be in [0, 1]")
        if self.consistency_weight < 0:
            raise ValueError("style_consistency_weight must be non-negative")
        if self.quantiles < 2 or self.num_sketches < 1:
            raise ValueError("Style quantiles and sketch count must be positive")
        generator = torch.Generator().manual_seed(args.seed + client_id)
        count = min(self.num_sketches, len(dataset))
        self.style_indices = torch.randperm(
            len(dataset), generator=generator
        )[:count].tolist()
        self.peer_style_bank = None
        self.training_diagnostics = {}

    def _encode_early(self, inputs):
        return self.classification_model.base.features[:FEATURE_SPLIT](inputs)

    def _encode_late(self, features):
        base = self.classification_model.base
        features = base.features[FEATURE_SPLIT:](features)
        features = base.avgpool(features)
        return base.classifier(torch.flatten(features, 1))

    @torch.no_grad()
    def compute_style_signature(self):
        if self.variant == "mixstyle":
            raise RuntimeError("Local MixStyle does not communicate a style signature")
        self.move2new_device()
        modes = {
            name: module.training
            for name, module in self.classification_model.named_modules()
        }
        self.classification_model.eval()
        signatures = []
        levels = torch.linspace(0, 1, self.quantiles, device=self.device)
        loader = DataLoader(
            Subset(self.dataset, self.style_indices),
            batch_size=self.args.batch_size,
            shuffle=False,
        )
        for data, _ in loader:
            feature = self._encode_early(data.to(self.device)).flatten(2).float()
            if self.variant == "cqt":
                signature = torch.quantile(
                    feature, levels, dim=2
                ).permute(1, 2, 0)
            signatures.append(signature.cpu())
        for name, module in self.classification_model.named_modules():
            module.training = modes[name]
        self.classification_model.cpu()
        return torch.cat(signatures, dim=0)

    def download_style_bank(self, style_bank):
        expected = self.quantiles
        if style_bank.ndim != 3 or style_bank.shape[-1] != expected:
            raise ValueError(f"Invalid {self.variant} peer style bank")
        self.peer_style_bank = style_bank

    def _selection(self, feature):
        return (
            torch.rand(feature.shape[0], device=feature.device)
            < self.transport_probability
        )

    def _shared_strength(self, count, feature):
        # Uniform mixing is shared by both variants so that only the
        # representation/source of style changes in the controlled study.
        return torch.rand(
            count, 1, 1, device=feature.device, dtype=feature.dtype
        )

    def _mixstyle(self, feature, selected):
        indices = selected.nonzero(as_tuple=False).flatten()
        if indices.numel() == 0 or feature.shape[0] < 2:
            return feature
        flat = feature.flatten(2)
        source = flat[indices]
        permutation = torch.randperm(feature.shape[0], device=feature.device)
        target = flat[permutation[indices]].detach()
        # MixStyle treats instance statistics as style targets, not as a
        # differentiable normalization path (Zhou et al., ICLR 2021).
        source_mean = source.mean(2, keepdim=True).detach()
        source_std = source.std(2, correction=0, keepdim=True).detach().clamp_min(1e-6)
        target_mean = target.mean(2, keepdim=True)
        target_std = target.std(2, correction=0, keepdim=True).clamp_min(1e-6)
        strength = self._shared_strength(indices.numel(), feature)
        mixed_mean = strength * source_mean + (1 - strength) * target_mean
        mixed_std = strength * source_std + (1 - strength) * target_std
        result = flat.clone()
        result[indices] = (
            (source - source_mean) / source_std * mixed_std + mixed_mean
        )
        return result.reshape_as(feature)

    def _cqt(self, feature, selected):
        if self.peer_style_bank is None:
            raise RuntimeError("CQT style bank was not downloaded")
        indices = selected.nonzero(as_tuple=False).flatten()
        if indices.numel() == 0:
            return feature
        batch, channels, height, width = feature.shape
        flat = feature.flatten(2)
        source = flat[indices]
        bank = self.peer_style_bank.to(feature.device)
        styles = bank[
            torch.randint(len(bank), (indices.numel(),), device=feature.device)
        ]
        target = F.interpolate(
            styles.reshape(indices.numel() * channels, 1, self.quantiles),
            size=height * width,
            mode="linear",
            align_corners=True,
        ).reshape(indices.numel(), channels, height * width)
        order = source.detach().argsort(dim=2)
        transported = torch.empty_like(source).scatter_(
            2, order, target.to(source.dtype)
        )
        strength = self._shared_strength(indices.numel(), feature)
        result = flat.clone()
        result[indices] = strength * source + (1 - strength) * transported
        return result.reshape_as(feature)

    def _augment(self, feature):
        selected = self._selection(feature)
        if self.variant == "mixstyle":
            result = self._mixstyle(feature, selected)
        else:
            result = self._cqt(feature, selected)
        return result, float(selected.float().mean())

    def _encode_augmented_late(self, feature):
        """Prevent the second path from updating late BN buffers twice."""
        late = self.classification_model.base.features[FEATURE_SPLIT:]
        batch_norms = [
            module for module in late.modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
        ]
        modes = [module.training for module in batch_norms]
        try:
            for module in batch_norms:
                module.eval()
            return self._encode_late(feature)
        finally:
            for module, training in zip(batch_norms, modes):
                module.train(training)

    def train(self):
        self.move2new_device()
        self.classification_model.train()
        totals = dict(
            clean_ce=0.0,
            augmented_ce=0.0,
            consistency=0.0,
            total_loss=0.0,
            transport_fraction=0.0,
        )
        batches = 0
        for _ in range(self.args.num_epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                early = self._encode_early(data)
                clean_logits = self.classification_model.classifier(
                    self._encode_late(early)
                )
                augmented, fraction = self._augment(early)
                augmented_logits = self.classification_model.classifier(
                    self._encode_augmented_late(augmented)
                )
                clean_ce = F.cross_entropy(clean_logits, target)
                augmented_ce = F.cross_entropy(augmented_logits, target)
                clean_log_probability = F.log_softmax(clean_logits, dim=1)
                augmented_log_probability = F.log_softmax(
                    augmented_logits, dim=1
                )
                mean_log_probability = (
                    torch.logaddexp(
                        clean_log_probability, augmented_log_probability
                    ) - math.log(2.0)
                )
                clean_probability = clean_log_probability.exp()
                augmented_probability = augmented_log_probability.exp()
                # Exact Jensen-Shannon consistency in log space. Using
                # probabilities as KL targets can create log(0) derivatives
                # after softmax underflow, even when the forward loss is finite.
                consistency = 0.5 * (
                    (
                        clean_probability
                        * (clean_log_probability - mean_log_probability)
                    ).sum(dim=1).mean()
                    + (
                        augmented_probability
                        * (augmented_log_probability - mean_log_probability)
                    ).sum(dim=1).mean()
                )
                loss = (
                    0.5 * (clean_ce + augmented_ce)
                    + self.consistency_weight * consistency
                )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(
                        f"Non-finite {self.variant} loss at client {self.client_id}: "
                        f"clean={float(clean_ce.detach()):.6g}, "
                        f"augmented={float(augmented_ce.detach()):.6g}, "
                        f"consistency={float(consistency.detach()):.6g}"
                    )
                loss.backward()
                invalid_gradients = [
                    name for name, parameter
                    in self.classification_model.named_parameters()
                    if parameter.grad is not None
                    and not bool(torch.isfinite(parameter.grad).all())
                ]
                if invalid_gradients:
                    raise FloatingPointError(
                        f"Non-finite {self.variant} gradient at client "
                        f"{self.client_id} in tensors {invalid_gradients[:8]}; "
                        "training stopped before optimizer step"
                    )
                self.optimizer.step()
                totals["clean_ce"] += float(clean_ce.detach())
                totals["augmented_ce"] += float(augmented_ce.detach())
                totals["consistency"] += float(consistency.detach())
                totals["total_loss"] += float(loss.detach())
                totals["transport_fraction"] += fraction
                batches += 1
            self.scheduler.step()
        self.training_diagnostics = {
            key: value / batches for key, value in totals.items()
        }
        self.classification_model.cpu()
        self.peer_style_bank = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.log(
            f"{local_time()}, Client {self.client_id}, {self.variant}, "
            f"clean={self.training_diagnostics['clean_ce']:.4f}, "
            f"augmented={self.training_diagnostics['augmented_ce']:.4f}, "
            f"consistency={self.training_diagnostics['consistency']:.6f}, "
            f"transport_fraction="
            f"{self.training_diagnostics['transport_fraction']:.4f}"
        )
