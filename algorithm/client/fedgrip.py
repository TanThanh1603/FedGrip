"""FedGRIP client with CQT and the unified HCPP adaptation mechanism."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from algorithm.aggregation.fedgrip import FedGRIPConfig
from algorithm.client.fedavg import FedAvgClient
from utils.optimizers_shcedulers import CosineAnnealingLRWithWarmup
from utils.tools import get_best_device, local_time


FEATURE_SPLIT = 7
QUANTILES = 9
STYLE_SKETCHES = 8
BACKBONE_LR_MULTIPLIER = 0.1


class FedGRIPClient(FedAvgClient):
    def __init__(self, args, dataset, client_id, logger):
        super().__init__(args, dataset, client_id, logger)
        self.config = FedGRIPConfig.from_args(args)
        if not hasattr(self.classification_model.base, "features"):
            raise ValueError("FedGRIP requires a MobileNet feature backbone")
        self.pretrained_base = {
            name: value.detach().cpu().clone()
            for name, value in self.classification_model.base.named_parameters()
        }
        self.linear_probe_rounds = int(math.ceil(
            self.args.round * self.config.linear_probe_ratio
        ))
        self.round_id = 0
        generator = torch.Generator().manual_seed(args.seed + client_id)
        count = min(STYLE_SKETCHES, len(dataset))
        self.style_indices = torch.randperm(
            len(dataset), generator=generator
        )[:count].tolist()
        self.peer_style_bank = None
        self.training_diagnostics = {}
        self._reset_optimizer()

    def _reset_optimizer(self, optimizer_state=None, scheduler_state=None):
        groups = [
            {
                "params": list(self.classification_model.base.parameters()),
                "lr": self.args.lr * BACKBONE_LR_MULTIPLIER,
                "weight_decay": 0.0,
            },
            {
                "params": list(self.classification_model.classifier.parameters()),
                "lr": self.args.lr,
                "weight_decay": self.args.weight_decay,
            },
        ]
        if self.args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                groups, lr=self.args.lr, momentum=0.9
            )
        elif self.args.optimizer == "adam":
            self.optimizer = torch.optim.Adam(groups, lr=self.args.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {self.args.optimizer}")
        self.scheduler = CosineAnnealingLRWithWarmup(
            self.optimizer, total_epochs=self.args.num_epochs * self.args.round
        )
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        if scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)

    def move2new_device(self):
        device = get_best_device(self.args.use_cuda)
        self.classification_model.to(device)
        if self.device is None or self.device != device:
            optimizer_state = self.optimizer.state_dict()
            scheduler_state = self.scheduler.state_dict()
            self._reset_optimizer(optimizer_state, scheduler_state)
            self.dataset.device = device
        self.device = device

    def _encode_early(self, inputs):
        return self.classification_model.base.features[:FEATURE_SPLIT](inputs)

    def _encode_late(self, features):
        base = self.classification_model.base
        features = base.features[FEATURE_SPLIT:](features)
        features = base.avgpool(features)
        return base.classifier(torch.flatten(features, 1))

    @torch.no_grad()
    def compute_style_sketch(self):
        self.move2new_device()
        modes = {
            name: module.training
            for name, module in self.classification_model.named_modules()
        }
        self.classification_model.eval()
        levels = torch.linspace(0, 1, QUANTILES, device=self.device)
        loader = DataLoader(
            Subset(self.dataset, self.style_indices),
            batch_size=self.args.batch_size, shuffle=False,
        )
        sketches = []
        for data, _ in loader:
            feature = self._encode_early(data.to(self.device)).flatten(2).float()
            sketches.append(torch.quantile(
                feature, levels, dim=2
            ).permute(1, 2, 0).cpu())
        for name, module in self.classification_model.named_modules():
            module.training = modes[name]
        self.classification_model.cpu()
        return torch.cat(sketches, dim=0)

    def download_style_bank(self, style_bank):
        if style_bank.ndim != 3 or style_bank.shape[-1] != QUANTILES:
            raise ValueError("Invalid FedGRIP style bank")
        self.peer_style_bank = style_bank

    def _transport_with_fraction(self, feature):
        if self.peer_style_bank is None:
            raise RuntimeError("Style bank must be downloaded before local training")
        batch, channels, height, width = feature.shape
        selected = (
            torch.rand(batch, device=feature.device)
            < self.config.transport_probability
        )
        if not bool(selected.any()):
            return feature, 0.0
        indices = selected.nonzero(as_tuple=False).flatten()
        count = indices.numel()
        bank = self.peer_style_bank.to(feature.device)
        if bank.shape[1] != channels:
            raise ValueError("Style sketch channel count does not match feature layer")
        styles = bank[torch.randint(len(bank), (count,), device=feature.device)]
        target = F.interpolate(
            styles.reshape(count * channels, 1, QUANTILES),
            size=height * width, mode="linear", align_corners=True,
        ).reshape(count, channels, height * width)
        flat = feature.flatten(2)
        source = flat[indices]
        order = source.detach().argsort(dim=2)
        transported = torch.empty_like(source).scatter_(
            2, order, target.to(source.dtype)
        )
        strength = torch.rand(
            count, 1, 1, device=feature.device, dtype=feature.dtype
        )
        result = flat.clone()
        result[indices] = strength * source + (1 - strength) * transported
        return result.reshape_as(feature), float(selected.float().mean())

    def _hcpp_pretrained_anchor_penalty(self):
        penalty = torch.zeros((), device=self.device)
        for name, parameter in self.classification_model.base.named_parameters():
            anchor = self.pretrained_base[name].to(
                device=parameter.device, dtype=parameter.dtype
            )
            penalty = penalty + (parameter - anchor).square().sum()
        return 0.5 * self.config.l2sp_weight * penalty

    def _encode_late_transport(self, features):
        late = self.classification_model.base.features[FEATURE_SPLIT:]
        batch_norms = [
            module for module in late.modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
        ]
        modes = [module.training for module in batch_norms]
        try:
            for module in batch_norms:
                module.eval()
            return self._encode_late(features)
        finally:
            for module, training in zip(batch_norms, modes):
                module.train(training)

    def train(self):
        self.move2new_device()
        linear_probe = self.round_id < self.linear_probe_rounds
        self.classification_model.train()
        for module in self.classification_model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        for parameter in self.classification_model.base.parameters():
            parameter.requires_grad_(not linear_probe)
        if linear_probe:
            self.classification_model.base.eval()
        classifier_lr_start = float(self.optimizer.param_groups[1]["lr"])
        backbone_lr_start = classifier_lr_start * BACKBONE_LR_MULTIPLIER
        self.optimizer.param_groups[0]["lr"] = backbone_lr_start
        self.scheduler.base_lrs[0] = (
            self.scheduler.base_lrs[1] * BACKBONE_LR_MULTIPLIER
        )
        totals = dict(
            clean_ce=0.0, transported_ce=0.0, consistency=0.0,
            hcpp_preservation=0.0, total_loss=0.0, transport_fraction=0.0,
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
                transported, fraction = self._transport_with_fraction(early)
                transported_logits = self.classification_model.classifier(
                    self._encode_late_transport(transported)
                )
                clean_ce = F.cross_entropy(clean_logits, target)
                transported_ce = F.cross_entropy(transported_logits, target)
                clean_probability = F.softmax(clean_logits, dim=1)
                transported_probability = F.softmax(transported_logits, dim=1)
                mean_probability = (
                    0.5 * (clean_probability + transported_probability)
                ).clamp_min(1e-7)
                consistency = 0.5 * (
                    F.kl_div(mean_probability.log(), clean_probability,
                             reduction="batchmean")
                    + F.kl_div(mean_probability.log(), transported_probability,
                               reduction="batchmean")
                )
                preservation = (
                    clean_ce.new_zeros(())
                    if linear_probe else self._hcpp_pretrained_anchor_penalty()
                )
                loss = (
                    0.5 * (clean_ce + transported_ce)
                    + self.config.consistency_weight * consistency + preservation
                )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(
                        f"Non-finite FedGRIP loss at client {self.client_id}, "
                        f"round {self.round_id}"
                    )
                loss.backward()
                self.optimizer.step()
                totals["clean_ce"] += float(clean_ce.detach())
                totals["transported_ce"] += float(transported_ce.detach())
                totals["consistency"] += float(consistency.detach())
                totals["hcpp_preservation"] += float(preservation.detach())
                totals["total_loss"] += float(loss.detach())
                totals["transport_fraction"] += fraction
                batches += 1
            self.scheduler.step()
        for parameter in self.classification_model.base.parameters():
            parameter.requires_grad_(True)
        self.training_diagnostics = {
            key: value / batches for key, value in totals.items()
        }
        self.training_diagnostics.update(
            phase="linear_probe" if linear_probe else "full_finetune",
            linear_probe_rounds=self.linear_probe_rounds,
            transport_clean_ce_gap=(
                self.training_diagnostics["transported_ce"]
                - self.training_diagnostics["clean_ce"]
            ),
            learning_rate_start=classifier_lr_start,
            learning_rate_end=float(self.optimizer.param_groups[1]["lr"]),
            backbone_learning_rate_start=backbone_lr_start,
            backbone_learning_rate_end=float(self.optimizer.param_groups[0]["lr"]),
        )
        self.classification_model.cpu()
        self.peer_style_bank = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.log(
            f"{local_time()}, Client {self.client_id}, FedGRIP "
            f"phase={self.training_diagnostics['phase']}, "
            f"clean={self.training_diagnostics['clean_ce']:.4f}, "
            f"transported={self.training_diagnostics['transported_ce']:.4f}, "
            f"consistency={self.training_diagnostics['consistency']:.6f}, "
            f"hcpp_preservation={self.training_diagnostics['hcpp_preservation']:.6f}, "
            f"total={self.training_diagnostics['total_loss']:.4f}, "
            f"transport_fraction={self.training_diagnostics['transport_fraction']:.4f}, "
            f"backbone_lr={backbone_lr_start:.8f}->"
            f"{self.training_diagnostics['backbone_learning_rate_end']:.8f}, "
            f"classifier_lr={classifier_lr_start:.8f}->"
            f"{self.training_diagnostics['learning_rate_end']:.8f}"
        )
