"""Clean FedAvg clients for LP-FT, L2-SP, and TGBA comparison."""
from copy import deepcopy
import math

import torch
from torch.nn import functional as F
from torch import optim
from torch.utils.flop_counter import FlopCounterMode

from algorithm.aggregation.fedavg_backbone_control import BackboneControlConfig
from algorithm.client.fedavg import FedAvgClient
from utils.optimizers_shcedulers import CosineAnnealingLRWithWarmup
from utils.tools import get_best_device, local_time


VARIANTS = ("lpft", "l2sp", "lpft_l2sp", "tgba")
_TRAINING_FLOPS_CACHE = {}
_TRAINING_FLOPS_METADATA_CACHE = {}


class FedAvgBackboneControlClient(FedAvgClient):
    """FedAvg local ERM with optional fixed LP-FT and/or L2-SP."""

    def __init__(self, args, dataset, client_id, logger):
        super().__init__(args, dataset, client_id, logger)
        self.control_variant = args.backbone_control_variant
        if self.control_variant not in VARIANTS:
            raise ValueError(f"Unknown backbone control: {self.control_variant}")
        if not hasattr(self.classification_model, "base"):
            raise ValueError("Backbone-control comparison requires model.base")
        self.control_config = BackboneControlConfig.from_args(args)
        self.pretrained_base = {
            name: value.detach().cpu().clone()
            for name, value in self.classification_model.base.named_parameters()
        }
        self.linear_probe_rounds = int(math.ceil(
            self.args.round * self.control_config.linear_probe_ratio
        )) if self.control_variant in {"lpft", "lpft_l2sp", "tgba"} else 0
        self.tgba_gate = 0.0
        if self.control_variant == "tgba":
            self._initialize_backbone_stages()
            self._initialize_parameter_group_optimizer()
        self.training_flops_per_sample = self._training_flops_profile()
        self.round_id = 0
        self.training_diagnostics = {}

    def _training_flops_profile(self):
        input_size = 128 if "domain" in self.args.dataset else 224
        key = (self.args.model, self.args.dataset, input_size)
        if key in _TRAINING_FLOPS_CACHE:
            return _TRAINING_FLOPS_CACHE[key]

        profiler = object.__new__(FedAvgBackboneControlClient)
        profiler.classification_model = deepcopy(self.classification_model).cpu()
        profiler._initialize_backbone_stages()
        data = torch.zeros(1, 3, input_size, input_size)
        target = torch.zeros(1, dtype=torch.long)
        profile = {}
        for gate in (0.0, 0.25, 0.5, 0.75, 1.0):
            profiler.classification_model.zero_grad(set_to_none=True)
            profiler.classification_model.train()
            profiler._apply_backbone_gate(gate)
            with FlopCounterMode(display=False) as counter:
                loss = F.cross_entropy(
                    profiler.classification_model(data), target
                )
                loss.backward()
            profile[gate] = int(counter.get_total_flops())
        _TRAINING_FLOPS_CACHE[key] = profile
        return profile

    def _initialize_backbone_stages(self):
        base = self.classification_model.base
        if not hasattr(base, "features") or not hasattr(base, "classifier"):
            raise ValueError(
                "TGBA gradual unfreezing requires a MobileNet-style backbone"
            )
        features = list(base.features.children())
        if len(features) != 17:
            raise ValueError(
                "TGBA stage map expects MobileNetV3-Large with 17 feature blocks"
            )
        self.backbone_stages = [
            tuple(features[0:7]),
            tuple(features[7:13]),
            tuple(features[13:17]),
            (base.classifier,),
        ]
        covered = {
            id(parameter)
            for stage in self.backbone_stages
            for module in stage
            for parameter in module.parameters()
        }
        expected = {id(parameter) for parameter in base.parameters()}
        if covered != expected:
            raise RuntimeError("TGBA stage map does not cover the whole backbone")

    def _apply_backbone_gate(self, gate):
        if not 0 <= gate <= 1:
            raise ValueError("TGBA backbone gate must be in [0, 1]")
        stage_count = len(self.backbone_stages)
        active_count = min(stage_count, int(math.ceil(gate * stage_count)))
        self.classification_model.base.eval()
        for parameter in self.classification_model.base.parameters():
            parameter.requires_grad_(False)
        active = (
            self.backbone_stages[stage_count - active_count:]
            if active_count else []
        )
        for stage in active:
            for module in stage:
                module.train()
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        trainable = sum(
            parameter.numel()
            for parameter in self.classification_model.base.parameters()
            if parameter.requires_grad
        )
        total = sum(
            parameter.numel()
            for parameter in self.classification_model.base.parameters()
        )
        return active_count, trainable, total

    def _initialize_parameter_group_optimizer(self):
        backbone = list(self.classification_model.base.parameters())
        backbone_ids = {id(parameter) for parameter in backbone}
        head = [
            parameter for parameter in self.classification_model.parameters()
            if id(parameter) not in backbone_ids
        ]
        groups = [
            {"params": backbone, "role": "backbone"},
            {"params": head, "role": "head"},
        ]
        if self.args.optimizer == "sgd":
            self.optimizer = optim.SGD(
                groups, lr=self.args.lr, momentum=0.9,
                weight_decay=self.args.weight_decay,
            )
        elif self.args.optimizer == "adam":
            self.optimizer = optim.Adam(
                groups, lr=self.args.lr, weight_decay=self.args.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.args.optimizer}")
        self.scheduler = CosineAnnealingLRWithWarmup(
            optimizer=self.optimizer,
            total_epochs=self.args.num_epochs * self.args.round,
        )

    def move2new_device(self):
        if self.control_variant != "tgba":
            return super().move2new_device()
        device = get_best_device(self.args.use_cuda)
        self.classification_model.to(device)
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        self.dataset.device = device
        self.device = device

    def _set_backbone_learning_rate(self, gate):
        head_lr = next(
            group["lr"] for group in self.optimizer.param_groups
            if group["role"] == "head"
        )
        for group in self.optimizer.param_groups:
            if group["role"] == "backbone":
                group["lr"] = head_lr if gate > 0 else 0.0
        return float(head_lr)

    def _l2sp_penalty(self):
        penalty = torch.zeros((), device=self.device)
        for name, parameter in self.classification_model.base.named_parameters():
            if not parameter.requires_grad:
                continue
            anchor = self.pretrained_base[name].to(
                device=parameter.device, dtype=parameter.dtype
            )
            penalty = penalty + (parameter - anchor).square().sum()
        return 0.5 * self.control_config.l2sp_weight * penalty

    def train(self):
        self.move2new_device()
        fixed_linear_probe = self.round_id < self.linear_probe_rounds
        applied_gate = (
            float(self.tgba_gate)
            if self.control_variant == "tgba"
            else (0.0 if fixed_linear_probe else 1.0)
        )
        linear_probe = applied_gate == 0
        use_l2sp = self.control_variant in {
            "l2sp", "lpft_l2sp", "tgba",
        }
        self.classification_model.train()
        if self.control_variant == "tgba":
            active_stages, trainable_backbone, total_backbone = (
                self._apply_backbone_gate(applied_gate)
            )
        else:
            for parameter in self.classification_model.base.parameters():
                parameter.requires_grad_(not linear_probe)
            if linear_probe:
                self.classification_model.base.eval()
            active_stages = 0 if linear_probe else 1
            trainable_backbone = sum(
                parameter.numel()
                for parameter in self.classification_model.base.parameters()
                if parameter.requires_grad
            )
            total_backbone = sum(
                parameter.numel()
                for parameter in self.classification_model.base.parameters()
            )

        head_learning_rate_start = (
            self._set_backbone_learning_rate(applied_gate)
            if self.control_variant == "tgba"
            else float(self.optimizer.param_groups[0]["lr"])
        )
        backbone_learning_rate_start = (
            head_learning_rate_start if applied_gate > 0 else 0.0
        )
        totals = dict(cross_entropy=0.0, l2sp=0.0, total_loss=0.0)
        batches = 0
        for _ in range(self.args.num_epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                cross_entropy = F.cross_entropy(
                    self.classification_model(data), target
                )
                l2sp = (
                    self._l2sp_penalty()
                    if use_l2sp and not linear_probe
                    else cross_entropy.new_zeros(())
                )
                loss = cross_entropy + l2sp
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(
                        f"Non-finite {self.control_variant} loss at client "
                        f"{self.client_id}, round {self.round_id}"
                    )
                loss.backward()
                self.optimizer.step()
                totals["cross_entropy"] += float(cross_entropy.detach())
                totals["l2sp"] += float(l2sp.detach())
                totals["total_loss"] += float(loss.detach())
                batches += 1
            self.scheduler.step()
            if self.control_variant == "tgba":
                self._set_backbone_learning_rate(applied_gate)

        for parameter in self.classification_model.base.parameters():
            parameter.requires_grad_(True)
        self.training_diagnostics = {
            key: value / batches for key, value in totals.items()
        }
        estimated_training_flops = (
            self.training_flops_per_sample[applied_gate]
            * len(self.train_loader.dataset)
            * self.args.num_epochs
        )
        estimated_l2sp_flops = (
            5 * trainable_backbone * batches
            if use_l2sp and not linear_probe else 0
        )
        estimated_training_flops += estimated_l2sp_flops
        self.training_diagnostics.update(
            phase=(
                "linear_probe" if linear_probe else
                ("full_finetune" if applied_gate == 1 else "gradual_unfreezing")
            ),
            linear_probe_rounds=self.linear_probe_rounds,
            applied_backbone_gate=applied_gate,
            active_backbone_stages=active_stages,
            total_backbone_stages=(
                len(self.backbone_stages)
                if self.control_variant == "tgba" else 1
            ),
            trainable_backbone_parameters=trainable_backbone,
            total_backbone_parameters=total_backbone,
            trainable_backbone_fraction=(
                trainable_backbone / total_backbone
                if total_backbone else 0.0
            ),
            profiled_training_flops_per_sample=(
                self.training_flops_per_sample[applied_gate]
            ),
            estimated_l2sp_flops=int(estimated_l2sp_flops),
            estimated_training_flops=int(estimated_training_flops),
            training_flops_method=(
                "torch_flop_counter_forward_backward_plus_"
                "analytical_l2sp"
            ),
            head_learning_rate_start=head_learning_rate_start,
            backbone_learning_rate_start=backbone_learning_rate_start,
            head_learning_rate_end=(
                next(
                    float(group["lr"])
                    for group in self.optimizer.param_groups
                    if group["role"] == "head"
                ) if self.control_variant == "tgba"
                else float(self.optimizer.param_groups[0]["lr"])
            ),
            backbone_learning_rate_end=(
                next(
                    float(group["lr"])
                    for group in self.optimizer.param_groups
                    if group["role"] == "backbone"
                ) if self.control_variant == "tgba"
                else float(self.optimizer.param_groups[0]["lr"])
            ),
        )
        self.classification_model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.log(
            f"{local_time()}, Client {self.client_id}, "
            f"{self.control_variant}, phase={self.training_diagnostics['phase']}, "
            f"ce={self.training_diagnostics['cross_entropy']:.4f}, "
            f"l2sp={self.training_diagnostics['l2sp']:.6f}, "
            f"total={self.training_diagnostics['total_loss']:.4f}, "
            f"gate={applied_gate:.2f}, "
            f"stages={active_stages}/"
            f"{self.training_diagnostics['total_backbone_stages']}, "
            f"trainable_backbone="
            f"{self.training_diagnostics['trainable_backbone_fraction']:.3f}, "
            f"train_flops={estimated_training_flops / 1e12:.4f}T, "
            f"head_lr={head_learning_rate_start:.8f}->"
            f"{self.training_diagnostics['head_learning_rate_end']:.8f}, "
            f"backbone_lr={backbone_learning_rate_start:.8f}->"
            f"{self.training_diagnostics['backbone_learning_rate_end']:.8f}"
        )


def training_flops_metadata(model_name, dataset):
    """Profile one sample once for post-hoc and online FLOPs accounting."""
    key = (model_name, dataset)
    if key in _TRAINING_FLOPS_METADATA_CACHE:
        return _TRAINING_FLOPS_METADATA_CACHE[key]
    from model.models import get_model_arch

    holder = object.__new__(FedAvgBackboneControlClient)
    holder.args = type("FlopsArgs", (), {
        "model": model_name,
        "dataset": dataset,
    })()
    holder.classification_model = get_model_arch(model_name)(dataset=dataset)
    per_sample = holder._training_flops_profile()
    metadata = dict(
        per_sample=per_sample,
        backbone_parameters=sum(
            parameter.numel()
            for parameter in holder.classification_model.base.parameters()
        ),
        method=(
            "torch_flop_counter_forward_backward_plus_analytical_l2sp"
        ),
    )
    _TRAINING_FLOPS_METADATA_CACHE[key] = metadata
    return metadata
