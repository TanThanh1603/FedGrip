"""FedGRIP/HCPP configuration and FedOMG source-update diagnostics."""
from dataclasses import dataclass, fields
import math

import torch

from algorithm.aggregation.fedomg import FedOMGConfig, matching_gradient


EPSILON = 1e-12


@dataclass(frozen=True)
class FedGRIPConfig:
    transport_probability: float = 0.5
    consistency_weight: float = 0.1
    linear_probe_ratio: float = 0.1
    l2sp_weight: float = 1e-4

    def validate(self):
        values = [getattr(self, field.name) for field in fields(self)]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("FedGRIP settings must be finite")
        if not 0 <= self.transport_probability <= 1:
            raise ValueError("transport_probability must be in [0, 1]")
        if self.consistency_weight < 0:
            raise ValueError("consistency_weight must be non-negative")
        if not 0 <= self.linear_probe_ratio < 1:
            raise ValueError("linear_probe_ratio must be in [0, 1)")
        if self.l2sp_weight < 0:
            raise ValueError("l2sp_weight must be non-negative")

    @classmethod
    def from_args(cls, args):
        config = cls(**{
            field.name: getattr(args, "grip_" + field.name, field.default)
            for field in fields(cls)
        })
        config.validate()
        return config


def normalized_entropy(weights):
    if weights.ndim != 1 or weights.numel() == 0:
        raise ValueError("Entropy expects a non-empty weight vector")
    if weights.numel() == 1:
        return weights.new_tensor(1.0)
    safe = weights.clamp_min(EPSILON)
    return -(safe * safe.log()).sum() / math.log(weights.numel())


def _effective_coefficients(task_updates, task_weights, config):
    gram = task_updates.mm(task_updates.t()).cpu()
    scale = (torch.diag(gram) + config.epsilon).sqrt().mean()
    gram = gram / scale.square()
    global_mean = gram.mean(dim=1, keepdim=True).mean(dim=0, keepdim=True)
    radius = (global_mean + config.epsilon).sqrt() * config.cagrad_c
    column = task_weights.detach().cpu().reshape(-1, 1)
    weighted_norm = (
        column.t().mm(gram).mm(column) + config.epsilon
    ).sqrt()
    multiplier = radius.reshape(-1) / (weighted_norm + config.epsilon)
    return (1.0 / len(task_weights) + column * multiplier).reshape(-1)


def matching_gradient_diagnostics(task_updates, fedomg_config=None):
    """Apply FedOMG matching and return source-update diagnostics."""
    if task_updates.ndim != 2 or task_updates.shape[0] < 2:
        raise ValueError("FedGRIP requires at least two source-domain updates")
    config = FedOMGConfig() if fedomg_config is None else fedomg_config
    with torch.enable_grad():
        aggregate, weights, objective = matching_gradient(task_updates, config)
    normalized = task_updates / task_updates.norm(
        dim=1, keepdim=True
    ).clamp_min(EPSILON)
    cosine = normalized.mm(normalized.t())
    off_diagonal = ~torch.eye(
        len(task_updates), dtype=torch.bool, device=cosine.device
    )
    diagnostics = dict(
        domain_weights=weights.float().tolist(),
        weight_entropy=float(normalized_entropy(weights)),
        effective_coefficients=_effective_coefficients(
            task_updates, weights, config
        ).float().tolist(),
        fedomg_objective=float(objective),
        domain_update_norms=task_updates.norm(dim=1).float().tolist(),
        domain_cosine_matrix=cosine.float().tolist(),
        mean_pairwise_cosine=float(cosine[off_diagonal].mean()),
        conflict_fraction=float((cosine[off_diagonal] < 0).float().mean()),
        fedomg_aggregate_norm=float(aggregate.norm()),
    )
    return aggregate, weights.float(), diagnostics
