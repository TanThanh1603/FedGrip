"""FedOMG on-server gradient matching used by the ICLR 2025 baseline.

This is a native PyTorch adaptation of the official implementation at
https://github.com/skydvn/FedOMG.  It preserves the published 20 SGD search
steps and Eq. (15)-style client-update combination while exposing explicit
configuration for reproducible experiments in this repository.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FedOMGConfig:
    # Official FedOMG-DG scripts use --global_model_lr 0.5.
    meta_lr: float = 0.5
    cagrad_c: float = 0.5
    optimizer_lr: float = 25.0
    optimizer_steps: int = 20
    momentum: float = 0.5
    epsilon: float = 1e-4

    def validate(self):
        if self.meta_lr <= 0 or self.optimizer_lr <= 0:
            raise ValueError("FedOMG learning rates must be positive")
        if self.cagrad_c < 0 or self.optimizer_steps < 1:
            raise ValueError("Invalid FedOMG search configuration")
        if not 0 <= self.momentum < 1 or self.epsilon <= 0:
            raise ValueError("Invalid FedOMG numerical configuration")


def matching_gradient(client_updates: torch.Tensor, config: FedOMGConfig):
    """Return the FedOMG aggregate update and its optimized task weights."""

    config.validate()
    if client_updates.ndim != 2 or client_updates.shape[0] == 0:
        raise ValueError("FedOMG expects a non-empty [clients, parameters] tensor")
    number_clients = client_updates.shape[0]
    gram = client_updates.mm(client_updates.t()).cpu()
    scale = (torch.diag(gram) + config.epsilon).sqrt().mean()
    gram = gram / scale.square()
    mean_column = gram.mean(dim=1, keepdim=True)
    global_mean = mean_column.mean(dim=0, keepdim=True)
    radius = (global_mean + config.epsilon).sqrt() * config.cagrad_c

    logits = torch.zeros(number_clients, 1, requires_grad=True)
    optimizer = torch.optim.SGD(
        [logits], lr=config.optimizer_lr, momentum=config.momentum
    )
    best_logits = logits.detach().clone()
    best_objective = float("inf")
    # The official loop evaluates the initial point plus 20 SGD updates.
    for step in range(config.optimizer_steps + 1):
        optimizer.zero_grad()
        task_weights = torch.softmax(logits, dim=0)
        norm = (
            task_weights.t().mm(gram).mm(task_weights) + config.epsilon
        ).sqrt()
        objective = task_weights.t().mm(mean_column) + radius * norm
        value = float(objective.detach().item())
        if value < best_objective:
            best_objective = value
            best_logits = logits.detach().clone()
        if step < config.optimizer_steps:
            objective.backward()
            optimizer.step()

    task_weights = torch.softmax(best_logits, dim=0)
    weighted_norm = (
        task_weights.t().mm(gram).mm(task_weights) + config.epsilon
    ).sqrt()
    multiplier = radius.reshape(-1) / (weighted_norm + config.epsilon)
    coefficients = (1.0 / number_clients + task_weights * multiplier).reshape(-1)
    aggregate = (
        coefficients.to(client_updates.device).unsqueeze(1) * client_updates
    ).sum(dim=0) / (1.0 + config.cagrad_c**2)
    return aggregate, task_weights.reshape(-1), best_objective
