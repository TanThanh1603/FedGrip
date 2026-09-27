"""Native SAM local updates, following FedOMG's FedSAM ESAM implementation.

Reference: skydvn/FedOMG, e1ddccabd1f4, FedOMG-DG/algorithms/fedsam.
Uses the project's optimizer/scheduler and data protocol (not a paper reproduction).
"""
import torch
from torch.nn import functional as F

from algorithm.client.fedavg import FedAvgClient
from utils.tools import local_time


def sam_step(model, optimizer, data, target, rho, clip_norm=10.0):
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(data), target)
    loss.backward()
    parameters = [p for p in model.parameters() if p.grad is not None]
    norm = torch.stack([p.grad.norm(2) for p in parameters]).norm(2)
    if not torch.isfinite(norm):
        raise FloatingPointError('Nonfinite SAM gradient')
    originals = [p.detach().clone() for p in parameters]
    try:
        with torch.no_grad():
            for p in parameters:
                p.add_(p.grad * (rho / (norm + 1e-7)))
        optimizer.zero_grad(set_to_none=True)
        perturbed_loss = F.cross_entropy(model(data), target)
        perturbed_loss.backward()
    finally:
        with torch.no_grad():
            for p, original in zip(parameters, originals):
                p.copy_(original)
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=True)
    optimizer.step()
    return loss.detach()


class FedSAMClient(FedAvgClient):
    def train(self):
        self.move2new_device()
        self.classification_model.train()
        total, batches = 0.0, 0
        for _ in range(self.args.num_epochs):
            for data, target in self.train_loader:
                loss = sam_step(self.classification_model, self.optimizer,
                                data.to(self.device), target.to(self.device), self.args.sam_rho)
                total += float(loss)
                batches += 1
            self.scheduler.step()
        self.classification_model.cpu()
        torch.cuda.empty_cache()
        self.logger.log(f'{local_time()}, Client {self.client_id}, FedSAM, Avg Loss: {total / max(batches, 1):.4f}')
