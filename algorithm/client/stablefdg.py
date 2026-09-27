"""StableFDG local training on the project's source-only partitions."""
import random
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from algorithm.client.fedavg import FedAvgClient
from model.stablefdg import StableFDGModel
from utils.optimizers_shcedulers import get_optimizer, CosineAnnealingLRWithWarmup
from utils.tools import local_time


class StableFDGClient(FedAvgClient):
    def initialize_model(self):
        self.classification_model = StableFDGModel(
            self.args.dataset, self.args.model, self.args.stable_exploration,
            self.args.stable_oversampling)
        self.device = None
        self.optimizer = get_optimizer(self.classification_model, self.args.optimizer,
                                       self.args.lr, weight_decay=self.args.weight_decay)
        self.scheduler = CosineAnnealingLRWithWarmup(
            self.optimizer, self.args.num_epochs * self.args.round)
        self.peer_style = None
        self.class_indices = {}
        for index, label in enumerate(self.dataset.labels):
            label = self.dataset.label_to_index[label]
            self.class_indices.setdefault(label, []).append(index)

    @torch.no_grad()
    def collect_style(self):
        self.move2new_device()
        self.classification_model.eval()
        total, squared, count = None, None, 0
        for data, _ in DataLoader(self.dataset, batch_size=self.args.batch_size):
            feature = self.classification_model.early(data.to(self.device))
            stats = torch.cat((feature.mean((2, 3)),
                               (feature.var((2, 3)) + 1e-6).sqrt()), 1).double()
            batch_sum, batch_square = stats.sum(0), stats.square().sum(0)
            total = batch_sum if total is None else total + batch_sum
            squared = batch_square if squared is None else squared + batch_square
            count += len(data)
        if count == 0:
            raise ValueError('StableFDG requires nonempty source clients')
        mean = total / count
        variance = ((squared - total.square() / count) / max(count - 1, 1)).clamp_min(0)
        self.classification_model.cpu()
        return mean.float().cpu(), (variance + 1e-6).sqrt().float().cpu()

    def train(self):
        self.move2new_device()
        model = self.classification_model
        model.train()
        peer = None if self.peer_style is None else tuple(x.to(self.device) for x in self.peer_style)
        total, batches = 0.0, 0
        for _ in range(self.args.num_epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                # One local training example per observed class, sorted for explicit lookup.
                extra = [self.dataset[random.choice(self.class_indices[c])]
                         for c in sorted(self.class_indices)]
                extra_x = torch.stack([item[0] for item in extra]).to(self.device)
                extra_y = torch.stack([item[1] for item in extra]).to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                features, queries = model.supplemental(extra_x)
                logits, labels = model(data, target, (features, queries, extra_y), peer)
                loss = F.cross_entropy(logits, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite StableFDG loss')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError('Nonfinite StableFDG gradient')
                self.optimizer.step()
                total += float(loss.detach())
                batches += 1
            self.scheduler.step()
        model.cpu()
        torch.cuda.empty_cache()
        self.logger.log(f'{local_time()}, Client {self.client_id}, StableFDG, Avg Loss: {total / max(batches, 1):.4f}')
