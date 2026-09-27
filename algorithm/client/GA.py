import torch

from algorithm.client.fedavg import FedAvgClient
from utils.optimizers_shcedulers import get_optimizer
from utils.tools import trainable_params, get_best_device, local_time
import numpy as np


class GAClient(FedAvgClient):
    def __init__(self, args, dataset, client_id, logger):
        super(GAClient, self).__init__(args, dataset, client_id, logger)

    def train(self):
        self.move2new_device()
        self.classification_model.train()
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(self.args.num_epochs):
            total_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                if len(data) <= 1:
                    continue
                self.optimizer.zero_grad()
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.classification_model(data)
                loss = criterion(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            self.scheduler.step()
        average_loss = total_loss / len(self.train_loader)
        self.classification_model.to(torch.device("cpu"))
        torch.cuda.empty_cache()
        self.logger.log(f"{local_time()}, Client {self.client_id}, Avg Loss: {average_loss:.4f}")
        self.average_loss = average_loss

    @torch.no_grad()
    def get_generalization_gap(
        self,
    ):
        device = get_best_device(self.args.use_cuda)
        self.classification_model.to(device)
        self.classification_model.eval()
        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(device)
            target = target.to(device)
            output = self.classification_model(data)
            loss = torch.nn.CrossEntropyLoss()(output, target)
            total_loss += loss.item()
        self.classification_model.to(torch.device("cpu"))
        average_loss = total_loss / len(self.train_loader)
        generalization_gap = average_loss - self.average_loss
        return generalization_gap
