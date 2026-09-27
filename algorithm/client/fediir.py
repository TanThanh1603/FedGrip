from typing import List
import torch
import torch.autograd as autograd

from algorithm.client.fedavg import FedAvgClient
from utils.optimizers_shcedulers import get_optimizer
from utils.tools import trainable_params, get_best_device, local_time


class FedIIRClient(FedAvgClient):
    def __init__(self, args, dataset, client_id, logger):
        super(FedIIRClient, self).__init__(args, dataset, client_id, logger)

    def get_client_grad(self):
        # return gradient of loss w.r.t. model parameters.
        device = get_best_device(self.args.use_cuda)
        self.classification_model.to(device)
        grad_sum = tuple(
            [
                torch.zeros_like(p).to(device)
                for p in self.classification_model.classifier.parameters()
            ]
        )
        criterion = torch.nn.CrossEntropyLoss()

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(device)
            target = target.to(device)
            output = self.classification_model(data)
            loss = criterion(output, target)
            grad_batch = autograd.grad(
                loss, self.classification_model.classifier.parameters(), create_graph=False
            )
            grad_sum = tuple(g1 + g2 for g1, g2 in zip(grad_sum, grad_batch))
        grad_client = tuple(grad.to(torch.device("cpu")) / (batch_idx + 1) for grad in grad_sum)
        self.classification_model.to(torch.device("cpu"))
        torch.cuda.empty_cache()
        return grad_client

    def set_grad_mean(self, grad_mean):
        self.grad_mean = grad_mean

    def train(self):
        self.move2new_device()
        self.classification_model.train()
        criterion = torch.nn.CrossEntropyLoss()
        self.grad_mean = tuple(grad.to(self.device) for grad in self.grad_mean)
        for epoch in range(self.args.num_epochs):
            total_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                if len(data) <= 1:
                    continue
                self.optimizer.zero_grad()
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.classification_model(data)
                loss_erm = criterion(output, target)
                grad_client = autograd.grad(
                    loss_erm, self.classification_model.classifier.parameters(), create_graph=True
                )
                penalty_value = 0
                for g_client, g_mean in zip(grad_client, self.grad_mean):
                    penalty_value += torch.sum((g_client - g_mean) ** 2)
                loss = loss_erm + self.args.gamma * penalty_value
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            self.scheduler.step()
        average_loss = total_loss / len(self.train_loader)
        self.classification_model.to(torch.device("cpu"))
        del self.grad_mean
        torch.cuda.empty_cache()
        self.logger.log(f"{local_time()}, Client {self.client_id}, Avg Loss: {average_loss:.4f}")
