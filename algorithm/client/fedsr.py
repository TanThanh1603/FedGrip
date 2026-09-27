import torch
import torch.nn.functional as F

from algorithm.client.fedavg import FedAvgClient
from utils.optimizers_shcedulers import get_optimizer
from utils.tools import trainable_params, get_best_device, local_time


class FedSRClient(FedAvgClient):
    def __init__(self, args, dataset, client_id, logger):
        super(FedSRClient, self).__init__(args, dataset, client_id, logger)
        self.L2R_coeff = args.L2R_coeff
        self.CMI_coeff = args.CMI_coeff

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
                z, (z_mu, z_sigma) = self.classification_model.featurize(data, return_dist=True)
                output = self.classification_model.classifier(z)
                loss = criterion(output, target)
                obj = loss
                regL2R = torch.zeros_like(obj)
                regCMI = torch.zeros_like(obj)
                regNegEnt = torch.zeros_like(obj)
                if self.L2R_coeff != 0.0:
                    regL2R = z.norm(dim=1).mean()
                    obj = obj + self.L2R_coeff * regL2R
                if self.CMI_coeff != 0.0:
                    r_sigma_softplus = F.softplus(self.classification_model.r_sigma)
                    r_mu = self.classification_model.r_mu[target]
                    r_sigma = r_sigma_softplus[target]
                    z_mu_scaled = z_mu * self.classification_model.C
                    z_sigma_scaled = z_sigma * self.classification_model.C
                    regCMI = (
                        torch.log(r_sigma)
                        - torch.log(z_sigma_scaled)
                        + (z_sigma_scaled**2 + (z_mu_scaled - r_mu) ** 2) / (2 * r_sigma**2)
                        - 0.5
                    )
                    regCMI = regCMI.sum(1).mean()
                    obj = obj + self.CMI_coeff * regCMI
                obj.backward()
                self.optimizer.step()
                total_loss += obj.item()
            self.scheduler.step()
        average_loss = total_loss / len(self.train_loader)
        self.classification_model.to(torch.device("cpu"))
        torch.cuda.empty_cache()
        self.logger.log(f"{local_time()}, Client {self.client_id}, Avg Loss: {average_loss:.4f}")
