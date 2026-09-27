from copy import deepcopy
from typing import Dict, List, OrderedDict
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import gc

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()
from model.models import get_model_arch
from utils.optimizers_shcedulers import get_optimizer, CosineAnnealingLRWithWarmup
from utils.tools import local_time, get_best_device


class FedAvgClient:
    def __init__(self, args, dataset, client_id, logger):
        self.args = args
        self.dataset = dataset
        self.client_id = client_id
        self.logger = logger
        self.train_loader = DataLoader(
            self.dataset, batch_size=self.args.batch_size, shuffle=True
        )

        self.initialize_model()

    def initialize_model(self):
        self.classification_model = get_model_arch(model_name=self.args.model)(
            dataset=self.args.dataset
        )
        self.device = None
        self.optimizer = get_optimizer(
            self.classification_model,
            self.args.optimizer,
            self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        self.scheduler = CosineAnnealingLRWithWarmup(
            optimizer=self.optimizer,
            total_epochs=self.args.num_epochs * self.args.round,
        )

    def load_model_weights(self, model_weights):
        self.classification_model.load_state_dict(model_weights)

    def get_model_weights(self) -> OrderedDict:
        # self.classification_model.to(torch.device("cpu"))
        return self.classification_model.state_dict()

    def move2new_device(self):
        device = get_best_device(self.args.use_cuda)
        # device = torch.device("cuda:0")
        self.classification_model.to(device)
        if self.device is None or self.device != device:
            optimizer_state = self.optimizer.state_dict()
            scheduler_state = self.scheduler.state_dict()
            self.optimizer = get_optimizer(
                self.classification_model,
                self.args.optimizer,
                self.args.lr,
                weight_decay=self.args.weight_decay,
            )
            self.scheduler = CosineAnnealingLRWithWarmup(
                optimizer=self.optimizer,
                total_epochs=self.args.num_epochs * self.args.round,
            )
            self.optimizer.load_state_dict(optimizer_state)
            self.scheduler.load_state_dict(scheduler_state)
            self.dataset.device = device

        self.device = device

    def train(
        self,
    ):
        self.move2new_device()
        self.classification_model.train()
        criterion = torch.nn.CrossEntropyLoss()

        for _ in range(self.args.num_epochs):
            total_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.classification_model(data)
                loss = criterion(output, target)

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            self.scheduler.step()
        average_loss = total_loss / len(self.train_loader)
        self.classification_model.to(torch.device("cpu"))
        torch.cuda.empty_cache()
        self.logger.log(
            f"{local_time()}, Client {self.client_id}, Avg Loss: {average_loss:.4f}"
        )
