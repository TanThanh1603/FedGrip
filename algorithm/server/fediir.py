from argparse import ArgumentParser, Namespace
from copy import deepcopy
from pathlib import Path
import sys

import torch

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(PROJECT_DIR.as_posix())
from algorithm.server.fedavg import FedAvgServer, get_fedavg_argparser
from algorithm.client.fediir import FedIIRClient
from data.dataset import FLDataset
from utils.tools import local_time


def get_fediir_argparser():
    parser = get_fedavg_argparser()
    #
    parser.add_argument("--gamma", type=float, default=0.0005)
    parser.add_argument("--ema", type=float, default=0.9)
    return parser


class FedIIRServer(FedAvgServer):
    def __init__(self, algo="FedIIR", args: Namespace = None):
        if args is None:
            args = get_fediir_argparser().parse_args()
        super().__init__(algo, args)

    def initialize_clients(self):
        self.client_list = [
            FedIIRClient(self.args, FLDataset(self.args, client_id), client_id, self.logger)
            for client_id in range(self.num_client)
        ]

    def process_classification(self):
        # distribute original model weights to clients
        for client_id in range(self.num_client):
            self.client_list[client_id].load_model_weights(
                deepcopy(self.classification_model.state_dict())
            )
        self.best_accuracy = 0
        self.grad_mean = tuple(
            torch.zeros_like(p) for p in self.classification_model.classifier.parameters()
        )
        for round_id in range(self.args.round):
            self.round_id = round_id
            self.logger.log("=" * 20, f"Round {round_id}", "=" * 20)
            self.client_gradient = [client.get_client_grad() for client in self.client_list]
            mean_client_grad = tuple(
                torch.mean(torch.stack(grads), dim=0) for grads in zip(*self.client_gradient)
            )
            # if round_id == 0:
            #     self.grad_mean = mean_client_grad
            # else:
            self.grad_mean = tuple(
                self.args.ema * mean + (1 - self.args.ema) * mean_client
                for mean, mean_client in zip(self.grad_mean, mean_client_grad)
            )
            self.logger.log(f"{local_time()}, Calculation of mean of gradient done.")
            for client_id in range(self.num_client):
                self.client_list[client_id].set_grad_mean(self.grad_mean)
                self.client_list[client_id].train()
            aggregated_weights = self.aggregate_model()
            self.classification_model.load_state_dict(aggregated_weights)
            for client_id in range(self.num_client):
                self.client_list[client_id].load_model_weights(aggregated_weights)
            if (round_id + 1) % self.args.test_gap == 0:
                self.validate_and_test()


if __name__ == "__main__":
    server = FedIIRServer()
    server.process_classification()
