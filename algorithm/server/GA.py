from argparse import ArgumentParser, Namespace
from copy import deepcopy
from pathlib import Path
import sys
from typing import List
import numpy as np

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(PROJECT_DIR.as_posix())
from algorithm.server.fedavg import FedAvgServer, get_fedavg_argparser
from algorithm.client.GA import GAClient
from data.dataset import FLDataset


def get_GA_argparser():
    parser = get_fedavg_argparser()
    parser.add_argument("--step_size", type=float, default=0.001)
    return parser


class GAServer(FedAvgServer):
    def __init__(self, algo="GA", args: Namespace = None):
        if args is None:
            args = get_GA_argparser().parse_args()
        super().__init__(algo, args)

    def initialize_clients(self):
        self.client_list = [
            GAClient(self.args, FLDataset(self.args, client_id), client_id, self.logger)
            for client_id in range(self.num_client)
        ]

    def get_agg_weight(self) -> List[float]:
        # Get the weight of each client at the time of aggregation
        if len(self.generalization_gap) == 0:
            num_data_each_client = [len(client.train_loader) for client in self.client_list]
            num_total_data = sum(num_data_each_client)
            weight_list = [num_data / num_total_data for num_data in num_data_each_client]
        else:
            mean_generalization_gap = sum(self.generalization_gap) / len(self.generalization_gap)
            centralized_generalization_gap = [
                g_gap - mean_generalization_gap for g_gap in self.generalization_gap
            ]
            max_centralized_gap = max(centralized_generalization_gap)
            weight_list_prime = [
                self.step_size * g_gap / max_centralized_gap + self.agg_weight[i]
                for i, g_gap in enumerate(centralized_generalization_gap)
            ]
            sum_weight_prime = sum(weight_list_prime)
            weight_list = [weight / sum_weight_prime for weight in weight_list_prime]
        if self.mu_aggregator is not None:
            weight_list = self.mu_aggregator.compute(
                weight_list, getattr(self, "round_id", 0)
            )
        return weight_list

    def aggregate_model(self):
        self.agg_weight = self.get_agg_weight()
        model_weight_each_client = [client.get_model_weights() for client in self.client_list]
        new_model_weight = {}
        for key in model_weight_each_client[0].keys():
            new_model_weight[key] = sum(
                [
                    model_weight[key] * weight
                    for model_weight, weight in zip(model_weight_each_client, self.agg_weight)
                ]
            )
        return new_model_weight

    def process_classification(self):
        for client_id in range(self.num_client):
            self.client_list[client_id].load_model_weights(
                deepcopy(self.classification_model.state_dict())
            )
        self.best_accuracy = 0
        self.generalization_gap: List[float] = []
        for round_id in range(self.args.round):
            self.round_id = round_id
            self.step_size = (
                self.args.step_size * (self.args.round - round_id) / self.args.round / 3
            )
            self.logger.log("=" * 20, f"Round {round_id}", "=" * 20)
            for client_id in range(self.num_client):
                self.client_list[client_id].train()
            aggregated_weights = self.aggregate_model()
            self.classification_model.load_state_dict(aggregated_weights)
            for client_id in range(self.num_client):
                self.client_list[client_id].load_model_weights(aggregated_weights)
            self.generalization_gap = [
                client.get_generalization_gap() for client in self.client_list
            ]
            if (round_id + 1) % self.args.test_gap == 0:
                self.validate_and_test()


if __name__ == "__main__":
    server = GAServer()
    server.process_classification()
