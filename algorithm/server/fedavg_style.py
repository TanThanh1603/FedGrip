"""FedAvg server and resumable loop for the controlled style comparison."""
from copy import deepcopy
import time

import torch

from algorithm.client.fedavg_style import FedAvgStyleClient
from algorithm.server.fedavg import FedAvgServer, get_fedavg_argparser
from data.dataset import FLDataset
from experiment_checkpoints import digest, write_json
from fdg_round_checkpoints import load_checkpoint, save_checkpoint


def get_fedavg_style_argparser():
    parser = get_fedavg_argparser()
    parser.add_argument(
        "--style_variant",
        choices=("mixstyle", "cqt"),
        default="cqt",
    )
    parser.add_argument("--style_transport_probability", type=float, default=0.5)
    parser.add_argument("--style_consistency_weight", type=float, default=0.1)
    parser.add_argument("--style_quantiles", type=int, default=9)
    parser.add_argument("--style_sketches", type=int, default=8)
    return parser


class FedAvgStyleServer(FedAvgServer):
    def initialize_model(self):
        if self.args.mu_on or self.args.augment:
            raise ValueError("FedAvg style comparison requires mu_on=0 and augment=False")
        if not self.args.model.startswith("mobile"):
            raise ValueError("FedAvg style comparison requires a MobileNet backbone")
        super().initialize_model()

    def initialize_clients(self):
        self.client_list = [
            FedAvgStyleClient(
                self.args, FLDataset(self.args, index), index, self.logger
            )
            for index in range(self.num_client)
        ]
        self.client_domains = []
        for client in self.client_list:
            domains = set(client.dataset.client_data[client.client_id]["domain"])
            if len(domains) != 1 or self.args.test_domain in domains:
                raise ValueError(
                    "Style comparison requires single-source clients without target data"
                )
            self.client_domains.append(next(iter(domains)))
        if len(set(self.client_domains)) < 2:
            raise ValueError("Style comparison requires at least two source domains")

    def prepare_style_round(self):
        if self.args.style_variant == "mixstyle":
            return dict(upload_floats=0, download_floats=0)
        signatures = [
            client.compute_style_signature() for client in self.client_list
        ]
        upload = sum(signature.numel() for signature in signatures)
        download = 0
        for client, own_domain in zip(self.client_list, self.client_domains):
            peer = [
                signature
                for signature, domain in zip(signatures, self.client_domains)
                if domain != own_domain
            ]
            bank = torch.cat(peer, dim=0)
            client.download_style_bank(bank)
            download += bank.numel()
        diagnostics = dict(upload_floats=upload, download_floats=download)
        self.logger.log("Style communication:", diagnostics)
        return diagnostics


def train(server, job, directory):
    checkpoint, signature = directory / "round_checkpoint.pt", digest(job)
    if checkpoint.exists():
        start, history = load_checkpoint(checkpoint, signature, server)
    else:
        initial = deepcopy(server.classification_model.state_dict())
        for client in server.client_list:
            client.load_model_weights(initial)
        server.best_accuracy = 0
        start, history = 0, []
        save_checkpoint(checkpoint, signature, start, server, history)
    server.logger.log(
        f"START/RESUME FedAvg+{server.args.style_variant} after "
        f"{start}/{server.args.round} rounds"
    )
    for round_id in range(start, server.args.round):
        server.round_id = round_id
        begin = time.monotonic()
        server.logger.log("=" * 20, f"Round {round_id}", "=" * 20)
        communication = server.prepare_style_round()
        for client in server.client_list:
            client.train()
        weights = server.aggregate_model()
        server.classification_model.load_state_dict(weights)
        for client in server.client_list:
            client.load_model_weights(weights)
        server.validate_and_test()
        history.append(dict(
            round=round_id,
            test_accuracy=float(server.best_accuracy),
            weights=[float(value) for value in server.agg_weight],
            style_communication=communication,
            client_training=[
                dict(client_id=client.client_id, **client.training_diagnostics)
                for client in server.client_list
            ],
            seconds_before_checkpoint=time.monotonic() - begin,
        ))
        save_checkpoint(checkpoint, signature, round_id + 1, server, history)
        write_json(directory / "round_metrics.json", history)
        server.logger.log(
            f"Checkpoint saved: completed {round_id + 1}/{server.args.round} rounds"
        )
    write_json(directory / "complete.json", dict(
        signature=signature,
        next_round=server.args.round,
        history=history,
        accuracy=float(server.best_accuracy),
        fast_final_accuracy=float(history[-1]["test_accuracy"]),
        fast_peak_accuracy=max(float(item["test_accuracy"]) for item in history),
        aggregation="FedAvg",
        style_variant=server.args.style_variant,
        model_selection="final_round",
    ))
