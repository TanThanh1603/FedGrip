"""FedAvg local training with the official FedOMG server aggregation rule."""

from collections import OrderedDict
from dataclasses import asdict

import torch

from algorithm.aggregation.fedomg import FedOMGConfig, matching_gradient
from algorithm.server.fedavg import FedAvgServer, get_fedavg_argparser
from utils.tools import local_time


def get_fedomg_argparser():
    parser = get_fedavg_argparser()
    group = parser.add_argument_group("FedOMG official-compatible aggregation")
    group.add_argument("--fedomg_meta_lr", type=float, default=0.5)
    group.add_argument("--fedomg_cagrad_c", type=float, default=0.5)
    group.add_argument("--fedomg_optimizer_lr", type=float, default=25.0)
    group.add_argument("--fedomg_optimizer_steps", type=int, default=20)
    group.add_argument("--fedomg_momentum", type=float, default=0.5)
    return parser


class FedOMGServer(FedAvgServer):
    """Replace only FedAvg aggregation; client-side training stays unchanged."""

    def __init__(self, algo="FedOMG", args=None):
        parsed = get_fedomg_argparser().parse_args() if args is None else args
        if getattr(parsed, "mu_on", 0):
            raise ValueError("FedOMG and DIU/AP cannot be enabled together")
        super().__init__(algo=algo, args=parsed)
        self.fedomg_config = FedOMGConfig(
            meta_lr=self.args.fedomg_meta_lr,
            cagrad_c=self.args.fedomg_cagrad_c,
            optimizer_lr=self.args.fedomg_optimizer_lr,
            optimizer_steps=self.args.fedomg_optimizer_steps,
            momentum=self.args.fedomg_momentum,
        )
        self.fedomg_config.validate()
        self.logger.log("FedOMG configuration:", asdict(self.fedomg_config))

    def aggregate_model(self) -> OrderedDict:
        sample_counts = torch.tensor(
            [len(client.train_loader.dataset) for client in self.client_list],
            dtype=torch.float32,
        )
        sample_weights = sample_counts / sample_counts.sum()
        global_parameters = dict(self.classification_model.named_parameters())
        client_parameters = [dict(client.classification_model.named_parameters())
                             for client in self.client_list]
        names = list(global_parameters)
        domain_clients = OrderedDict()
        for client_id, client in enumerate(self.client_list):
            domain = client.dataset.client_data[client.client_id]["domain"][0]
            domain_clients.setdefault(domain, []).append(client_id)
        updates = []
        for client_ids in domain_clients.values():
            domain_mass = sample_weights[client_ids].sum()
            within_domain = sample_weights[client_ids] / domain_mass
            pieces = []
            for name in names:
                domain_parameter = sum(
                    client_parameters[client_id][name].detach().cpu()
                    * within_domain[position]
                    for position, client_id in enumerate(client_ids)
                )
                pieces.append(
                    (domain_parameter - global_parameters[name].detach().cpu()).reshape(-1)
                )
            # FedOMG-DG treats each source domain as one optimization task.
            updates.append(torch.cat(pieces) * domain_mass)
        aggregate, task_weights, objective = matching_gradient(
            torch.stack(updates), self.fedomg_config
        )

        state = OrderedDict(
            (name, value.detach().cpu().clone())
            for name, value in self.classification_model.state_dict().items()
        )
        offset = 0
        for name in names:
            parameter = global_parameters[name]
            count = parameter.numel()
            state[name] = parameter.detach().cpu() + (
                aggregate[offset : offset + count].reshape(parameter.shape)
                * self.fedomg_config.meta_lr
            )
            offset += count
        if offset != aggregate.numel():
            raise RuntimeError("FedOMG parameter-vector length mismatch")

        parameter_names = set(names)
        client_states = [
            client.classification_model.state_dict() for client in self.client_list
        ]
        for name in state.keys() - parameter_names:
            averaged = sum(
                client_state[name].detach().cpu() * sample_weights[client_id]
                for client_id, client_state in enumerate(client_states)
            )
            state[name] = averaged.to(dtype=state[name].dtype)

        self.agg_weight = task_weights.tolist()
        self.logger.log(
            f"{local_time()}, FedOMG Aggregation, "
            f"task_weights={[round(value, 6) for value in self.agg_weight]}, "
            f"objective={objective:.6f}"
        )
        return state
