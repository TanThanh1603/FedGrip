"""FedGRIP: FedOMG aggregation, CQT, and unified HCPP client adaptation."""
from copy import deepcopy
from dataclasses import fields
import math
import pickle
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from algorithm.aggregation.fedgrip import (
    FedGRIPConfig, matching_gradient_diagnostics,
)
from algorithm.aggregation.fedomg import FedOMGConfig
from algorithm.client.fedgrip import FedGRIPClient
from algorithm.server.fedavg import FedAvgServer
from algorithm.server.fedomg import get_fedomg_argparser
from data.dataset import FLDataset
from experiment_checkpoints import (
    atomic_output, capture_rng, cpu_tree, digest, restore_rng, write_json,
)
from fdg_round_checkpoints import owner_state, restore_owner
from utils.tools import local_time


def get_fedgrip_argparser():
    parser = get_fedomg_argparser()
    for field in fields(FedGRIPConfig):
        parser.add_argument(
            "--grip_" + field.name, type=type(field.default), default=field.default
        )
    return parser


class FedGRIPServer(FedAvgServer):
    def initialize_model(self):
        self.grip_config = FedGRIPConfig.from_args(self.args)
        self.linear_probe_rounds = int(math.ceil(
            self.args.round * self.grip_config.linear_probe_ratio
        ))
        self.fedomg_config = FedOMGConfig(
            meta_lr=self.args.fedomg_meta_lr,
            cagrad_c=self.args.fedomg_cagrad_c,
            optimizer_lr=self.args.fedomg_optimizer_lr,
            optimizer_steps=self.args.fedomg_optimizer_steps,
            momentum=self.args.fedomg_momentum,
        )
        self.fedomg_config.validate()
        if not self.args.model.startswith("mobile"):
            raise ValueError("FedGRIP requires a MobileNet backbone")
        if self.args.test_gap != 1:
            raise ValueError("FedGRIP requires round-dense evaluation")
        super().initialize_model()

    def initialize_clients(self):
        self.client_list = [
            FedGRIPClient(self.args, FLDataset(self.args, index), index, self.logger)
            for index in range(self.num_client)
        ]
        self.client_domains = []
        for client in self.client_list:
            domains = set(client.dataset.client_data[client.client_id]["domain"])
            if len(domains) != 1 or self.args.test_domain in domains:
                raise ValueError(
                    "FedGRIP requires single-source clients without target data"
                )
            self.client_domains.append(next(iter(domains)))
        self.domains = list(dict.fromkeys(self.client_domains))
        if len(self.domains) < 2:
            raise ValueError("FedGRIP requires at least two source domains")
        validation_domains = self.validation_set.client_data["validation"]["domain"]
        self.validation_domain_indices = {
            domain: [
                index for index, value in enumerate(validation_domains)
                if value == domain
            ]
            for domain in dict.fromkeys(validation_domains)
        }

    def distribute_style_banks(self):
        sketches = [client.compute_style_sketch() for client in self.client_list]
        for client, own_domain in zip(self.client_list, self.client_domains):
            peer = [
                sketch for sketch, source in zip(sketches, self.client_domains)
                if source != own_domain
            ]
            client.download_style_bank(torch.cat(peer, dim=0))

    def _domain_states(self):
        states = [client.get_model_weights() for client in self.client_list]
        result = []
        for domain in self.domains:
            indices = [
                index for index, source in enumerate(self.client_domains)
                if source == domain
            ]
            sizes = torch.tensor(
                [len(self.client_list[index].dataset) for index in indices],
                dtype=torch.float64,
            )
            weights = sizes / sizes.sum()
            state = {}
            for key, value in states[0].items():
                if value.is_floating_point():
                    state[key] = sum(
                        states[index][key].detach().cpu() * float(weight)
                        for index, weight in zip(indices, weights)
                    )
                else:
                    state[key] = torch.stack([
                        states[index][key].detach().cpu() for index in indices
                    ]).max(0).values
            result.append(state)
        return result

    def _domain_masses(self):
        counts = torch.tensor([
            sum(
                len(client.dataset)
                for client, source in zip(self.client_list, self.client_domains)
                if source == domain
            )
            for domain in self.domains
        ], dtype=torch.float32)
        return counts / counts.sum()

    def _parameter_layout(self):
        names = list(dict(self.classification_model.named_parameters()))
        total = sum(
            dict(self.classification_model.named_parameters())[name].numel()
            for name in names
        )
        return names, [], total

    @staticmethod
    def _mixed_buffers(old, domain_states, weights, parameter_names):
        mixed = {}
        for key, value in old.items():
            if key in parameter_names:
                continue
            if key.endswith("running_mean"):
                variance_key = key[:-len("running_mean")] + "running_var"
                means = torch.stack([state[key].double() for state in domain_states])
                variances = torch.stack([
                    state[variance_key].double() for state in domain_states
                ])
                shape = (len(domain_states),) + (1,) * (means.ndim - 1)
                weight = weights.double().reshape(shape)
                mean = (weight * means).sum(0)
                variance = (
                    weight * (variances + means.square())
                ).sum(0) - mean.square()
                mixed[key] = mean.to(value.dtype)
                mixed[variance_key] = variance.clamp_min(0).to(
                    old[variance_key].dtype
                )
            elif key.endswith("running_var"):
                continue
            elif value.is_floating_point():
                mixed[key] = sum(
                    state[key] * float(weight)
                    for state, weight in zip(domain_states, weights)
                ).to(value.dtype)
            else:
                mixed[key] = torch.stack([
                    state[key] for state in domain_states
                ]).max(0).values.to(value.dtype)
        return mixed

    @torch.no_grad()
    def aggregate_model(self):
        old = deepcopy(self.classification_model.state_dict())
        domain_states = self._domain_states()
        names, _, total = self._parameter_layout()
        matrix = torch.stack([
            torch.cat([
                (state[name] - old[name]).reshape(-1).float() for name in names
            ])
            for state in domain_states
        ])
        if matrix.shape[1] != total:
            raise RuntimeError("FedGRIP parameter-vector length mismatch")
        if not bool(torch.isfinite(matrix).all()):
            raise FloatingPointError("Non-finite FedGRIP client update")
        masses = self._domain_masses()
        aggregate, domain_weights, diagnostics = matching_gradient_diagnostics(
            matrix * masses.unsqueeze(1), self.fedomg_config
        )
        update = self.fedomg_config.meta_lr * aggregate
        result = deepcopy(old)
        offset = 0
        for name in names:
            count = old[name].numel()
            result[name] = old[name] + update[
                offset:offset + count
            ].reshape_as(old[name])
            offset += count
        result.update(self._mixed_buffers(
            old, domain_states, masses, set(names)
        ))
        self.agg_weight = []
        for client, source in zip(self.client_list, self.client_domains):
            domain_index = self.domains.index(source)
            members = [
                member for member, member_source in zip(
                    self.client_list, self.client_domains
                ) if member_source == source
            ]
            total_samples = sum(len(member.dataset) for member in members)
            self.agg_weight.append(
                float(domain_weights[domain_index])
                * len(client.dataset) / total_samples
            )
        diagnostics.update(
            domains=self.domains,
            buffer_weighting="sample_mass",
            buffer_weights=[float(value) for value in masses],
            effective_client_weights=self.agg_weight,
            client_training=[
                dict(client_id=client.client_id, domain=domain,
                     **client.training_diagnostics)
                for client, domain in zip(self.client_list, self.client_domains)
            ],
        )
        self.route_diagnostics = diagnostics
        self.logger.log(
            f"{local_time()}, FedGRIP weights: "
            f"domain_weights={[round(x, 6) for x in diagnostics['domain_weights']]}, "
            f"buffer_weights={[round(x, 6) for x in diagnostics['buffer_weights']]}, "
            f"entropy={diagnostics['weight_entropy']:.6f}, "
            f"objective={diagnostics['fedomg_objective']:.6f}, "
            f"conflict_fraction={diagnostics['conflict_fraction']:.6f}"
        )
        return result

    def validate_model(self):
        self.classification_model.eval()
        self.classification_model.to(self.device)
        source = {}
        total_loss = total_correct = total_count = 0
        with torch.no_grad():
            for domain, indices in self.validation_domain_indices.items():
                loss_sum = correct = count = 0
                loader = DataLoader(
                    Subset(self.validation_set, indices),
                    batch_size=self.args.batch_size, shuffle=False,
                )
                for data, target in loader:
                    data, target = data.to(self.device), target.to(self.device)
                    logits = self.classification_model(data)
                    loss_sum += float(F.cross_entropy(
                        logits, target, reduction="sum"
                    ))
                    correct += int((logits.argmax(1) == target).sum())
                    count += target.numel()
                source[domain] = dict(
                    loss=loss_sum / count,
                    accuracy=100 * correct / count,
                    samples=count,
                )
                total_loss += loss_sum
                total_correct += correct
                total_count += count
        source["overall"] = dict(
            loss=total_loss / total_count,
            accuracy=100 * total_correct / total_count,
            samples=total_count,
        )
        target_accuracy = self.evaluate(self.test_set)
        self.classification_model.cpu()
        self.logger.log(f"{local_time()}, Source validation diagnostics: {source}")
        self.logger.log(
            f"{local_time()}, Test, Accuracy: {target_accuracy:.2f}%"
        )
        return source, target_accuracy


def _save(path, signature, next_round, server, history):
    record = dict(
        signature=signature, torch_version=str(torch.__version__),
        next_round=next_round, server=owner_state(server),
        clients=[owner_state(client) for client in server.client_list],
        ids=[client.client_id for client in server.client_list],
        history=history, rng=capture_rng(),
    )
    with atomic_output(path) as stream:
        torch.save(record, stream)


def _load(path, signature, server):
    record = torch.load(path, map_location="cpu", weights_only=False)
    if (record["signature"] != signature
            or record["torch_version"] != str(torch.__version__)):
        raise ValueError("Checkpoint configuration/code/PyTorch mismatch")
    if record["ids"] != [client.client_id for client in server.client_list]:
        raise ValueError("Checkpoint clients mismatch")
    restore_owner(server, record["server"])
    for client, state in zip(server.client_list, record["clients"]):
        restore_owner(client, state)
        client.device = None
    restore_rng(record["rng"])
    return record["next_round"], record["history"]


def train(server, job, directory):
    checkpoint, signature = directory / "round_checkpoint.pt", digest(job)
    if checkpoint.exists():
        start, history = _load(checkpoint, signature, server)
    else:
        initial = deepcopy(server.classification_model.state_dict())
        for client in server.client_list:
            client.load_model_weights(initial)
        server.best_accuracy = 0
        start, history = 0, []
        _save(checkpoint, signature, start, server, history)
    server.logger.log(
        f"START/RESUME FedGRIP after {start}/{server.args.round} rounds"
    )
    for round_id in range(start, server.args.round):
        server.round_id = round_id
        begin = time.monotonic()
        server.logger.log(f"==================== Round {round_id} ====================")
        server.distribute_style_banks()
        for client in server.client_list:
            client.round_id = round_id
            client.train()
        weights = server.aggregate_model()
        server.classification_model.load_state_dict(weights)
        for client in server.client_list:
            client.load_model_weights(weights)
        source, test_accuracy = server.validate_model()
        if not math.isfinite(source["overall"]["loss"]):
            raise FloatingPointError(
                f"Non-finite source-validation loss at round {round_id}"
            )
        server.route_diagnostics["source_validation"] = source
        history.append(dict(
            round=round_id, test_accuracy=float(test_accuracy),
            route=server.route_diagnostics,
            seconds_before_checkpoint=time.monotonic() - begin,
        ))
        _save(checkpoint, signature, round_id + 1, server, history)
        write_json(directory / "round_metrics.json", history)
        server.logger.log(
            f"Checkpoint saved: completed {round_id + 1}/{server.args.round} rounds"
        )

    accuracy = float(history[-1]["test_accuracy"])
    server.best_accuracy = accuracy
    state = cpu_tree(server.classification_model.state_dict())
    with atomic_output(directory / "official_model.pt") as stream:
        torch.save(state, stream)
    with open(server.path2output_dir + "/test_accuracy.pkl", "wb") as stream:
        pickle.dump(accuracy, stream)
    server.logger.log(f"{local_time()}, Official Test, Accuracy: {accuracy:.2f}%")
    write_json(directory / "complete.json", dict(
        signature=signature, next_round=server.args.round,
        history=history, accuracy=accuracy,
        final_accuracy=accuracy,
        peak_accuracy=max(float(item["test_accuracy"]) for item in history),
        model_selection="final_round",
        buffer_weighting="sample_mass",
        bn_policy="phase_consistent_frozen_statistics",
    ))
