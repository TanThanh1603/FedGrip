"""FedAvg backbone-control baselines with resumable round execution."""
from copy import deepcopy
from dataclasses import fields
import math
import pickle
import time

import torch

from algorithm.aggregation.fedavg_backbone_control import BackboneControlConfig
from algorithm.aggregation.reversible_tgba import reversible_tgba_update
from algorithm.client.fedavg_backbone_control import (
    VARIANTS, FedAvgBackboneControlClient,
)
from algorithm.server.fedavg import FedAvgServer, get_fedavg_argparser
from data.dataset import FLDataset
from experiment_checkpoints import (
    atomic_output, capture_rng, cpu_tree, digest, restore_rng, write_json,
)
from fdg_round_checkpoints import owner_state, restore_owner
from utils.tools import local_time


CANDIDATE_GATES = (0.25, 0.5, 0.75, 1.0)


def rounds_per_probe(rounds):
    """Use P=(communication rounds / 10)*2 as the probe interval."""
    if int(rounds) < 1:
        raise ValueError("Communication rounds must be positive")
    return max(1, int(int(rounds) / 10 * 2))


def probe_event_budget(rounds):
    """Return the resulting maximum number of periodic probe events."""
    interval = rounds_per_probe(rounds)
    return int(math.ceil(int(rounds) / interval))


def get_backbone_control_argparser():
    parser = get_fedavg_argparser()
    parser.add_argument(
        "--backbone_control_variant", choices=VARIANTS, default="lpft"
    )
    for field in fields(BackboneControlConfig):
        parser.add_argument(
            "--control_" + field.name,
            type=type(field.default),
            default=field.default,
        )
    return parser


class FedAvgBackboneControlServer(FedAvgServer):
    """Use FedAvg aggregation with one controlled backbone strategy."""

    def initialize_model(self):
        self.control_config = BackboneControlConfig.from_args(self.args)
        self.control_variant = self.args.backbone_control_variant
        if self.control_variant not in VARIANTS:
            raise ValueError(f"Unknown backbone control: {self.control_variant}")
        if self.args.mu_on or self.args.augment:
            raise ValueError("Backbone-control comparison requires plain FedAvg")
        FedAvgServer.initialize_model(self)
        if not hasattr(self.classification_model, "base"):
            raise ValueError("Backbone-control comparison requires model.base")

        parameters = dict(self.classification_model.named_parameters())
        self.control_parameter_names = list(parameters)
        self.control_initial_parameters = torch.cat([
            parameters[name].detach().cpu().reshape(-1).float()
            for name in self.control_parameter_names
        ])
        self.control_backbone_mask = torch.cat([
            torch.full(
                (parameters[name].numel(),),
                not name.startswith("classifier."),
                dtype=torch.bool,
            )
            for name in self.control_parameter_names
        ])
        self.tgba_gate = 0.0
        self.tgba_phase = "head_calibration"
        self.tgba_accepted_gate = 0.0
        self.tgba_probe_pending = False
        self.tgba_probe_events = 0
        self.tgba_alignment_sums = {
            gate: 0.0 for gate in CANDIDATE_GATES
        }
        self.route_diagnostics = {}

    def initialize_dataset(self):
        super().initialize_dataset()
        domains = self.validation_set.client_data["validation"]["domain"]
        if self.args.test_domain in domains:
            raise ValueError("TGBA gate control must not contain target data")
        self.validation_domains = list(dict.fromkeys(domains))

    def initialize_clients(self):
        self.client_list = [
            FedAvgBackboneControlClient(
                self.args, FLDataset(self.args, index), index, self.logger
            )
            for index in range(self.num_client)
        ]
        self.client_domains = []
        for client in self.client_list:
            domains = set(client.dataset.client_data[client.client_id]["domain"])
            if len(domains) != 1 or self.args.test_domain in domains:
                raise ValueError(
                    "Backbone-control comparison requires single-source clients"
                )
            self.client_domains.append(next(iter(domains)))
        self.domains = list(dict.fromkeys(self.client_domains))
        if len(self.domains) < 2:
            raise ValueError("TGBA agreement requires at least two source domains")
        if set(self.validation_domains) != set(self.domains):
            raise ValueError(
                "LODO source domains must exactly match validation domains"
            )

    def _project_candidate(self, candidate, old):
        parameters = dict(self.classification_model.named_parameters())
        vector = torch.cat([
            candidate[name].reshape(-1).float()
            for name in self.control_parameter_names
        ])
        initial = self.control_initial_parameters
        backbone = self.control_backbone_mask
        displacement = vector[backbone] - initial[backbone]
        radius = (
            self.control_config.max_relative_drift
            * initial[backbone].norm().clamp_min(1e-12)
        )
        if displacement.norm() > radius:
            vector[backbone] = initial[backbone] + displacement * (
                radius / displacement.norm().clamp_min(1e-12)
            )
        result = deepcopy(candidate)
        offset = 0
        for name in self.control_parameter_names:
            count = parameters[name].numel()
            result[name] = vector[offset:offset + count].reshape_as(old[name])
            offset += count
        return result

    @staticmethod
    def _state_key_active_at_gate(name, gate):
        if name.startswith("classifier."):
            return True
        if name.startswith("base.classifier."):
            return gate >= 0.25
        prefix = "base.features."
        if not name.startswith(prefix):
            return False
        block = int(name[len(prefix):].split(".", 1)[0])
        if gate >= 1.0:
            return True
        if gate >= 0.75:
            return block >= 7
        if gate >= 0.5:
            return block >= 13
        return False

    def _masked_aggregate_candidate(
        self, old, client_states, sample_counts, gate, heldout=None,
    ):
        keep = [
            index for index, domain in enumerate(self.client_domains)
            if domain != heldout
        ]
        total = sum(sample_counts[index] for index in keep)
        candidate = {}
        for name in old:
            active = self._state_key_active_at_gate(name, gate)
            if active and old[name].is_floating_point():
                candidate[name] = sum(
                    client_states[index][name]
                    * (sample_counts[index] / total)
                    for index in keep
                )
            else:
                candidate[name] = old[name].detach().clone()
        return self._project_candidate(candidate, old)

    def _gate_parameter_mask(self, gate):
        parameters = dict(self.classification_model.named_parameters())
        return torch.cat([
            torch.full(
                (parameters[name].numel(),),
                self._state_key_active_at_gate(name, gate),
                dtype=torch.bool,
            )
            for name in self.control_parameter_names
        ])

    @torch.no_grad()
    def _lodo_alignment_candidates(self, domain_updates, domain_counts):
        """Score norm-matched gate directions by worst LODO transfer."""
        candidates = {
            gate: dict(fold_alignments={}, norm_scales={})
            for gate in CANDIDATE_GATES
        }
        for heldout_index, heldout in enumerate(self.domains):
            keep = [
                index for index in range(len(self.domains))
                if index != heldout_index
            ]
            total = sum(domain_counts[index] for index in keep)
            full_update = sum(
                domain_updates[index] * (domain_counts[index] / total)
                for index in keep
            )
            heldout_update = domain_updates[heldout_index]
            full_backbone = full_update[self.control_backbone_mask]
            heldout_backbone = heldout_update[self.control_backbone_mask]
            full_norm = full_backbone.norm()
            heldout_norm = heldout_backbone.norm()
            for gate in CANDIDATE_GATES:
                mask = self._gate_parameter_mask(gate)[
                    self.control_backbone_mask
                ]
                masked = full_backbone * mask
                masked_norm = masked.norm()
                if float(masked_norm) <= 1e-12 or float(heldout_norm) <= 1e-12:
                    alignment = -1.0
                    scale = 0.0
                else:
                    scale = float(full_norm / masked_norm.clamp_min(1e-12))
                    norm_matched = masked * scale
                    alignment = float(
                        heldout_backbone.dot(norm_matched)
                        / (
                            heldout_norm * norm_matched.norm().clamp_min(1e-12)
                        )
                    )
                candidates[gate]["fold_alignments"][heldout] = alignment
                candidates[gate]["norm_scales"][heldout] = scale

        next_count = self.tgba_probe_events + 1
        for gate, record in candidates.items():
            score = min(record["fold_alignments"].values())
            self.tgba_alignment_sums[gate] += score
            record.update(
                score=float(score),
                cumulative_score=float(
                    self.tgba_alignment_sums[gate] / next_count
                ),
                probe_observations=next_count,
                norm_reference="full_backbone_update_per_lodo_fold",
            )
        return candidates

    def _select_gate(self, candidates):
        best_score = max(
            record["cumulative_score"] for record in candidates.values()
        )
        tolerance = self.control_config.selection_margin
        eligible = [
            gate for gate, record in candidates.items()
            if record["cumulative_score"] >= best_score - tolerance
        ]
        selected = min(eligible)
        return selected, dict(
            best_cumulative_score=float(best_score),
            tie_tolerance=float(tolerance),
            eligible_gates=[float(gate) for gate in sorted(eligible)],
        )

    def _domain_parameter_updates(self, client_states, old, sample_counts):
        updates = []
        domain_counts = []
        for domain in self.domains:
            members = [
                index for index, value in enumerate(self.client_domains)
                if value == domain
            ]
            total = sum(sample_counts[index] for index in members)
            domain_counts.append(total)
            state = {
                name: sum(
                    client_states[index][name]
                    * (sample_counts[index] / total)
                    for index in members
                )
                for name in self.control_parameter_names
            }
            updates.append(torch.cat([
                (state[name] - old[name]).reshape(-1).float()
                for name in self.control_parameter_names
            ]))
        return torch.stack(updates), domain_counts

    @torch.no_grad()
    def aggregate_model(self):
        if self.control_variant != "tgba":
            result = super().aggregate_model()
            round_flops = sum(
                client.training_diagnostics.get("estimated_training_flops", 0)
                for client in self.client_list
            )
            self.route_diagnostics = dict(
                controller=self.control_variant,
                effective_client_weights=[float(x) for x in self.agg_weight],
                estimated_round_training_flops=int(round_flops),
                training_flops_method=(
                    "torch_flop_counter_forward_backward_plus_"
                    "analytical_l2sp"
                ),
                client_training=[
                    dict(
                        client_id=client.client_id,
                        domain=domain,
                        **client.training_diagnostics,
                    )
                    for client, domain in zip(
                        self.client_list, self.client_domains
                    )
                ],
            )
            self.logger.log(
                f"{local_time()}, FedAvg+{self.control_variant}: "
                f"estimated training FLOPs={round_flops / 1e12:.4f}T"
            )
            return result

        old = deepcopy(self.classification_model.state_dict())
        client_states = [
            client.get_model_weights() for client in self.client_list
        ]
        sample_counts = [
            len(client.train_loader.dataset) for client in self.client_list
        ]
        total_samples = sum(sample_counts)
        self.agg_weight = [count / total_samples for count in sample_counts]
        averaged = {}
        for name in client_states[0]:
            if client_states[0][name].is_floating_point():
                averaged[name] = sum(
                    state[name] * weight
                    for state, weight in zip(client_states, self.agg_weight)
                )
            else:
                averaged[name] = old[name].detach().clone()

        parameters = dict(self.classification_model.named_parameters())
        if list(parameters) != self.control_parameter_names:
            raise RuntimeError("Backbone parameter layout changed")
        raw_domain_updates, domain_counts = self._domain_parameter_updates(
            client_states, old, sample_counts
        )
        fedavg_update = torch.cat([
            (averaged[name] - old[name]).reshape(-1).float()
            for name in self.control_parameter_names
        ])
        current = torch.cat([
            old[name].reshape(-1).float()
            for name in self.control_parameter_names
        ])
        applied_gate = float(self.tgba_gate)
        was_probe = bool(self.tgba_probe_pending)
        update, diagnostics = reversible_tgba_update(
            raw_domain_updates,
            fedavg_update,
            current,
            self.control_initial_parameters,
            self.control_backbone_mask,
            self.control_config,
        )
        controller = dict(
            gate=applied_gate,
            phase=(
                "full_finetune" if applied_gate == 1
                else ("head_calibration" if applied_gate == 0
                      else "gradual_unfreezing")
            ),
        )

        result = deepcopy(old)
        offset = 0
        for name in self.control_parameter_names:
            count = old[name].numel()
            result[name] = old[name] + update[
                offset:offset + count
            ].reshape_as(old[name])
            offset += count
        parameter_names = set(self.control_parameter_names)
        for name in result.keys() - parameter_names:
            if old[name].is_floating_point():
                result[name] = averaged[name]
            else:
                result[name] = old[name].detach().clone()

        lodo = None
        lodo_candidates = None
        evaluation_flops = 0
        transition = "hold"
        warmup_rounds = int(math.ceil(
            self.args.round * self.control_config.linear_probe_ratio
        ))
        next_round = self.round_id + 1
        if was_probe:
            lodo_candidates = self._lodo_alignment_candidates(
                raw_domain_updates, domain_counts
            )
            selected_gate, selection = self._select_gate(lodo_candidates)
            selected_record = lodo_candidates[selected_gate]
            result = self._masked_aggregate_candidate(
                old, client_states, sample_counts, selected_gate
            )
            transition = "norm_matched_lodo_gate_selected"
            lodo = dict(
                selected_record,
                role="selected",
                selected_gate=float(selected_gate),
                rollback=False,
                **selection,
            )
            self.tgba_accepted_gate = float(selected_gate)
            self.tgba_probe_events += 1
            self.tgba_probe_pending = False
            controller["gate"] = float(selected_gate)
            controller["phase"] = (
                "full_finetune" if selected_gate == 1
                else "gradual_unfreezing"
            )
        elif self.round_id < warmup_rounds:
            self.tgba_accepted_gate = 0.0
            controller["gate"] = 0.0
            controller["phase"] = "head_calibration"
            transition = "linear_probe_warmup"
            if next_round == warmup_rounds and next_round < self.args.round:
                self.tgba_probe_pending = True
                controller["gate"] = 1.0
                controller["phase"] = "full_finetune_probe"
                transition = "periodic_probe_scheduled_after_warmup"
        else:
            self.tgba_accepted_gate = applied_gate
            controller["gate"] = applied_gate
            controller["phase"] = (
                "full_finetune" if applied_gate == 1
                else "gradual_unfreezing"
            )
            periodic_probe_due = (
                next_round < self.args.round
                and next_round >= warmup_rounds
                and (next_round - warmup_rounds)
                % rounds_per_probe(self.args.round) == 0
            )
            if periodic_probe_due:
                self.tgba_probe_pending = True
                controller["gate"] = 1.0
                controller["phase"] = "full_finetune_probe"
                transition = "periodic_probe_scheduled"

        diagnostics["transition"] = transition
        diagnostics["applied_backbone_gate"] = applied_gate
        diagnostics["update_committed"] = True
        if was_probe:
            committed = torch.cat([
                (result[name] - old[name]).reshape(-1).float()
                for name in self.control_parameter_names
            ])
            initial_backbone = self.control_initial_parameters[
                self.control_backbone_mask
            ]
            result_backbone = torch.cat([
                result[name].reshape(-1).float()
                for name in self.control_parameter_names
            ])[self.control_backbone_mask]
            diagnostics["committed_backbone_update_norm"] = float(
                committed[self.control_backbone_mask].norm()
            )
            diagnostics["head_update_norm"] = float(
                committed[~self.control_backbone_mask].norm()
            )
            diagnostics["pretrained_backbone_relative_distance_after"] = float(
                (result_backbone - initial_backbone).norm()
                / initial_backbone.norm().clamp_min(1e-12)
            )
        diagnostics["next_backbone_gate"] = float(controller["gate"])
        diagnostics["accepted_backbone_gate"] = float(
            self.tgba_accepted_gate
        )
        diagnostics["probe_pending"] = bool(self.tgba_probe_pending)
        diagnostics["probe_events"] = int(self.tgba_probe_events)
        diagnostics["lodo"] = lodo
        diagnostics["lodo_candidates"] = lodo_candidates
        self.tgba_gate = float(controller["gate"])
        self.tgba_phase = controller["phase"]

        round_flops = int(sum(
            client.training_diagnostics.get("estimated_training_flops", 0)
            for client in self.client_list
        ))
        self.route_diagnostics = dict(
            diagnostics,
            controller="tgba",
            aggregation="sample_weighted_fedavg",
            domains=self.domains,
            effective_client_weights=[float(x) for x in self.agg_weight],
            estimated_round_training_flops=round_flops,
            estimated_lodo_evaluation_flops=int(evaluation_flops),
            estimated_round_total_flops=int(round_flops + evaluation_flops),
            training_flops_method=(
                "torch_flop_counter_forward_backward_plus_analytical_l2sp"
            ),
            client_training=[
                dict(
                    client_id=client.client_id,
                    domain=domain,
                    **client.training_diagnostics,
                )
                for client, domain in zip(
                    self.client_list, self.client_domains
                )
            ],
        )
        self.logger.log(
            f"{local_time()}, FedAvg+TGBA: "
            f"weights={[round(x, 6) for x in self.agg_weight]}, "
            f"head_cos={diagnostics['head_update_cosine']:.6f}, "
            f"backbone_cos={diagnostics['backbone_update_cosine']:.6f}, "
            f"conflict={diagnostics['conflict_detected']}, "
            f"gate={diagnostics['applied_backbone_gate']:.2f}->"
            f"{diagnostics['next_backbone_gate']:.2f}, "
            f"transition={diagnostics['transition']}, "
            f"train_flops={round_flops / 1e12:.4f}T, "
            f"lodo_eval_flops={evaluation_flops / 1e12:.4f}T, "
            f"distance="
            f"{diagnostics['pretrained_backbone_relative_distance_after']:.6f}, "
            f"projected={diagnostics['trust_region_projected']}"
        )
        if lodo is not None:
            self.logger.log(
                f"{local_time()}, TGBA norm-matched LODO alignment: "
                f"role={lodo['role']}, "
                f"fold_alignments="
                f"{ {key: round(value, 6) for key, value in lodo['fold_alignments'].items()} }, "
                f"score={lodo['score']:.6f}, "
                f"cumulative_score={lodo['cumulative_score']:.6f}, "
                f"selected_gate={lodo.get('selected_gate', self.tgba_accepted_gate):.2f}, "
                f"target_data_used=False"
            )
        if lodo_candidates is not None:
            self.logger.log(
                f"{local_time()}, TGBA cumulative gate scores: "
                f"{ {gate: round(record['cumulative_score'], 6) for gate, record in lodo_candidates.items()} }, "
                f"selected={self.tgba_accepted_gate:.2f}"
            )
        return result


def _controller_state(server):
    return dict(
        gate=server.tgba_gate,
        accepted_gate=server.tgba_accepted_gate,
        phase=server.tgba_phase,
        probe_pending=server.tgba_probe_pending,
        probe_events=server.tgba_probe_events,
        alignment_sums=cpu_tree(server.tgba_alignment_sums),
    )


def _save(path, signature, next_round, server, history):
    record = dict(
        signature=signature,
        torch_version=str(torch.__version__),
        next_round=next_round,
        server=owner_state(server),
        clients=[owner_state(client) for client in server.client_list],
        ids=[client.client_id for client in server.client_list],
        history=history,
        controller=_controller_state(server),
        rng=capture_rng(),
    )
    with atomic_output(path) as stream:
        torch.save(record, stream)


def _load(path, signature, server):
    record = torch.load(path, map_location="cpu", weights_only=False)
    if (
        record["signature"] != signature
        or record["torch_version"] != str(torch.__version__)
    ):
        raise ValueError("Checkpoint configuration/code/PyTorch mismatch")
    if record["ids"] != [client.client_id for client in server.client_list]:
        raise ValueError("Checkpoint clients mismatch")
    restore_owner(server, record["server"])
    for client, state in zip(server.client_list, record["clients"]):
        restore_owner(client, state)
        client.device = None
    controller = record["controller"]
    if server.control_variant == "tgba":
        for name in (
            "gate", "accepted_gate", "phase", "probe_pending",
            "probe_events", "alignment_sums",
        ):
            setattr(server, "tgba_" + name, controller[name])
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
        f"START/RESUME {server.control_variant} after "
        f"{start}/{server.args.round} rounds"
    )
    for round_id in range(start, server.args.round):
        server.round_id = round_id
        begin = time.monotonic()
        server.logger.log(f"==================== Round {round_id} ====================")
        for client in server.client_list:
            client.round_id = round_id
            client.tgba_gate = (
                server.tgba_gate if server.control_variant == "tgba" else 1.0
            )
            client.train()
        weights = server.aggregate_model()
        previous_flops = (
            history[-1]["route"].get(
                "cumulative_estimated_training_flops", 0
            ) if history else 0
        )
        server.route_diagnostics[
            "cumulative_estimated_training_flops"
        ] = int(
            previous_flops
            + server.route_diagnostics["estimated_round_training_flops"]
        )
        previous_total_flops = (
            history[-1]["route"].get(
                "cumulative_estimated_total_flops",
                history[-1]["route"].get(
                    "cumulative_estimated_training_flops", 0
                ),
            ) if history else 0
        )
        server.route_diagnostics[
            "cumulative_estimated_total_flops"
        ] = int(
            previous_total_flops
            + server.route_diagnostics.get(
                "estimated_round_total_flops",
                server.route_diagnostics["estimated_round_training_flops"],
            )
        )
        server.classification_model.load_state_dict(weights)
        for client in server.client_list:
            client.load_model_weights(weights)
        server.validate_and_test()
        history.append(dict(
            round=round_id,
            test_accuracy=float(server.best_accuracy),
            weights=[float(value) for value in server.agg_weight],
            route=server.route_diagnostics,
            seconds_before_checkpoint=time.monotonic() - begin,
        ))
        _save(checkpoint, signature, round_id + 1, server, history)
        write_json(directory / "round_metrics.json", history)
        server.logger.log(
            f"Checkpoint saved: completed {round_id + 1}/"
            f"{server.args.round} rounds"
        )

    state = cpu_tree(server.classification_model.state_dict())
    with atomic_output(directory / "final_model.pt") as stream:
        torch.save(state, stream)
    with open(directory / "test_accuracy.pkl", "wb") as stream:
        pickle.dump(float(server.best_accuracy), stream)
    write_json(directory / "complete.json", dict(
        signature=signature,
        next_round=server.args.round,
        history=history,
        accuracy=float(server.best_accuracy),
        final_controller=_controller_state(server),
        output_model="final_round_no_averaging",
        aggregation="sample_weighted_fedavg",
        cqt_enabled=False,
        gradient_matching_enabled=False,
        swad_enabled=False,
    ))
