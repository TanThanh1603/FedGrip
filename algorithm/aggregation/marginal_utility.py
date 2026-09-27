from collections import Counter, defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
import json
import os
import random
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from utils.tools import local_time


DEFAULT_PROBE_MASS = 0.10
DEFAULT_RARITY_POWER = 0.50
DEFAULT_LOSS_SHRINKAGE = 4.0
DEFAULT_COUNT_SMOOTHING = 1.0
DEFAULT_INTERACTION_STRENGTH = 0.50
DEFAULT_AP_MAD_PENALTY = 0.50
DEFAULT_AP_RELATIVE_MARGIN = 0.002
DEFAULT_UTILITY_BLEND = 0.70
DEFAULT_UTILITY_SCALE_FLOOR = 1e-3


@contextmanager
def preserve_rng_state():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def isolate_probe_rng(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        if not self.config.isolate_probe_rng:
            return method(self, *args, **kwargs)
        with preserve_rng_state():
            round_id = kwargs.get("round_id")
            if round_id is None and len(args) >= 2:
                round_id = args[1]
            server = getattr(self, "server", None)
            experiment_seed = getattr(getattr(server, "args", None), "seed", 0)
            configured_probe_seed = getattr(self.config, "probe_seed", 0)
            probe_seed = int(
                experiment_seed + configured_probe_seed + (round_id or 0)
            )
            random.seed(probe_seed)
            np.random.seed(probe_seed)
            torch.manual_seed(probe_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(probe_seed)
            return method(self, *args, **kwargs)

    return wrapped


def median_absolute_deviation(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    center = float(np.median(values))
    return float(np.median(np.abs(values - center)))


def convex_singleton_probe(prior, client_id: int, probe_mass: float):
    probe = (1.0 - probe_mass) * np.asarray(prior, dtype=np.float64)
    probe[client_id] += probe_mass
    return probe


def convex_pair_probe(prior, first: int, second: int, probe_mass: float):
    probe = (1.0 - probe_mass) * np.asarray(prior, dtype=np.float64)
    probe[first] += 0.5 * probe_mass
    probe[second] += 0.5 * probe_mass
    return probe


def robust_ap_decision(relative_advantages, penalty: float, margin: float):
    advantages = np.asarray(relative_advantages, dtype=np.float64)
    median = float(np.median(advantages))
    mad = median_absolute_deviation(advantages)
    robust_advantage = median - penalty * mad
    return robust_advantage > margin, median, mad, robust_advantage


@dataclass(frozen=True)
class DeficitInteractionUtilityConfig:
    warmup_rounds: int = 2
    validation_folds: int = 3
    max_validation_samples_per_domain: int = 384
    utility_blend: float = DEFAULT_UTILITY_BLEND
    isolate_probe_rng: bool = True
    probe_seed: int = 104729
    domain_balanced_prior: bool = False
    probe_mass: float = DEFAULT_PROBE_MASS
    rarity_power: float = DEFAULT_RARITY_POWER
    loss_shrinkage: float = DEFAULT_LOSS_SHRINKAGE
    count_smoothing: float = DEFAULT_COUNT_SMOOTHING
    interaction_strength: float = DEFAULT_INTERACTION_STRENGTH
    ap_mad_penalty: float = DEFAULT_AP_MAD_PENALTY
    ap_relative_margin: float = DEFAULT_AP_RELATIVE_MARGIN
    utility_scale_floor: float = DEFAULT_UTILITY_SCALE_FLOOR
    eps: float = 1e-8

    @staticmethod
    def from_args(args) -> "DeficitInteractionUtilityConfig":
        return DeficitInteractionUtilityConfig(
            warmup_rounds=getattr(args, "mu_warmup_rounds", 2),
            validation_folds=getattr(args, "mu_validation_folds", 3),
            max_validation_samples_per_domain=getattr(
                args, "mu_max_validation_samples_per_domain", 384
            ),
            utility_blend=getattr(
                args, "mu_utility_blend", DEFAULT_UTILITY_BLEND
            ),
            isolate_probe_rng=bool(getattr(args, "mu_isolate_probe_rng", 1)),
            probe_seed=getattr(args, "mu_probe_seed", 104729),
            domain_balanced_prior=bool(
                getattr(args, "mu_domain_balanced_prior", 0)
            ),
            probe_mass=getattr(args, "diu_probe_mass", DEFAULT_PROBE_MASS),
            rarity_power=getattr(
                args, "diu_rarity_power", DEFAULT_RARITY_POWER
            ),
            loss_shrinkage=getattr(
                args, "diu_loss_shrinkage", DEFAULT_LOSS_SHRINKAGE
            ),
            count_smoothing=getattr(
                args, "diu_count_smoothing", DEFAULT_COUNT_SMOOTHING
            ),
            interaction_strength=getattr(
                args,
                "diu_interaction_strength",
                DEFAULT_INTERACTION_STRENGTH,
            ),
            ap_mad_penalty=getattr(
                args, "diu_ap_mad_penalty", DEFAULT_AP_MAD_PENALTY
            ),
            ap_relative_margin=getattr(
                args,
                "diu_ap_relative_margin",
                DEFAULT_AP_RELATIVE_MARGIN,
            ),
            utility_scale_floor=getattr(
                args,
                "diu_utility_scale_floor",
                DEFAULT_UTILITY_SCALE_FLOOR,
            ),
        )


def add_mu_argparser(parser):
    group = parser.add_argument_group("DIU + Adaptive Prior Weighting")
    group.add_argument("--mu_warmup_rounds", type=int, default=2)
    group.add_argument("--mu_validation_folds", type=int, default=3)
    group.add_argument("--mu_max_validation_samples_per_domain", type=int, default=384)
    group.add_argument(
        "--mu_utility_blend",
        type=float,
        default=0.7,
        help="Fixed blend between the selected prior and utility-aware weights.",
    )
    group.add_argument("--mu_isolate_probe_rng", type=int, default=1)
    group.add_argument("--mu_probe_seed", type=int, default=104729)
    group.add_argument("--diu_probe_mass", type=float, default=DEFAULT_PROBE_MASS)
    group.add_argument("--diu_rarity_power", type=float, default=DEFAULT_RARITY_POWER)
    group.add_argument(
        "--diu_loss_shrinkage", type=float, default=DEFAULT_LOSS_SHRINKAGE
    )
    group.add_argument(
        "--diu_count_smoothing", type=float, default=DEFAULT_COUNT_SMOOTHING
    )
    group.add_argument(
        "--diu_interaction_strength",
        type=float,
        default=DEFAULT_INTERACTION_STRENGTH,
    )
    group.add_argument(
        "--diu_ap_mad_penalty", type=float, default=DEFAULT_AP_MAD_PENALTY
    )
    group.add_argument(
        "--diu_ap_relative_margin",
        type=float,
        default=DEFAULT_AP_RELATIVE_MARGIN,
        help="Robust relative-advantage threshold used by AP.",
    )
    group.add_argument(
        "--diu_utility_scale_floor",
        type=float,
        default=DEFAULT_UTILITY_SCALE_FLOOR,
    )
    return parser


class DeficitInteractionUtilityAggregator:
    """Official FedCMU aggregation: DIU client utility followed by AP."""

    def __init__(
        self,
        server,
        config: DeficitInteractionUtilityConfig = None,
    ):
        self.server = server
        self.config = config or DeficitInteractionUtilityConfig()
        self.source_domains = [
            domain
            for domain in server.validation_set.client_data["validation"]["domain"]
            if domain != server.args.test_domain
        ]
        self.source_domains = list(dict.fromkeys(self.source_domains))
        self.client_domain_proportions = self._get_client_domain_proportions()
        self.validation_folds = self._build_validation_folds()
        self.probe_mass = float(self.config.probe_mass)
        self.rarity_power = float(self.config.rarity_power)
        self.loss_shrinkage = float(self.config.loss_shrinkage)
        self.count_smoothing = float(self.config.count_smoothing)
        self.interaction_strength = float(self.config.interaction_strength)
        self.ap_mad_penalty = float(self.config.ap_mad_penalty)
        self.ap_relative_margin = float(self.config.ap_relative_margin)
        self.utility_scale_floor = float(self.config.utility_scale_floor)
        self._validate_diu_configuration()
        self.num_labels = len(self.server.validation_set.label_to_index)
        self.training_group_counts = self._training_group_counts()
        self.weighting_mode = "fedavg"
        self.debug_log_path = os.path.join(
            self.server.path2output_dir, "mu_debug.jsonl"
        )
        with open(self.debug_log_path, "w"):
            pass

    def _log_event(self, event: str, **payload) -> None:
        message = {
            "time": local_time(),
            "event": event,
            **payload,
        }
        json_message = json.dumps(
            message, sort_keys=True, default=lambda value: value.tolist()
        )
        with open(self.debug_log_path, "a") as debug_log:
            debug_log.write(json_message + "\n")
        self.server.logger.log(f"MU {json_message}", markup=False)

    def _get_client_domain_proportions(self) -> List[Dict[str, float]]:
        client_domain_proportions = []
        for client_id in range(self.server.num_client):
            domains = self.server.client_list[client_id].dataset.client_data[client_id][
                "domain"
            ]
            if not domains:
                raise ValueError(f"Client {client_id} has no source-domain metadata.")
            counts = {domain: domains.count(domain) for domain in set(domains)}
            total = sum(counts.values())
            client_domain_proportions.append(
                {domain: count / total for domain, count in counts.items()}
            )
        return client_domain_proportions

    def _build_validation_folds(self) -> Dict[str, List[List[int]]]:
        domains = self.server.validation_set.client_data["validation"]["domain"]
        labels = self.server.validation_set.client_data["validation"]["labels"]
        num_folds = max(self.config.validation_folds, 1)
        folds = {
            domain: [[] for _ in range(num_folds)] for domain in self.source_domains
        }
        for domain in self.source_domains:
            label_indices = {}
            for index, (sample_domain, label) in enumerate(zip(domains, labels)):
                if sample_domain == domain:
                    label_indices.setdefault(label, []).append(index)
            for indices in label_indices.values():
                for offset, index in enumerate(indices):
                    folds[domain][offset % num_folds].append(index)
            per_fold_limit = max(
                self.config.max_validation_samples_per_domain // num_folds, 1
            )
            folds[domain] = [
                self._class_balanced_limit(fold, labels, per_fold_limit)
                for fold in folds[domain]
            ]
        return folds

    def _class_balanced_limit(
        self, indices: List[int], labels: List[str], limit: int
    ) -> List[int]:
        if len(indices) <= limit:
            return indices
        label_buckets = {}
        for index in indices:
            label_buckets.setdefault(labels[index], []).append(index)
        selected = []
        offsets = {label: 0 for label in label_buckets}
        ordered_labels = sorted(label_buckets)
        while len(selected) < limit:
            added = False
            for label in ordered_labels:
                offset = offsets[label]
                bucket = label_buckets[label]
                if offset < len(bucket):
                    selected.append(bucket[offset])
                    offsets[label] += 1
                    added = True
                    if len(selected) == limit:
                        break
            if not added:
                break
        return selected

    def _validate_diu_configuration(self):
        if not 0.0 < self.probe_mass <= 1.0:
            raise ValueError("diu_probe_mass must be in (0, 1]")
        if self.rarity_power < 0.0:
            raise ValueError("diu_rarity_power must be non-negative")
        if self.loss_shrinkage < 0.0 or self.count_smoothing <= 0.0:
            raise ValueError("DIU smoothing constants must be valid")
        if self.interaction_strength < 0.0:
            raise ValueError("diu_interaction_strength must be non-negative")
        if self.ap_mad_penalty < 0.0 or self.ap_relative_margin < 0.0:
            raise ValueError("DIU AP robustness constants must be non-negative")
        if self.utility_scale_floor <= 0.0:
            raise ValueError("diu_utility_scale_floor must be positive")

    def _training_group_counts(self):
        counts = Counter()
        label_to_index = self.server.validation_set.label_to_index
        for client in self.server.client_list:
            client_id = client.client_id
            client_data = client.dataset.client_data[client_id]
            for domain, label in zip(
                client_data["domain"], client_data["labels"]
            ):
                if domain in self.source_domains:
                    counts[(domain, int(label_to_index[label]))] += 1
        return counts

    def _evaluate_group_losses(self, model):
        """Return smoothed CE for every fold/domain/class source group."""
        fold_count = len(next(iter(self.validation_folds.values())))
        fold_losses = []
        model.eval()
        model.to(self.server.device)
        try:
            with torch.no_grad():
                for fold_id in range(fold_count):
                    groups = {}
                    for domain in self.source_domains:
                        indices = self.validation_folds[domain][fold_id]
                        if not indices:
                            continue
                        loader = DataLoader(
                            Subset(self.server.validation_set, indices),
                            batch_size=self.server.args.batch_size,
                        )
                        class_loss_sums = np.zeros(
                            self.num_labels, dtype=np.float64
                        )
                        class_counts = np.zeros(
                            self.num_labels, dtype=np.int64
                        )
                        domain_loss_sum = 0.0
                        domain_count = 0
                        for data, target in loader:
                            output = model(data)
                            losses = F.cross_entropy(
                                output, target, reduction="none"
                            )
                            target_array = target.detach().cpu().numpy()
                            loss_array = losses.detach().cpu().numpy()
                            class_loss_sums += np.bincount(
                                target_array,
                                weights=loss_array,
                                minlength=self.num_labels,
                            )
                            class_counts += np.bincount(
                                target_array, minlength=self.num_labels
                            ).astype(np.int64)
                            domain_loss_sum += float(loss_array.sum())
                            domain_count += int(target_array.size)

                        domain_loss = domain_loss_sum / max(domain_count, 1)
                        for class_id, count in enumerate(class_counts):
                            if count <= 0:
                                continue
                            class_loss = class_loss_sums[class_id] / count
                            smoothed = (
                                count * class_loss
                                + self.loss_shrinkage * domain_loss
                            ) / (count + self.loss_shrinkage)
                            groups[(domain, class_id)] = float(smoothed)
                    fold_losses.append(groups)
        finally:
            model.to(torch.device("cpu"))
        return fold_losses

    @staticmethod
    def _client_state(client):
        state = client.get_model_weights()
        return state[0] if isinstance(state, (list, tuple)) else state

    def _aggregate_client_state(self, weights: np.ndarray):
        """Aggregate client classifiers with normalized client weights."""
        weights = np.asarray(weights, dtype=np.float64)
        total = float(weights.sum())
        if not np.all(np.isfinite(weights)) or total <= self.config.eps:
            raise ValueError(f"Invalid DIU probe weights: {weights}")
        weights = weights / total

        client_states = [
            self._client_state(client) for client in self.server.client_list
        ]
        global_state = self.server.classification_model.state_dict()
        aggregate = {}
        for key, global_value in global_state.items():
            if torch.is_floating_point(global_value):
                value = torch.zeros_like(global_value)
                for weight, state in zip(weights, client_states):
                    value.add_(
                        state[key].to(value.device), alpha=float(weight)
                    )
                aggregate[key] = value
            else:
                aggregate[key] = global_value.clone()
        return aggregate

    def _group_losses_for_weights(self, weights):
        model = deepcopy(self.server.classification_model)
        try:
            model.load_state_dict(self._aggregate_client_state(weights))
            return self._evaluate_group_losses(model)
        finally:
            del model

    def _balanced_fold_risk(self, fold_losses):
        domain_risks = []
        for domain in self.source_domains:
            values = [
                loss
                for (group_domain, _), loss in fold_losses.items()
                if group_domain == domain
            ]
            if values:
                domain_risks.append(float(np.mean(values)))
        if not domain_risks:
            raise ValueError("No source validation groups were available")
        return float(np.mean(domain_risks))

    def _select_adaptive_prior(self, sample_weights):
        domain_weights = self._domain_balanced_prior(sample_weights)
        sample_losses = self._group_losses_for_weights(sample_weights)
        if np.allclose(sample_weights, domain_weights, atol=1e-12):
            domain_losses = sample_losses
        else:
            domain_losses = self._group_losses_for_weights(domain_weights)

        sample_risks = [self._balanced_fold_risk(fold) for fold in sample_losses]
        domain_risks = [self._balanced_fold_risk(fold) for fold in domain_losses]
        relative_advantages = [
            (sample_risk - domain_risk) / max(sample_risk, self.config.eps)
            for sample_risk, domain_risk in zip(sample_risks, domain_risks)
        ]
        selected, median, mad, robust_advantage = robust_ap_decision(
            relative_advantages,
            self.ap_mad_penalty,
            self.ap_relative_margin,
        )
        selected = bool(self.config.domain_balanced_prior and selected)
        base_prior = domain_weights if selected else sample_weights
        base_losses = domain_losses if selected else sample_losses
        selected_prior = "domain_balanced" if selected else "sample_size"
        standard_score = float(np.median(sample_risks))
        domain_score = float(np.median(domain_risks))
        return (
            base_prior,
            base_losses,
            selected_prior,
            standard_score,
            domain_score,
        )

    def _deficit_weights(self, base_fold_losses):
        group_values = defaultdict(list)
        for fold in base_fold_losses:
            for group, loss in fold.items():
                group_values[group].append(loss)
        group_means = {
            group: float(np.mean(values))
            for group, values in group_values.items()
        }

        deficit_weights = {}
        for domain in self.source_domains:
            domain_groups = [
                group for group in group_means if group[0] == domain
            ]
            if not domain_groups:
                continue
            losses = np.asarray(
                [group_means[group] for group in domain_groups],
                dtype=np.float64,
            )
            center = float(np.median(losses))
            difficulty_scale = max(
                1.4826 * median_absolute_deviation(losses),
                self.utility_scale_floor,
            )
            raw = []
            for group in domain_groups:
                count = self.training_group_counts.get(group, 0)
                rarity = (count + self.count_smoothing) ** (-self.rarity_power)
                normalized_difficulty = (
                    group_means[group] - center
                ) / difficulty_scale
                difficulty = float(np.logaddexp(0.0, normalized_difficulty))
                raw.append(rarity * difficulty)
            raw = np.asarray(raw, dtype=np.float64)
            if not np.all(np.isfinite(raw)) or raw.sum() <= self.config.eps:
                raw = np.ones(len(domain_groups), dtype=np.float64)
            raw /= raw.sum()
            raw /= len(self.source_domains)
            for group, weight in zip(domain_groups, raw):
                deficit_weights[group] = float(weight)

        total = sum(deficit_weights.values())
        if total <= self.config.eps:
            raise ValueError("DIU produced no valid deficit weights")
        return {
            group: weight / total for group, weight in deficit_weights.items()
        }

    @staticmethod
    def _relative_group_gain(base_losses, candidate_losses, eps):
        gains = []
        for base_fold, candidate_fold in zip(base_losses, candidate_losses):
            fold_gain = {}
            for group, base_loss in base_fold.items():
                candidate_loss = candidate_fold.get(group)
                if candidate_loss is None:
                    continue
                fold_gain[group] = (
                    base_loss - candidate_loss
                ) / max(base_loss, eps)
            gains.append(fold_gain)
        return gains

    def _estimate_diu(self, base_prior, base_losses):
        num_clients = self.server.num_client
        singleton_gains = []
        for client_id in range(num_clients):
            weights = convex_singleton_probe(
                base_prior, client_id, self.probe_mass
            )
            losses = self._group_losses_for_weights(weights)
            singleton_gains.append(
                self._relative_group_gain(base_losses, losses, self.config.eps)
            )

        pair_gains = {}
        for first in range(num_clients):
            for second in range(first + 1, num_clients):
                weights = convex_pair_probe(
                    base_prior, first, second, self.probe_mass
                )
                losses = self._group_losses_for_weights(weights)
                pair_gains[(first, second)] = self._relative_group_gain(
                    base_losses, losses, self.config.eps
                )

        deficit_weights = self._deficit_weights(base_losses)
        fold_count = len(base_losses)
        composite = []
        all_composite_values = []
        for client_id in range(num_clients):
            client_folds = []
            other_mass = max(
                1.0 - float(base_prior[client_id]), self.config.eps
            )
            for fold_id in range(fold_count):
                fold_values = {}
                for group in deficit_weights:
                    singleton = singleton_gains[client_id][fold_id].get(
                        group, 0.0
                    )
                    expected_interaction = 0.0
                    for other_id in range(num_clients):
                        if other_id == client_id:
                            continue
                        pair = tuple(sorted((client_id, other_id)))
                        pair_gain = pair_gains[pair][fold_id].get(group, 0.0)
                        other_singleton = singleton_gains[other_id][fold_id].get(
                            group, 0.0
                        )
                        synergy = pair_gain - 0.5 * (
                            singleton + other_singleton
                        )
                        expected_interaction += (
                            float(base_prior[other_id]) / other_mass
                        ) * synergy
                    value = singleton + (
                        self.interaction_strength * expected_interaction
                    )
                    fold_values[group] = value
                    all_composite_values.append(value)
                client_folds.append(fold_values)
            composite.append(client_folds)

        utility_scale = max(
            1.4826 * median_absolute_deviation(all_composite_values),
            self.utility_scale_floor,
        )
        fold_utilities = np.zeros((fold_count, num_clients), dtype=np.float64)
        for client_id in range(num_clients):
            domain_proportions = self.client_domain_proportions[client_id]
            for fold_id in range(fold_count):
                numerator = 0.0
                denominator = 0.0
                for group, deficit_weight in deficit_weights.items():
                    domain_factor = 1.0 - domain_proportions.get(group[0], 0.0)
                    weight = deficit_weight * domain_factor
                    numerator += weight * np.tanh(
                        composite[client_id][fold_id][group] / utility_scale
                    )
                    denominator += weight
                fold_utilities[fold_id, client_id] = (
                    numerator / max(denominator, self.config.eps)
                )
        utilities = np.median(fold_utilities, axis=0)

        return utilities

    def _utility_distribution(
        self, base_prior: np.ndarray, utilities: np.ndarray
    ) -> np.ndarray:
        logits = np.log(base_prior + self.config.eps) + utilities
        logits -= logits.max()
        scores = np.exp(logits)
        return scores / scores.sum()

    def _domain_balanced_prior(self, sample_weights: np.ndarray) -> np.ndarray:
        balanced = np.zeros_like(sample_weights)
        domain_totals = self._domain_weight_totals(sample_weights)
        active_domains = [
            domain
            for domain, total in domain_totals.items()
            if total > self.config.eps
        ]
        if not active_domains:
            return sample_weights.copy()

        domain_weight = 1.0 / len(active_domains)
        for client_id, proportions in enumerate(self.client_domain_proportions):
            for domain in active_domains:
                client_domain_mass = (
                    sample_weights[client_id] * proportions.get(domain, 0.0)
                )
                if client_domain_mass > self.config.eps:
                    balanced[client_id] += (
                        domain_weight
                        * client_domain_mass
                        / domain_totals[domain]
                    )

        if balanced.sum() <= self.config.eps:
            return sample_weights.copy()
        return balanced / balanced.sum()

    def _domain_weight_totals(self, weights: np.ndarray) -> Dict[str, float]:
        return {
            domain: float(
                sum(
                    weight * proportions.get(domain, 0.0)
                    for weight, proportions in zip(
                        weights, self.client_domain_proportions
                    )
                )
            )
            for domain in self.source_domains
        }

    @isolate_probe_rng
    def compute(self, sample_weights: List[float], round_id: int) -> List[float]:
        sample_weights = np.asarray(sample_weights, dtype=np.float64)
        if not np.all(np.isfinite(sample_weights)) or sample_weights.sum() <= 0.0:
            raise ValueError(f"Invalid DIU sample weights: {sample_weights}")
        sample_weights /= sample_weights.sum()
        if round_id < self.config.warmup_rounds:
            self.weighting_mode = "fedavg"
            return sample_weights.tolist()

        (
            base_prior,
            base_losses,
            selected_prior,
            standard_score,
            domain_score,
        ) = self._select_adaptive_prior(sample_weights)
        utilities = self._estimate_diu(base_prior, base_losses)
        utility_weights = self._utility_distribution(base_prior, utilities)
        blend = float(self.config.utility_blend)
        if not 0.0 <= blend <= 1.0:
            raise ValueError(f"mu_utility_blend must be in [0, 1], got {blend}")
        final_weights = (1.0 - blend) * base_prior + blend * utility_weights
        if not np.all(np.isfinite(final_weights)):
            raise ValueError(f"DIU produced non-finite weights: {final_weights}")
        if np.any(final_weights < -1e-12):
            raise ValueError(f"DIU produced negative weights: {final_weights}")
        if not np.isclose(final_weights.sum(), 1.0, atol=1e-8):
            raise ValueError(f"DIU weights do not sum to one: {final_weights}")

        self.weighting_mode = "diu_ap_fixed_blend"

        def compact_vector(values):
            return np.round(np.asarray(values, dtype=np.float64), 6).tolist()

        self._log_event(
            "diu_ap_weights",
            round=round_id,
            ap={
                "standard_score": round(standard_score, 6),
                "domain_score": round(domain_score, 6),
                "selected": selected_prior,
            },
            diu={
                "client_scores": compact_vector(utilities),
                "client_weights": compact_vector(utility_weights),
            },
            final_weights=compact_vector(final_weights),
        )
        return final_weights.tolist()
