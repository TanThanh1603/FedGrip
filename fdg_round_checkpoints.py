"""Round-boundary resume for the retained FL/FDG servers."""
from copy import deepcopy
import time

import torch

from experiment_checkpoints import (atomic_output, capture_rng, cpu_tree, digest,
                                    restore_rng, write_json)

MODELS = ("classification_model",)
OPTIMIZERS = ("optimizer", "scheduler")
AUXILIARY = (
    "generalization_gap", "agg_weight", "step_size", "grad_mean",
    "average_loss", "best_accuracy", "grip_temporal_update",
)


def owner_state(owner):
    models = {}
    for name in MODELS:
        module = getattr(owner, name, None)
        if isinstance(module, torch.nn.Module):
            models[name] = dict(weights=cpu_tree(module.state_dict()),
                                modes={k: m.training for k, m in module.named_modules()},
                                alpha=getattr(module, "alpha", None))
    return dict(models=models,
                optimizers={n: cpu_tree(getattr(owner, n).state_dict()) for n in OPTIMIZERS if hasattr(owner, n)},
                auxiliary={n: cpu_tree(getattr(owner, n)) for n in AUXILIARY if hasattr(owner, n)})


def restore_owner(owner, state):
    for name, record in state["models"].items():
        module = getattr(owner, name)
        module.load_state_dict(record["weights"])
        for key, child in module.named_modules():
            child.training = record["modes"][key]
        if record["alpha"] is not None:
            module.alpha = record["alpha"]
    for name, value in state["optimizers"].items():
        getattr(owner, name).load_state_dict(value)
    for name, value in state["auxiliary"].items():
        setattr(owner, name, value)


def save_checkpoint(path, signature, next_round, server, history):
    state = dict(signature=signature, torch_version=str(torch.__version__), next_round=next_round,
                 server=owner_state(server), clients=[owner_state(c) for c in server.client_list],
                 ids=[c.client_id for c in server.client_list], history=history, rng=capture_rng())
    with atomic_output(path) as stream:
        torch.save(state, stream)


def load_checkpoint(path, signature, server):
    state = torch.load(path, map_location="cpu", weights_only=False)  # trusted local checkpoint only
    if state["signature"] != signature or state["torch_version"] != str(torch.__version__):
        raise ValueError("Checkpoint configuration/code/PyTorch mismatch")
    if state["ids"] != [c.client_id for c in server.client_list] or len(state["clients"]) != len(server.client_list):
        raise ValueError("Checkpoint clients mismatch")
    restore_owner(server, state["server"])
    for client, record in zip(server.client_list, state["clients"]):
        restore_owner(client, record)
        client.device = None  # native migration recreates optimizers with their restored state
    restore_rng(state["rng"])
    return state["next_round"], state["history"]


def train(server, job, directory, warmup_checkpoint=None, warmup_signature=None,
          publish_warmup=False, warmup_rounds=2):
    method = job["method"]
    checkpoint, signature = directory / "round_checkpoint.pt", digest(job)
    if checkpoint.exists():
        start, history = load_checkpoint(checkpoint, signature, server)
    elif warmup_checkpoint is not None and warmup_checkpoint.exists() and not publish_warmup:
        if warmup_signature is None:
            raise ValueError("A shared warm-up checkpoint requires its compatibility signature")
        start, history = load_checkpoint(warmup_checkpoint, warmup_signature, server)
        if start != warmup_rounds:
            raise ValueError(
                f"Shared warm-up ends at round {start}, expected {warmup_rounds}"
            )
        # Immediately create the experiment-local checkpoint. A later Ctrl+C
        # therefore never depends on the shared cache remaining in place.
        save_checkpoint(checkpoint, signature, start, server, history)
        server.logger.log(
            f"Loaded shared baseline warm-up; next round {start}: {warmup_checkpoint}"
        )
    else:
        initial = server.classification_model.state_dict()
        for client in server.client_list:
            client.load_model_weights(deepcopy(initial))
        server.best_accuracy = 0
        if method == "GA":
            server.generalization_gap = []
        if method == "FedIIR":
            server.grad_mean = tuple(torch.zeros_like(p) for p in server.classification_model.classifier.parameters())
        start, history = 0, []
        save_checkpoint(checkpoint, signature, start, server, history)
    server.logger.log(f"START/RESUME round {start}; {method} {job['variant']}")
    for round_id in range(start, server.args.round):
        server.round_id = round_id
        server.logger.log("=" * 20, f"Round {round_id}", "=" * 20)
        begin = time.monotonic()
        if method == "GA":
            server.step_size = server.args.step_size * (server.args.round - round_id) / server.args.round / 3
        if method == "FedIIR":
            server.client_gradient = [c.get_client_grad() for c in server.client_list]
            means = tuple(torch.mean(torch.stack(g), dim=0) for g in zip(*server.client_gradient))
            server.grad_mean = tuple(server.args.ema * g + (1 - server.args.ema) * m
                                     for g, m in zip(server.grad_mean, means))
        for client in server.client_list:
            if method == "FedIIR":
                client.set_grad_mean(server.grad_mean)
            client.train()
        weights = server.aggregate_model()
        server.classification_model.load_state_dict(weights)
        for client in server.client_list:
            client.load_model_weights(weights)
        if method == "GA":
            server.generalization_gap = [c.get_generalization_gap() for c in server.client_list]
        if (round_id + 1) % server.args.test_gap == 0:
            server.validate_and_test()
        history.append(dict(round=round_id, test_accuracy=float(server.best_accuracy),
                            weights=[float(w) for w in server.agg_weight],
                            seconds_before_checkpoint=time.monotonic() - begin))
        save_checkpoint(checkpoint, signature, round_id + 1, server, history)
        if (publish_warmup and warmup_checkpoint is not None
                and round_id + 1 == warmup_rounds):
            if warmup_signature is None:
                raise ValueError("Cannot publish warm-up without a compatibility signature")
            warmup_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                warmup_checkpoint, warmup_signature, round_id + 1, server, history
            )
            server.logger.log(f"Published shared warm-up: {warmup_checkpoint}")
        server.logger.log(f"Checkpoint saved: round {round_id}; next round {round_id + 1}")
    write_json(directory / "complete.json", dict(signature=signature, next_round=server.args.round,
               history=history, accuracy=float(server.best_accuracy)))
