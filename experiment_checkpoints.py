"""Small round-boundary checkpoint helpers for standalone experiments."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


@contextmanager
def atomic_output(path, mode="wb"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, mode) as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    with atomic_output(path, "w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Cannot resume CUDA RNG with a different device count.")
        torch.cuda.set_rng_state_all(state["cuda"])


def save_round(path, signature, next_round, server, history):
    state = dict(signature=signature, next_round=next_round,
                 model=cpu_tree(server.classification_model.state_dict()),
                 ids=[client.client_id for client in server.client_list],
                 optimizers=[cpu_tree(c.optimizer.state_dict()) for c in server.client_list],
                 schedulers=[cpu_tree(c.scheduler.state_dict()) for c in server.client_list],
                 rng=capture_rng(), history=history, best_accuracy=server.best_accuracy,
                 torch_version=str(torch.__version__))
    with atomic_output(path) as stream:
        torch.save(state, stream)


def load_round(path, signature, server):
    # Trusted local files only: optimizer and RNG checkpoints contain Python objects.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["signature"] != signature or state["torch_version"] != str(torch.__version__):
        raise ValueError("Checkpoint protocol/PyTorch mismatch.")
    if state["ids"] != [c.client_id for c in server.client_list]:
        raise ValueError("Checkpoint client IDs mismatch.")
    if not len(state["ids"]) == len(state["optimizers"]) == len(state["schedulers"]):
        raise ValueError("Checkpoint client state count mismatch.")
    server.classification_model.load_state_dict(state["model"])
    for client, optimizer, scheduler in zip(server.client_list, state["optimizers"], state["schedulers"]):
        client.load_model_weights(state["model"])
        client.optimizer.load_state_dict(optimizer)
        client.scheduler.load_state_dict(scheduler)
        client.device = None  # native FedAvgClient will migrate restored optimizer tensors
    server.best_accuracy = state["best_accuracy"]
    restore_rng(state["rng"])
    return state["next_round"], state["history"]
