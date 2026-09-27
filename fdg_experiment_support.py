"""Shared protocol helpers for retained baseline and FedOMG runners."""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import pickle

from rich.console import Console

from shared_partitions import shared_partition_dir
from utils.tools import Logger


ROOT = Path(__file__).resolve().parent
METHODS = {
    method: method if method == "GA" else method.lower()
    for method in (
        "FedAvg", "FedSR", "FedIIR", "GA"
    )
}
DOMAINS = {
    "office_home": ("art", "clipart", "product", "realworld"),
    "pacs": ("photo", "art_painting", "cartoon", "sketch"),
    "caltech_10": ("amazon", "caltech10", "dslr", "webcam"),
    "vlcs": ("caltech", "labelme", "sun", "voc"),
}
PARTITION_FILES = (
    "args.pkl", "client_data.pkl", "client_stats.pkl", "dataset_stats.pkl"
)
TRAINING = dict(
    model="mobile3l", round=10, num_epochs=3, batch_size=32,
    lr=.001, optimizer="adam", weight_decay=.0001, augment=False,
    test_gap=1, save_log=True,
)


class AppendLogger(Logger):
    """Use the native Rich display while preserving previous log sessions."""

    def __init__(self, path):
        self.stdout = Console(log_path=False, log_time=False)
        self.enable_log = True
        self.logfile_stream = Path(path).open("a", encoding="utf-8")
        self.logger = Console(
            file=self.logfile_stream, record=True, log_path=False, log_time=False
        )

    def log(self, *values, **kwargs):
        super().log(*values, **kwargs)
        self.logfile_stream.flush()


def configuration(method, variant, dataset, seed):
    if variant != "baseline":
        raise ValueError(f"Unsupported variant: {variant}")
    module = importlib.import_module("algorithm.server." + METHODS[method])
    parser = getattr(module, "get_" + METHODS[method] + "_argparser")
    config = vars(parser().parse_args([]))
    config.update(
        TRAINING,
        dataset=dataset,
        seed=seed,
        optimizer="adam",
    )
    for key in ("output_dir", "partition_info_dir", "use_cuda"):
        config.pop(key, None)
    return config


def _partition_is_complete(path):
    return all((path / name).is_file() for name in PARTITION_FILES)


def canonical_partition(dataset, seed, target, allow_create=True):
    """Reuse a frozen shared partition, creating it once when authorized."""
    relative = shared_partition_dir(seed, target)
    absolute = ROOT / "data" / dataset / relative
    if not _partition_is_complete(absolute):
        if not allow_create:
            raise FileNotFoundError(f"Missing canonical partition: {absolute}")
        from data.partition_data import partition_and_statistic

        partition_and_statistic(argparse.Namespace(
            dataset=dataset,
            test_domain=target,
            seed=seed,
            num_clients_per_domain=2,
            directory_name=relative,
            hetero_method="dirichlet",
            alpha=0.0,
        ))
    if not _partition_is_complete(absolute):
        raise RuntimeError(f"Canonical partition generation failed: {absolute}")
    with (absolute / "args.pkl").open("rb") as stream:
        saved = pickle.load(stream)
    if (saved.get("dataset") != dataset
            or int(saved.get("seed", -1)) != seed
            or saved.get("test_domain") != target):
        raise ValueError(f"Canonical partition metadata mismatch: {absolute}")
    return relative
