from copy import deepcopy
import os
from pathlib import Path
import sys
from argparse import Namespace, ArgumentParser
import pickle
import time
import pandas as pd
from typing import Dict, List
import numpy as np
from rich.console import Console
import torch
from multiprocessing import cpu_count, Pool

# PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()
# sys.path.append(PROJECT_DIR.as_posix())

from data.partition_data import (
    partition_and_statistic,
    get_partition_arguments,
    ALL_DOMAINS,
)
from algorithm.server.fedavg import FedAvgServer, get_fedavg_argparser
from algorithm.server.fedsr import FedSRServer, get_fedsr_argparser
from algorithm.server.GA import GAServer, get_GA_argparser
from algorithm.server.fediir import FedIIRServer, get_fediir_argparser
from algorithm.server.fedsam import FedSAMServer, get_fedsam_argparser
from algorithm.server.stablefdg import StableFDGServer, get_stablefdg_argparser
from utils.tools import local_time
from shared_partitions import ensure_shared_partition_dir


PROJECT_DIR = Path(__file__).resolve().parent

algo2server = {
    "FedAvg": FedAvgServer,
    "FedSR": FedSRServer,
    "GA": GAServer,
    "FedIIR": FedIIRServer,
    "FedSAM": FedSAMServer,
    "StableFDG": StableFDGServer,
}
algo2argparser = {
    "FedAvg": get_fedavg_argparser(),
    "FedSR": get_fedsr_argparser(),
    "GA": get_GA_argparser(),
    "FedIIR": get_fediir_argparser(),
    "FedSAM": get_fedsam_argparser(),
    "StableFDG": get_stablefdg_argparser(),
}


def get_output_dir(args):
    if algo in ["FedAvg", "GA", "FedSAM", "StableFDG"]:
        return begin_time
    elif algo == "FedSR":
        output_dir = f"L2R_{args.L2R_coeff}_CMI_{args.CMI_coeff}"
    elif algo == "FedIIR":
        output_dir = f"gamma_{args.gamma}_ema_{args.ema}"

    output_dir = output_dir + "_" + begin_time
    return output_dir


def get_main_argparser():
    parser = ArgumentParser(description="Main arguments.")
    parser.add_argument("-a", "--algo", type=str, default="FedMS", choices=list(algo2server.keys()))
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="pacs",
        choices=list(ALL_DOMAINS.keys()),
    )
    return parser


def process(test_domain):
    time.sleep(np.random.randint(0, 5))
    # 1. partition data
    if shared_partition_seed is not None:
        dir_name = ensure_shared_partition_dir(
            PROJECT_DIR,
            dataset,
            shared_partition_seed,
            test_domain,
        )
    elif resume_dataset_dir is None:
        data_args = get_partition_arguments()
        data_args.test_domain = test_domain
        dir_name = os.path.join(begin_time, test_domain)
        data_args.directory_name = dir_name
        partition_and_statistic(deepcopy(data_args))
    else:
        dir_name = os.path.join(resume_dataset_dir, test_domain)
    # 2. train
    fl_args, _ = algo2argparser[algo].parse_known_args()
    if shared_partition_seed is not None:
        # The training RNG and the frozen partition must carry the same seed.
        fl_args.seed = shared_partition_seed
    fl_args.partition_info_dir = dir_name
    fl_args.output_dir = (
        get_output_dir(fl_args) if resume_run_log_dir is None else resume_run_log_dir
    )
    if "domainnet" in fl_args.dataset:
        fl_args.batch_size = 128
    server = algo2server[algo](args=deepcopy(fl_args))
    server.process_classification()


def get_table():
    test_accuracy = {}
    args, _ = algo2argparser[algo].parse_known_args()
    path2dir = os.path.join(
        "out",
        algo,
        dataset,
        get_output_dir(args) if resume_run_log_dir is None else resume_run_log_dir,
    )
    table_domains = ALL_DOMAINS[dataset] if resume_run_log_dir is not None else domains
    for domain in table_domains:
        with open(os.path.join(path2dir, domain, "test_accuracy.pkl"), "rb") as f:
            test_accuracy[domain] = round(pickle.load(f), 2)
    average_accuracy = round(np.mean(list(test_accuracy.values())), 2)
    test_accuracy["average"] = average_accuracy
    test_accuracy_df = pd.DataFrame(test_accuracy, index=[algo])
    test_accuracy_df.to_csv(os.path.join(path2dir, "test_accuracy.csv"))
    return test_accuracy


if __name__ == "__main__":
    begin_time = local_time()
    algo = sys.argv[1]
    assert algo in algo2server.keys()
    del sys.argv[1]
    if algo in ("FedSAM", "StableFDG") and any(flag in sys.argv for flag in ("-h", "--help")):
        algo2argparser[algo].print_help()
        sys.exit(0)
    runtime_parser = ArgumentParser(add_help=False)
    runtime_parser.add_argument(
        "-d", "--dataset", required=True, choices=list(ALL_DOMAINS.keys())
    )
    runtime_parser.add_argument("--only-test-domain", default=None)
    runtime_parser.add_argument("--resume-run-log-dir", default=None)
    runtime_parser.add_argument("--resume-dataset-dir", default=None)
    runtime_parser.add_argument(
        "--shared-partition-seed",
        type=int,
        default=None,
        help=(
            "reuse data/<dataset>/fedavg_shared_partition_seedN for every "
            "method and lock the training seed to N"
        ),
    )
    runtime_args, _ = runtime_parser.parse_known_args()

    dataset = runtime_args.dataset
    resume_run_log_dir = runtime_args.resume_run_log_dir
    resume_dataset_dir = runtime_args.resume_dataset_dir
    shared_partition_seed = runtime_args.shared_partition_seed
    if shared_partition_seed is not None and resume_dataset_dir is not None:
        raise ValueError(
            "Use either --shared-partition-seed or --resume-dataset-dir, not both"
        )
    if runtime_args.only_test_domain is None:
        domains = ALL_DOMAINS[dataset]
    else:
        if runtime_args.only_test_domain not in ALL_DOMAINS[dataset]:
            raise ValueError(
                f"Unknown test domain {runtime_args.only_test_domain!r} for {dataset}. "
                f"Choose from {ALL_DOMAINS[dataset]}."
            )
        domains = [runtime_args.only_test_domain]
    multiprocess = False
    if multiprocess:
        num_processes = min(len(domains), cpu_count())
        pool = Pool(processes=num_processes)
        try:
            pool.map(process, domains)
            pool.close()
            pool.join()
        except Exception as e:
            pool.terminate()
            pool.join()
            raise RuntimeError("An error occurred in one of the worker processes.") from e
    else:
        for domain in domains:
            process(domain)
    get_table()
