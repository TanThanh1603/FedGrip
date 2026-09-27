from copy import deepcopy
import os
from pathlib import Path
import pickle
from typing import Dict, List, Union
from matplotlib import pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import argparse
import sys
from collections import defaultdict

PROJECT_DIR = Path(__file__).parent.parent.absolute()
CURRENT_DIR = Path(__file__).parent.absolute()

sys.path.append(PROJECT_DIR.as_posix())
from utils.heterogeneity import heterogeneity

ALL_DOMAINS = {
    "pacs": ["photo", "art_painting", "cartoon", "sketch"],
    "vlcs": ["caltech", "labelme", "sun", "voc"],
    "office_home": ["art", "clipart", "product", "realworld"],
    "domainnet": ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"],
    "minidomainnet": ["clipart", "painting", "real", "sketch"],
    "caltech_10": ["amazon", "caltech10", "dslr", "webcam"],
}

OFFICE_HOME_DIRS = {
    "art": "Art",
    "clipart": "Clipart",
    "product": "Product",
    "realworld": "Real World",
}

CALTECH_10_DIRS = {
    "amazon": "amazon",
    "caltech10": "caltech10",
    "dslr": "dslr",
    "webcam": "webcam",
}


def _get_domain_paths(dataset: str, domains: List[str]) -> Dict[str, str]:
    if dataset == "office_home":
        return {
            domain: os.path.join(
                PROJECT_DIR,
                "data",
                dataset,
                OFFICE_HOME_DIRS[domain],
            )
            for domain in domains
        }
    if dataset == "caltech_10":
        return {
            domain: os.path.join(
                PROJECT_DIR,
                "data",
                dataset,
                "images",
                CALTECH_10_DIRS[domain],
            )
            for domain in domains
        }
    return {
        domain: os.path.join(PROJECT_DIR, f"data/{dataset}/raw", domain)
        for domain in domains
    }


def get_partition_arguments():
    parser = argparse.ArgumentParser(
        description="Federated Domain Generalization Dataset Partitioning"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="pacs",
        choices=["pacs", "vlcs", "office_home", "domainnet", "minidomainnet", "caltech_10"],
    )
    parser.add_argument("--test_domain", type=str, default="sketch", help="Test domain")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--num_clients_per_domain",
        type=int,
        default=2,
        help="Number of clients for each domain",
    )
    parser.add_argument(
        "--directory_name",
        type=str,
        default="partition_info",
        help="name of directory to save partition info",
    )
    # heterogeneity
    parser.add_argument(
        "--hetero_method", type=str, default="dirichlet", help="Heterogeneity method"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="Alpha value for Dirichlet heterogeneity",
    )
    args, _ = parser.parse_known_args()
    return args


def partition_data(args) -> Dict[Union[int, str], Dict[str, List]]:
    """
    Summary:
        Generate data partition for federated learning.
        Validation dataset is extracted from training data with ratio 10%.

    Args:
        args (_type_): _description_

    Returns:
        Dict[Union[int, str], Dict[str, List]]:
            keys: client_id, 'validation', 'test'
            values: Dict with keys 'files', 'labels', 'domain', values are list
                'files': list of file paths
                'labels': list of labels
                'domain': list of domains of each sample
    """
    domains = ALL_DOMAINS[args.dataset]

    test_domain = args.test_domain
    client_domains = [domain for domain in domains if domain != test_domain]
    # num_clients = args.num_clients_per_domain * len(client_domains)
    all_files = defaultdict(list)
    all_labels = defaultdict(list)
    if args.dataset != "minidomainnet":
        domain_paths = _get_domain_paths(args.dataset, domains)
        for domain, path in domain_paths.items():
            for cls in os.listdir(path):
                cls_path = os.path.join(path, cls)
                files = os.listdir(cls_path)
                all_files[domain].extend([os.path.join(cls_path, f) for f in files])
                all_labels[domain].extend([cls] * len(files))
    else:
        domain_paths = {
            domain: os.path.join(PROJECT_DIR, f"data/domainnet/raw", domain)
            for domain in domains
        }
        for domain in domains:
            for mod in ["train", "test"]:
                path = os.path.join(
                    CURRENT_DIR, args.dataset, "splits_mini", f"{domain}_{mod}.txt"
                )
                with open(path, "r") as f:
                    split = f.readlines()
                    files = [line.split(" ")[0] for line in split]
                    label = [line.split(" ")[1][:-1] for line in split]
                    files_path = [
                        os.path.join(domain_paths[domain], f.split("/", 1)[1])
                        for f in files
                    ]
                    all_files[domain].extend(files_path)
                    all_labels[domain].extend(label)

    domain_distribution = heterogeneity(args, client_domains)
    # partition training data
    client_data = defaultdict(lambda: defaultdict(list))
    for domain, assignment in domain_distribution.items():
        n = len(all_files[domain])  # Number of files in this domain
        indices = np.arange(n)
        np.random.shuffle(indices)
        split_idx = int(n * 0.9)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        num_training_data = len(train_indices)
        num_elements_gt_zero = sum(1 for val in assignment if val > 0)
        current_idx = 0
        for client_id, proportion in enumerate(assignment):
            # Distribute files to client assigned with this domain
            if proportion > 0:
                if num_elements_gt_zero != 1:
                    # if this client is not the last client with non-zero proportion
                    num_samples = int(proportion * num_training_data)
                else:
                    # if is the last, assign all remaining samples
                    num_samples = num_training_data - current_idx + 1
                client_data_indices = train_indices[
                    current_idx : current_idx + num_samples
                ]
                client_data[client_id]["files"].extend(
                    [all_files[domain][i] for i in client_data_indices]
                )
                client_data[client_id]["labels"].extend(
                    [all_labels[domain][i] for i in client_data_indices]
                )
                client_data[client_id]["domain"].extend([domain] * num_samples)
                current_idx += num_samples
                num_elements_gt_zero -= 1
        client_data["validation"]["files"].extend(
            [all_files[domain][i] for i in val_indices]
        )
        client_data["validation"]["labels"].extend(
            [all_labels[domain][i] for i in val_indices]
        )
        client_data["validation"]["domain"].extend([domain] * len(val_indices))

    # data for test and validation
    client_data["test"] = {
        "files": all_files[test_domain],
        "labels": all_labels[test_domain],
        "domain": [test_domain] * len(all_files[test_domain]),
    }
    return defaultdict_to_dict(client_data)


def client_statistics(client_data) -> Dict[int, Dict[str, Dict[str, int]]]:
    """
    Calculate the statistics of samples per domain and label for each client.

    Args:
        client_data (_type_): return of function partition_data

    Returns:
        Dict[int, Dict[str, Dict[str, int]]]:
            client_id:
                "domain": #samples from this domain
                "label": #samples for this label
    """
    client_stats = defaultdict(lambda: defaultdict(dict))
    for client_id, data in client_data.items():
        if isinstance(client_id, int):  # Skip 'validation' and 'test'
            domains = set(data["domain"])
            for domain in domains:
                client_stats[client_id]["domain"][domain] = data["domain"].count(domain)
            labels = set(data["labels"])
            for label in labels:
                client_stats[client_id]["label"][label] = data["labels"].count(label)
    return defaultdict_to_dict(client_stats)


def dataset_statistics(args):
    stats = {"domain": {}, "label": {}}
    domains = ALL_DOMAINS[args.dataset]
    if args.dataset != "minidomainnet":
        domain_paths = _get_domain_paths(args.dataset, domains)
        for domain, path in domain_paths.items():
            labels = os.listdir(path)
            num_samples = 0
            for label in labels:
                label_dir_path = os.path.join(domain_paths[domain], label)
                files = os.listdir(label_dir_path)
                num_files = len(files)
                stats["label"][label] = stats["label"].get(label, 0) + num_files
                num_samples += num_files
            stats["domain"][domain] = num_samples
    else:
        domain_paths = {
            domain: os.path.join(PROJECT_DIR, f"data/domainnet/raw", domain)
            for domain in domains
        }
        for domain in domains:
            for mod in ["train", "test"]:
                path = os.path.join(
                    CURRENT_DIR, args.dataset, "splits_mini", f"{domain}_{mod}.txt"
                )
                with open(path, "r") as f:
                    split = f.readlines()
                    for line in split:
                        file_path, label = line.split(" ")
                        label = label[:-1]
                        stats["label"][label] = stats["label"].get(label, 0) + 1
                        stats["domain"][domain] = stats["domain"].get(domain, 0) + 1
    return stats


def plot_sample_distribution(client_stats, plot_type="domain", save_path=None):
    """
    Plot the distribution of samples across clients with grouped bars.

    Args:
        client_stats (Dict[int, Dict[str, Dict[str, int]]]):
            The statistics of samples per domain and label for each client.
        plot_type (str):
            The type of distribution to plot: "domain" or "label".
        save_path (str or None):
            If provided, the plot will be saved to this path.
    """
    clients = sorted(client_stats.keys())
    num_clients = len(clients)

    # Extract relevant data for plotting
    if plot_type == "domain":
        items = sorted(
            {
                domain
                for stats in client_stats.values()
                for domain in stats["domain"].keys()
            }
        )
        data = {
            item: [client_stats[client]["domain"].get(item, 0) for client in clients]
            for item in items
        }
        ylabel = "Number of Samples from Domain"
        title = "Domain Distribution Across Clients"
    elif plot_type == "label":
        items = sorted(
            {
                label
                for stats in client_stats.values()
                for label in stats["label"].keys()
            }
        )
        data = {
            item: [client_stats[client]["label"].get(item, 0) for client in clients]
            for item in items
        }
        ylabel = "Number of Samples per Label"
        title = "Label Distribution Across Clients"
    else:
        raise ValueError("Invalid plot_type. Choose 'domain' or 'label'.")

    fig, ax = plt.subplots()

    bar_width = 0.2  # Width of each bar
    indices = np.arange(num_clients)  # The x locations for the groups

    # Plot each domain/label as a separate set of bars
    for i, item in enumerate(items):
        ax.bar(indices + i * bar_width, data[item], bar_width, label=item)

    ax.set_xlabel("Client ID")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(indices + bar_width * (len(items) - 1) / 2)
    ax.set_xticklabels(clients)
    ax.legend(title=plot_type.capitalize())

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")


def defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d


def partition_and_statistic(args):

    np.random.seed(args.seed)
    test_domain = args.test_domain
    all_domains = ALL_DOMAINS[args.dataset]
    assert (
        test_domain in all_domains
    ), f"Test domain {test_domain} not found in {args.dataset}"
    client_data = partition_data(args)
    client_stats = client_statistics(client_data)
    dataset_stats = dataset_statistics(args)

    save_path = os.path.join(CURRENT_DIR, args.dataset, args.directory_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plot_sample_distribution(
        client_stats,
        plot_type="domain",
        save_path=os.path.join(save_path, "domain_distribution.png"),
    )

    # Save client_data as .pkl file
    with open(os.path.join(save_path, "client_data.pkl"), "wb") as f:
        pickle.dump(client_data, f)

    # Save client_stats as .pkl file
    with open(os.path.join(save_path, "client_stats.pkl"), "wb") as f:
        pickle.dump(client_stats, f)

    # Save dataset_stats as .pkl file
    with open(os.path.join(save_path, "dataset_stats.pkl"), "wb") as f:
        pickle.dump(dataset_stats, f)

    # Save Args as .pkl file
    with open(os.path.join(save_path, "args.pkl"), "wb") as f:
        pickle.dump(vars(args), f)


if __name__ == "__main__":
    args = get_partition_arguments()
    partition_and_statistic(deepcopy(args))
