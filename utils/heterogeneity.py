import numpy as np
from collections import defaultdict
from typing import List, Dict


def heterogeneity(args, client_domains: List[str], **kwargs) -> Dict[str, List[float]]:
    """Summary

    Args:
        args (type): Description
        client_domains (List[str]): Domains for training

    Returns:
        Dict[str,List[float]]:
            Domain distribution, where the key is the domain name and the value is a list.
            Each element in the list corresponds to the proportion of this domain on this client.
    """
    np.random.seed(args.seed)
    num_clients = args.num_clients_per_domain * len(client_domains)
    hetero_method = args.hetero_method
    if hetero_method == "dirichlet":
        domain_distribution = Dirichlet_heterogeneity(args.alpha, client_domains, num_clients)
    elif hetero_method == "uniform":
        domain_distribution = defaultdict(list)
        for domain in client_domains:
            domain_distribution[domain] = (np.ones(num_clients) / num_clients).tolist()
    else:
        raise ValueError("Unknown heterogeneity method")
    return domain_distribution


def Dirichlet_heterogeneity(alpha, client_domains, num_clients) -> Dict[str, List[float]]:
    domain_distribution = defaultdict(list)
    if not alpha > 0:
        # domain seperation
        client_domain_indices = np.array_split(np.arange(num_clients), len(client_domains))
        for i, domain in enumerate(client_domains):
            distribution = [
                1 if client in client_domain_indices[i] else 0 for client in range(num_clients)
            ]
            domain_distribution[domain] = (
                np.array(distribution) / (num_clients / len(client_domains))
            ).tolist()
    else:
        for domain in client_domains:
            proportions = np.random.dirichlet([alpha] * num_clients)
            domain_distribution[domain] = proportions.tolist()
    return domain_distribution
