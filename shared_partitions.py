"""Canonical names for paired FedAvg-compatible data partitions.

The directories named by this module are stable aliases.  Their targets may
retain historical experiment names, but every method should use these aliases
so that a seed always resolves to the same frozen partition within a dataset.
"""

from pathlib import Path


FEDAVG_SHARED_PARTITION_PREFIX = "fedavg_shared_partition_seed"

# Historical storage names. These are implementation details used only to
# recreate the stable aliases on a machine where the partition data exists.
LEGACY_PARTITION_ROOTS = {
    "pacs": {
        39: "fedcmu_final_ablation_shared_seed39",
        40: "support_full_2026-08-20-14-22-56-021007",
        41: "fedcmu_final_ablation_shared_seed41",
        42: "fedcmu_refinement_shared_seed42",
        43: "fedcmu_final_ablation_shared_seed43",
        44: "fedcmu_final_ablation_shared_seed44",
    },
    "office_home": {
        38: "fedcmu_refinement_shared_seed38",
        39: "fedcmu_refinement_shared_seed39",
        40: "fedcmu_refinement_shared_seed40",
        41: "fedcmu_refinement_shared_seed41",
        42: "fedcmu_refinement_shared_seed42",
        43: "fedcmu_refinement_shared_seed43",
        44: "fedcmu_refinement_shared_seed44",
    },
    "caltech_10": {
        39: "fedcmu_final_ablation_shared_seed39",
        40: "fedcmu_final_ablation_shared_seed40",
        41: "fedcmu_final_ablation_shared_seed41",
        42: "fedcmu_final_ablation_shared_seed42",
        43: "fedcmu_final_ablation_shared_seed43",
        44: "fedcmu_final_ablation_shared_seed44",
    },
}


def shared_partition_root(seed: int) -> str:
    if seed < 0:
        raise ValueError("Partition seed must be non-negative")
    return f"{FEDAVG_SHARED_PARTITION_PREFIX}{seed}"


def shared_partition_dir(seed: int, domain: str) -> str:
    if not domain:
        raise ValueError("Partition domain must be non-empty")
    return f"{shared_partition_root(seed)}/{domain}"


def ensure_shared_partition_dir(
    project_dir: Path,
    dataset: str,
    seed: int,
    domain: str,
) -> str:
    """Create the stable alias when its verified historical target exists."""

    relative = shared_partition_dir(seed, domain)
    dataset_root = Path(project_dir) / "data" / dataset
    alias_root = dataset_root / shared_partition_root(seed)
    if alias_root.exists():
        return relative
    if alias_root.is_symlink():
        raise FileNotFoundError(f"Broken shared-partition alias: {alias_root}")

    try:
        legacy_name = LEGACY_PARTITION_ROOTS[dataset][seed]
    except KeyError as error:
        raise FileNotFoundError(
            f"No paired partition is registered for {dataset} seed={seed}"
        ) from error
    legacy_root = dataset_root / legacy_name
    if not legacy_root.is_dir():
        raise FileNotFoundError(
            f"Cannot create {alias_root.name}: source partition is missing: "
            f"{legacy_root}"
        )
    alias_root.symlink_to(legacy_name, target_is_directory=True)
    return relative
