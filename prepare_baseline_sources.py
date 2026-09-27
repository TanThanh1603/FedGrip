"""Download the baseline source packages used by the experiments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from urllib.parse import quote
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
DESTINATION = ROOT / "third_party/baselines/sources"
USER_AGENT = "fedccrl-baseline-installer"
INSTALLER_VERSION = 2
SOURCE_SUFFIXES = {
    ".py", ".sh", ".yaml", ".yml", ".json", ".txt", ".md", ".toml", ".cfg",
}
SOURCES = {
    "pfl": {
        "repo": "skydvn/FedOMG",
        "branch": "main",
        "prefixes": ("system/", "FedOMG-DG/algorithms/fedsam/"),
        "roots": ("README.md", "LICENSE", "requirements.txt"),
        "provides": ("PerAvg", "FedRoD", "FedPAC", "FedBABU", "FedSAM"),
    },
    "fedas": {
        "repo": "xiyuanyang45/FedAS",
        "branch": "main",
        "prefixes": ("system/",),
        "roots": ("README.md", "Instructions.md", "requirements.txt", "LICENSE"),
        "provides": ("FedAS",),
    },
    "stablefdg": {
        "repo": "savertm/StableFDG_github",
        "branch": "master",
        "prefixes": ("Dassl.pytorch/", "StableFDG/"),
        "roots": ("README.md", "requirements.txt", "LICENSE"),
        "provides": ("StableFDG",),
    },
}


def request_json(url):
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def request_bytes(url):
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=120) as response:
        return response.read()


def selected_files(tree, prefixes, roots):
    result = []
    for item in tree:
        path = item["path"]
        if item["type"] != "blob":
            continue
        if path in roots or (
            any(path.startswith(prefix) for prefix in prefixes)
            and Path(path).suffix.lower() in SOURCE_SUFFIXES
        ):
            result.append(item)
    return result


def install(name, spec):
    repo = spec["repo"]
    branch = spec["branch"]
    branch_record = request_json(f"https://api.github.com/repos/{repo}/branches/{branch}")
    commit = branch_record["commit"]["sha"]
    tree_record = request_json(
        f"https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1"
    )
    if tree_record.get("truncated"):
        raise RuntimeError(f"GitHub tree response was truncated for {repo}")
    files = selected_files(tree_record["tree"], spec["prefixes"], spec["roots"])
    destination = DESTINATION / name
    existing = destination / "INSTALL_MANIFEST.json"
    if existing.exists():
        installed = json.loads(existing.read_text())
        if (installed.get("commit") == commit
                and installed.get("installer_version") == INSTALLER_VERSION):
            print(f"{name}: already installed at {commit[:12]}", flush=True)
            return installed
    temporary = DESTINATION / f".{name}.installing"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    records = []
    for position, item in enumerate(files, 1):
        path = item["path"]
        encoded = quote(path, safe="/")
        content = request_bytes(
            f"https://raw.githubusercontent.com/{repo}/{commit}/{encoded}"
        )
        target = temporary / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        records.append({
            "path": path,
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "git_blob": item["sha"],
        })
        if position % 50 == 0:
            print(f"{name}: downloaded {position}/{len(files)} files", flush=True)
    manifest = {
        "repository": f"https://github.com/{repo}",
        "branch": branch,
        "commit": commit,
        "installer_version": INSTALLER_VERSION,
        "provides": list(spec["provides"]),
        "files": records,
    }
    (temporary / "INSTALL_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if destination.exists():
        shutil.rmtree(destination)
    temporary.replace(destination)
    print(
        f"{name}: installed {len(files)} files ({sum(x['bytes'] for x in records)} bytes) "
        f"at {commit[:12]}", flush=True,
    )
    return manifest


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    manifests = {name: install(name, spec) for name, spec in SOURCES.items()}
    inventory = {
        "installed_baselines": {
            baseline: source
            for source, manifest in manifests.items()
            for baseline in manifest["provides"]
        },
        "already_native_in_fedccrl": ["FedAvg", "FedGA", "FedIIR", "FedSR", "FedOMG"],
        "scope": (
            "Pinned upstream reference implementations. They retain their original "
            "frameworks and are not silently registered as native fedccrl algorithms."
        ),
    }
    (DESTINATION / "inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Inventory: {DESTINATION / 'inventory.json'}", flush=True)


if __name__ == "__main__":
    main()
