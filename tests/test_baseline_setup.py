import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
BASELINES = ROOT / "third_party/baselines/sources"
EXPECTED = {
    "PerAvg": "pfl/system/flcore/clients/clientperavg.py",
    "FedRoD": "pfl/system/flcore/clients/clientrod.py",
    "FedPAC": "pfl/system/flcore/clients/clientpac.py",
    "FedBABU": "pfl/system/flcore/clients/clientbabu.py",
    "FedAS": "fedas/system/flcore/clients/clientas.py",
    "FedSAM": "pfl/FedOMG-DG/algorithms/fedsam/optimizer/esam.py",
    "StableFDG": (
        "stablefdg/Dassl.pytorch/dassl/modeling/ops/style_insert.py"
    ),
}


class BaselineSetupTests(unittest.TestCase):
    def test_all_missing_table_baselines_are_installed(self):
        inventory = json.loads((BASELINES / "inventory.json").read_text())
        self.assertEqual(set(inventory["installed_baselines"]), set(EXPECTED))
        for baseline, relative in EXPECTED.items():
            self.assertTrue((BASELINES / relative).is_file(), baseline)

    def test_installed_sources_match_pinned_manifests(self):
        for manifest_path in BASELINES.glob("*/INSTALL_MANIFEST.json"):
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(len(manifest["commit"]), 40)
            for record in manifest["files"]:
                content = (manifest_path.parent / record["path"]).read_bytes()
                self.assertEqual(hashlib.sha256(content).hexdigest(), record["sha256"])


if __name__ == "__main__":
    unittest.main()
