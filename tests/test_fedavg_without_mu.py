import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from algorithm.server.fedavg import FedAvgServer, get_fedavg_argparser
from algorithm.server.fedomg import get_fedomg_argparser
from algorithm.server.fedgrip import get_fedgrip_argparser
import fdg_experiment_support as core


class PlainFedAvgTests(unittest.TestCase):
    def test_parsers_have_no_legacy_parameters(self):
        for parser in (get_fedavg_argparser, get_fedomg_argparser, get_fedgrip_argparser):
            args = vars(parser().parse_args([]))
            self.assertFalse(any(key.startswith(("mu_", "diu_")) for key in args))

    def test_standard_sample_weighting_without_mu_attribute(self):
        server = object.__new__(FedAvgServer)
        server.algo = "FedAvg"
        server.logger = Mock()
        server.client_list = [
            SimpleNamespace(train_loader=SimpleNamespace(dataset=range(n)))
            for n in (2, 3, 5)
        ]
        self.assertEqual(server.get_agg_weight(), [0.2, 0.3, 0.5])

    def test_baseline_configuration_is_clean(self):
        args = core.configuration("FedAvg", "baseline", "vlcs", 40)
        self.assertFalse(any(key.startswith(("mu_", "diu_")) for key in args))
        with self.assertRaises(ValueError):
            core.configuration("FedAvg", "diu_ap", "vlcs", 40)


if __name__ == "__main__":
    unittest.main()
