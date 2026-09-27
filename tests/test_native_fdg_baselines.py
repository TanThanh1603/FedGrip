import unittest
from copy import deepcopy
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from algorithm.client.fedsam import sam_step
from algorithm.server.fedsam import get_fedsam_argparser
from algorithm.server.stablefdg import get_stablefdg_argparser, StableFDGServer
from model.stablefdg import StableFDGModel, StyleExploration, FeatureHighlighter, share_style


class NativeBaselineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(40)

    def test_sam_zero_radius_equals_sgd(self):
        model = nn.Linear(4, 3)
        reference = deepcopy(model)
        x, y = torch.randn(5, 4), torch.tensor([0, 1, 2, 0, 1])
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        expected = torch.optim.SGD(reference.parameters(), lr=.01)
        F.cross_entropy(reference(x), y).backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 10)
        expected.step()
        sam_step(model, optimizer, x, y, 0)
        for a, b in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(a, b)

    def test_sam_restores_perturbation_on_failure(self):
        model = nn.Linear(4, 3)
        weights = deepcopy(model.state_dict())
        forward = model.forward
        calls = []
        def failing(x):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError('second pass')
            return forward(x)
        with patch.object(model, 'forward', side_effect=failing):
            with self.assertRaisesRegex(RuntimeError, 'second pass'):
                sam_step(model, torch.optim.SGD(model.parameters(), lr=.01),
                         torch.randn(3, 4), torch.tensor([0, 1, 2]), .1)
        for k, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, weights[k]))

    def test_sam_matches_upstream_esam_update(self):
        import importlib.util
        from pathlib import Path
        path = Path(__file__).resolve().parents[1] / 'third_party/baselines/sources/pfl/FedOMG-DG/algorithms/fedsam/optimizer/esam.py'
        spec = importlib.util.spec_from_file_location('reference_esam', path)
        upstream = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(upstream)
        model = nn.Linear(4, 3)
        reference = deepcopy(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.01, momentum=.9)
        esam = upstream.ESAM(reference.parameters(), reference_optimizer, .1)
        for _ in range(2):
            x, y = torch.randn(5, 4), torch.tensor([0, 1, 2, 0, 1])
            esam.paras = [x, y, F.cross_entropy, reference]
            esam.step()
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 10)
            reference_optimizer.step()
            sam_step(model, optimizer, x, y, .1)
            for a, b in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(a, b)

    def test_native_client_collect_train(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from torch.utils.data import TensorDataset
        from algorithm.client.stablefdg import StableFDGClient
        data = TensorDataset(torch.randn(4, 3, 64, 64), torch.tensor([0, 3, 0, 3]))
        data.labels = [0, 3, 0, 3]
        data.label_to_index = {0: 0, 3: 3}
        args = SimpleNamespace(dataset='pacs', model='res18', stable_exploration=3.,
                               stable_oversampling=2, batch_size=4, num_epochs=1,
                               round=2, optimizer='sgd', lr=.001, weight_decay=.0001,
                               use_cuda=False)
        with patch('algorithm.client.stablefdg.StableFDGModel',
                   side_effect=lambda *a: StableFDGModel(*a, pretrained=False)):
            client = StableFDGClient(args, data, 0, Mock())
        before = deepcopy(client.classification_model.state_dict())
        mean, std = client.collect_style()
        self.assertEqual(mean.shape, (128,))
        self.assertTrue(torch.isfinite(std).all())
        for key, value in before.items():
            self.assertTrue(torch.equal(value, client.classification_model.state_dict()[key]))
        client.peer_style = (mean, std)
        client.train()
        client.train()
        self.assertFalse(torch.equal(before['classifier.weight'], client.classification_model.classifier.weight))

    def test_style_sharing_short_batch_and_gradients(self):
        for size in (1, 2, 5, 32):
            x = torch.randn(size, 4, 3, 3, requires_grad=True)
            with patch('model.stablefdg.random.random', return_value=0):
                result = share_style(x, (torch.ones(8), torch.ones(8) * .1))
            self.assertEqual(result.shape, x.shape)
            result.square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_exploration_label_alignment(self):
        module = StyleExploration(3, 4)
        x = torch.randn(3, 4, 3, 3, requires_grad=True)
        labels = torch.tensor([1, 1, 5])
        extra_labels = torch.tensor([1, 5])
        with patch('model.stablefdg.random.random', return_value=0):
            result, result_labels, first = module(
                x, labels, torch.randn(2, 4, 3, 3), extra_labels, True)
        self.assertEqual(len(result), 7)
        self.assertEqual(len(result_labels), len(result))
        self.assertEqual(set(result_labels.tolist()), {1, 5})
        self.assertFalse(first)
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_highlighter_singleton_noncontiguous_class(self):
        module = FeatureHighlighter(4)
        x = torch.randn(3, 4, 3, 3, requires_grad=True)
        q, _ = module.query_key(torch.randn(2, 4, 3, 3))
        out = module(x, torch.tensor([1, 1, 5]), q, torch.tensor([1, 5]))
        self.assertEqual(out.shape, (3, 8))
        out.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_real_resnet_train_eval(self):
        model = StableFDGModel('pacs', pretrained=False, oversampling=2)
        model.train()
        extra_y = torch.tensor([0, 3])
        features, query = model.supplemental(torch.randn(2, 3, 64, 64))
        with patch('model.stablefdg.random.random', return_value=0):
            logits, labels = model(torch.randn(3, 3, 64, 64), torch.tensor([0, 3, 0]),
                                   (features, query, extra_y), None)
        F.cross_entropy(logits, labels).backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        model.eval()
        with torch.no_grad():
            self.assertEqual(model(torch.randn(1, 3, 64, 64)).shape, (1, 7))

    def test_cli(self):
        self.assertEqual(get_fedsam_argparser().parse_args([]).sam_rho, .1)
        self.assertEqual(get_stablefdg_argparser().parse_args([]).model, 'res18')

    def test_server_sharing_uses_other_client(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        server = StableFDGServer.__new__(StableFDGServer)
        server.args = SimpleNamespace(round=2, test_gap=1)
        server.num_client = 3
        server.logger = Mock()
        server.classification_model = nn.Linear(1, 1)
        server.client_list = [Mock() for _ in range(3)]
        observed = [[] for _ in range(3)]
        for cid, client in enumerate(server.client_list):
            client.collect_style.return_value = cid
            client.train.side_effect = lambda cid=cid, client=client: observed[cid].append(client.peer_style)
        server.aggregate_model = lambda: deepcopy(server.classification_model.state_dict())
        server.validate_and_test = Mock()
        server.process_classification()
        for cid, values in enumerate(observed):
            self.assertIsNone(values[0])
            self.assertNotEqual(values[1], cid)
        self.assertEqual(server.validate_and_test.call_count, 2)


if __name__ == '__main__':
    unittest.main()
