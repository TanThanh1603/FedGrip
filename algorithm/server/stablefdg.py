from copy import deepcopy
import random

from algorithm.server.fedavg import FedAvgServer, get_fedavg_argparser
from algorithm.client.stablefdg import StableFDGClient
from data.dataset import FLDataset
from model.stablefdg import StableFDGModel
from utils.tools import get_best_device


def get_stablefdg_argparser():
    parser = get_fedavg_argparser()
    for action in parser._actions:
        if action.dest == 'model':
            action.choices = ['res18', 'res50']
    parser.set_defaults(model='res18')
    parser.add_argument('--stable_exploration', type=float, default=3.0)
    parser.add_argument('--stable_oversampling', type=int, default=32)
    return parser


class StableFDGServer(FedAvgServer):
    def __init__(self, algo='StableFDG', args=None):
        args = get_stablefdg_argparser().parse_args() if args is None else args
        if args.model not in ('res18', 'res50'):
            raise ValueError('StableFDG supports res18/res50 only')
        if args.stable_oversampling < 0 or args.stable_exploration < 0:
            raise ValueError('StableFDG exploration/oversampling must be nonnegative')
        super().__init__(algo, args)
        if self.num_client < 2:
            raise ValueError('StableFDG style sharing needs at least two source clients')
        self.logger.log('StableFDG native adaptation: sharing + exploration/oversampling + AFH; '
                        'project partition, optimizer and aggregation protocol; not exact upstream reproduction')

    def initialize_model(self):
        self.classification_model = StableFDGModel(
            self.args.dataset, self.args.model, self.args.stable_exploration,
            self.args.stable_oversampling)
        self.device = get_best_device(self.args.use_cuda)

    def initialize_clients(self):
        self.client_list = [StableFDGClient(self.args, FLDataset(self.args, cid), cid, self.logger)
                            for cid in range(self.num_client)]

    def process_classification(self):
        weights = deepcopy(self.classification_model.state_dict())
        for client in self.client_list:
            client.load_model_weights(weights)
        self.best_accuracy = 0
        for round_id in range(self.args.round):
            self.round_id = round_id
            self.logger.log('=' * 20, f'Round {round_id}', '=' * 20)
            styles = [client.collect_style() for client in self.client_list]
            # A random cyclic permutation is a derangement: never share own style.
            order = random.sample(range(self.num_client), self.num_client)
            peers = {cid: order[(i + 1) % len(order)] for i, cid in enumerate(order)}
            for cid, client in enumerate(self.client_list):
                # Upstream enables inter-client style sharing after its first round.
                client.peer_style = styles[peers[cid]] if round_id > 0 else None
                client.train()
            weights = self.aggregate_model()
            self.classification_model.load_state_dict(weights)
            for client in self.client_list:
                client.load_model_weights(weights)
            if (round_id + 1) % self.args.test_gap == 0:
                self.validate_and_test()
