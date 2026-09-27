from algorithm.server.fedavg import FedAvgServer, get_fedavg_argparser
from algorithm.client.fedsam import FedSAMClient
from data.dataset import FLDataset


def get_fedsam_argparser():
    parser = get_fedavg_argparser()
    parser.add_argument('--sam_rho', type=float, default=0.1,
                        help='SAM perturbation radius; upstream FedOMG FedSAM script default')
    return parser


class FedSAMServer(FedAvgServer):
    def __init__(self, algo='FedSAM', args=None):
        args = get_fedsam_argparser().parse_args() if args is None else args
        if args.sam_rho < 0:
            raise ValueError('sam_rho must be nonnegative')
        super().__init__(algo, args)

    def initialize_clients(self):
        self.client_list = [FedSAMClient(self.args, FLDataset(self.args, cid), cid, self.logger)
                            for cid in range(self.num_client)]
