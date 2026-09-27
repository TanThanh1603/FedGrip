from argparse import ArgumentParser, Namespace
from copy import deepcopy
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(PROJECT_DIR.as_posix())

from algorithm.server.fedavg import FedAvgServer, get_fedavg_argparser
from algorithm.client.fedsr import FedSRClient
from data.dataset import FLDataset


def get_fedsr_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("--L2R_coeff", type=float, default=1e-3)
    parser.add_argument("--CMI_coeff", type=float, default=1e-3)
    return parser


class FedSRServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "FedSR",
        args: Namespace = None,
    ):
        if args is None:
            args = get_fedsr_argparser().parse_args()
        args.model = f"fedsr_{args.model}"
        super().__init__(algo, args)

    def initialize_clients(self):
        self.client_list = [
            FedSRClient(self.args, FLDataset(self.args, client_id), client_id, self.logger)
            for client_id in range(self.num_client)
        ]


if __name__ == "__main__":
    server = FedSRServer()
    server.process_classification()
