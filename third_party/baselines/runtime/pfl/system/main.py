#!/usr/bin/env python
import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging
import random

from torch import nn
from flcore.servers.serverrod import FedROD
from flcore.servers.serverbabu import FedBABU
from flcore.servers.serverpac import FedPAC
from flcore.servers.serverperavg import PerAvg
from flcore.trainmodel.models import FedAvgCNN, BaseHeadSplit

from utils.result_utils import average_data
from utils.mem_utils import MemReporter

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)

# hyper-params for Text tasks
vocab_size = 98635
max_len = 200
emb_dim = 32


def run(args):
    time_list = []
    reporter = MemReporter()
    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        started = time.time()
        name = args.dataset.lower()
        if "mnist" in name:
            args.model = FedAvgCNN(
                in_features=1, num_classes=args.num_classes, dim=1024
            ).to(args.device)
        elif "cifar" in name:
            args.model = FedAvgCNN(
                in_features=3, num_classes=args.num_classes, dim=1600
            ).to(args.device)
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")
        print(args.model)
        if args.algorithm == "PerAvg":
            server = PerAvg(args, i)
        elif args.algorithm in {"FedROD", "FedBABU", "FedPAC"}:
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server_class = {
                "FedROD": FedROD, "FedBABU": FedBABU, "FedPAC": FedPAC
            }[args.algorithm]
            server = server_class(args, i)
        else:
            raise ValueError(f"Unsupported baseline: {args.algorithm}")
        server.train()
        time_list.append(time.time() - started)
    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    average_data(
        dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times
    )
    print("All done!")
    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="test",
                        help="The goal for this experiment")
    parser.add_argument("--seed", type=int, default=39)
    #     parser.add_argument('-dev', "--device", type=str, default="cuda",
    #                         choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=int, default=0)
    parser.add_argument('-data', "--dataset", type=str, default="mnist")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-mstr', "--model_str", type=str, default="cnn")
    parser.add_argument('-m', "--model", type=str, default="cnn")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.05,
                        help="Local learning rate")
    # parser.add_argument('-gld', "--global_learning_rate_decay", type=bool, default=True)
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=10)
    parser.add_argument('-ls', "--local_epochs", type=int, default=1,
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-dp', "--privacy", type=bool, default=False,
                        help="differential privacy")
    parser.add_argument('-dps', "--dp_sigma", type=float, default=0.0)
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=1)
    parser.add_argument('-log', "--log", action='store_true')
    # data
    parser.add_argument("--noniid", action='store_true')
    parser.add_argument('--balance', action='store_true')
    parser.add_argument('--alpha_dirich', default=0.1, type=float)

    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    # FedFomo
    parser.add_argument('-M', "--M", type=int, default=5,
                        help="Server only sends M client models to one client at each round")
    # APFL
    parser.add_argument('-al', "--alpha", type=float, default=1.0)
    # Ditto / FedRep
    parser.add_argument('-pls', "--plocal_steps", type=int, default=1)
    # MOON
    parser.add_argument('-tau', "--tau", type=float, default=1.0)
    # FedBABU
    parser.add_argument('-fts', "--fine_tuning_epochs", type=int, default=10)
    # FedPAC prototype regularization (missing from the released parser).
    parser.add_argument('-lam', "--lamda", type=float, default=1.0)
    # APPLE
    parser.add_argument('-dlr', "--dr_learning_rate", type=float, default=0.0)
    parser.add_argument('-L', "--L", type=float, default=1.0)
    # FedGen
    parser.add_argument('-nd', "--noise_dim", type=int, default=512)
    parser.add_argument('-glr', "--generator_learning_rate", type=float, default=0.005)
    parser.add_argument('-hd', "--hidden_dim", type=int, default=512)
    parser.add_argument('-se', "--server_epochs", type=int, default=1000)
    parser.add_argument('-lf', "--localize_feature_extractor", type=bool, default=False)
    # SCAFFOLD
    parser.add_argument('-slr', "--server_learning_rate", type=float, default=1.0)
    # FedALA
    parser.add_argument('-et', "--eta", type=float, default=1.0)
    parser.add_argument('-s', "--rand_percent", type=int, default=80)
    parser.add_argument('-p', "--layer_idx", type=int, default=2,
                        help="More fine-grained than its original paper.")
    # FedKD
    parser.add_argument('-mlr', "--mentee_learning_rate", type=float, default=0.005)
    parser.add_argument('-Ts', "--T_start", type=float, default=0.95)
    parser.add_argument('-Te', "--T_end", type=float, default=0.98)
    # GPFL
    parser.add_argument('-lamr', "--lamda_reg", type=float, default=0.0)

    # FedOMG
    parser.add_argument('-omgr',"--omg_rounds", type=int, default=25, help='Number of omg rounds')
    parser.add_argument('-omglr',"--omg_learning_rate", type=float, default=25, help='omg learning rate')
    parser.add_argument('-mmt',"--omg_momentum", type=float, default=0.1)
    parser.add_argument('-ss',"--omg_step_size", type=int, default=30, help='omg step size')
    parser.add_argument('-gam',"--omg_gamma", type=float, default=0.5, help='omg gamma')
    parser.add_argument('--omg_c', type=float, default=0.25, help='omg c parameter')
    parser.add_argument('--omg_meta_lr', type=float, default=1.0, help='omg meta learning rate')
    parser.add_argument('--grad_balance', action='store_true', help='Enable gradient balancing')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=args.device_id)

    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local steps: {}".format(args.local_epochs))
    print("Local learning rate: {}".format(args.local_learning_rate))
    print("Local learning rate decay: {}".format(args.learning_rate_decay))
    if args.learning_rate_decay:
        print("Local learning rate decay gamma: {}".format(args.learning_rate_decay_gamma))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Clients randomly join: {}".format(args.random_join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Client select regarding time: {}".format(args.time_select))
    if args.time_select:
        print("Time thresh hold: {}".format(args.time_threthold))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Number of classes: {}".format(args.num_classes))
    print("Backbone: {}".format(args.model))
    print("Using device: {}".format(args.device))
    print("Using DP: {}".format(args.privacy))
    if args.privacy:
        print("Sigma for DP: {}".format(args.dp_sigma))
    print("Auto break: {}".format(args.auto_break))
    if not args.auto_break:
        print("Global rounds: {}".format(args.global_rounds))
    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("DLG attack: {}".format(args.dlg_eval))
    if args.dlg_eval:
        print("DLG attack round gap: {}".format(args.dlg_gap))
    print("Total number of new clients: {}".format(args.num_new_clients))
    print("Fine tuning epoches on new clients: {}".format(args.fine_tuning_epoch_new))
    print("=" * 50)

    run(args)
