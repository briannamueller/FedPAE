#!/usr/bin/env python
import os
import time
import warnings
import logging
from pathlib import Path

import numpy as np
import torch

from flcore.servers.serverpae import FedPAE
from arg_parser import build_arg_parser

logger = logging.getLogger()
logger.setLevel(logging.ERROR)
warnings.simplefilter("ignore")
torch.manual_seed(0)


def run(args):
    time_list = []

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        if args.model_family == "HtFE-img-2-gray":
            args.models = [
                'FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024)',
                'FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024)',
            ]

        elif args.model_family == "HtFE-img-2":
            args.models = [
                'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)',
                'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
            ]

        elif args.model_family == "HtFE-img-3":
            args.models = [
                'resnet10(num_classes=args.num_classes)',
                'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
                'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)',
            ]

        elif args.model_family == "HtFE-img-4":
            args.models = [
                'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)',
                'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)',
                'mobilenet_v2(pretrained=False, num_classes=args.num_classes)',
                'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
            ]

        elif args.model_family == "HtFE-img-5":
            args.models = [
                'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)',
                'mobilenet_v2(pretrained=False, num_classes=args.num_classes)',
                'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
                'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)',
                'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)',
            ]

        elif args.model_family == "HtFE-img-8":
            args.models = [
                'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)',
                'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)',
                'mobilenet_v2(pretrained=False, num_classes=args.num_classes)',
                'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
                'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)',
                'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)',
                'torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)',
                'torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)',
            ]

        else:
            raise NotImplementedError(f"Unknown model_family: {args.model_family}")

        for model in args.models:
            print(model)

        server = FedPAE(args, i)
        server.train()

        time_list.append(time.time() - start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    print("All done!")


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("cuda is not available, falling back to cpu.")
            args.device = "cpu"
        elif args.device_id:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    args.ckpt_root = Path(args.ckpt_root).expanduser()
    args.outputs_root = Path(getattr(args, "outputs_root", args.ckpt_root / "outputs")).expanduser()

    print("=" * 50)
    for arg in vars(args):
        print(arg, '=', getattr(args, arg))
    print("=" * 50)

    run(args)
