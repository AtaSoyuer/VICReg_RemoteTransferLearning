from pathlib import Path
import argparse
import json
import os
import random
import signal
import sys
import time
import urllib

from torch import nn, optim
from torchvision import datasets, transforms
import torch

import resnet

import numpy as np
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json

import glob
from geopy.geocoders import Nominatim
import re
import pandas as pd
from torchvision.io import read_image
from sklearn.preprocessing import LabelEncoder
from torch.utils.tensorboard import SummaryWriter
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import math
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True   #OTHERWISE TRUNCATED IMAGE FILE ERROR SOMEWHERE IN ENUMERATE(DATALOADER)!!!!

import resnet

import torchvision

def get_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained model on ImageNet"
    )

    # Data
    parser.add_argument("--data-dir", type=Path, help="path to dataset")
    parser.add_argument(
        "--train-percent",
        default=100,
        type=int,
        choices=(100, 10, 1),
        help="size of traing set in percent",
    )

    # Checkpoint
    parser.add_argument("--pretrained", type=Path, help="path to pretrained model")
    parser.add_argument(
        "--exp-dir",
        default="./checkpoint/lincls/",
        type=Path,
        metavar="DIR",
        help="path to checkpoint directory",
    )
    parser.add_argument(
        "--print-freq", default=100, type=int, metavar="N", help="print frequency"
    )

    # Model
    parser.add_argument("--arch", type=str, default="resnet50")

    # Optim
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--batch-size", default=256, type=int, metavar="N", help="mini-batch size"
    )
    parser.add_argument(
        "--lr-backbone",
        default=0.0,
        type=float,
        metavar="LR",
        help="backbone base learning rate",
    )
    parser.add_argument(
        "--lr-head",
        default=0.3,
        type=float,
        metavar="LR",
        help="classifier base learning rate",
    )
    parser.add_argument(
        "--weight-decay", default=1e-6, type=float, metavar="W", help="weight decay"
    )
    parser.add_argument(
        "--weights",
        default="freeze",
        type=str,
        choices=("finetune", "freeze"),
        help="finetune or freeze resnet weights",
    )

    # Running
    parser.add_argument(
        "--workers",
        default=8,
        type=int,
        metavar="N",
        help="number of data loader workers",
    )
    
    parser.add_argument('--gpu', default='cuda',
                        help='device to use for training / testing')
    return parser

def main():
    parser = get_arguments()
    args = parser.parse_args()
    if args.train_percent in {1, 10}:
        args.train_files = urllib.request.urlopen(
            f"https://raw.githubusercontent.com/google-research/simclr/master/imagenet_subsets/{args.train_percent}percent.txt"
        ).readlines()
    args.ngpus_per_node = torch.cuda.device_count()
    if "SLURM_JOB_ID" in os.environ:
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
    # single-node distributed training
    args.rank = 0
    args.dist_url = f"tcp://localhost:{random.randrange(49152, 65535)}"
    args.world_size = args.ngpus_per_node
    
    main_worker(args.gpu, args)




def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def main_worker(gpu, args):

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    #backbone, embedding = resnet.__dict__[args.arch](zero_init_residual=True)
    #state_dict = torch.load(args.pretrained, map_location="cpu")
    #if "model" in state_dict:
        #state_dict = state_dict["model"]
        #state_dict = {
            #key.replace("module.backbone.", ""): value
            #for (key, value) in state_dict.items()
        #}
    #backbone.load_state_dict(state_dict, strict=False)
    
    #backbone, embedding = torch.hub.load('facebookresearch/vicreg:main', 'resnet50'), 2048
    
    backbone, embedding = torch.load('exp/model.pth'), 2048

    head = nn.Linear(embedding, 1000)                            #CHANGE ACCORDING TO NUMBER OF CLASSES!!!!!
    head.weight.data.normal_(mean=0.0, std=0.01)
    head.bias.data.zero_()
    model = nn.Sequential(backbone, head)
    model.cuda(gpu)

    if args.weights == "freeze":
        backbone.requires_grad_(False)
        head.requires_grad_(True)

    criterion = nn.CrossEntropyLoss().cuda(gpu)

    param_groups = [dict(params=head.parameters(), lr=args.lr_head)]
    
    if args.weights == "finetune":
        param_groups.append(dict(params=backbone.parameters(), lr=args.lr_backbone))
        
    optimizer = optim.SGD(param_groups, 0, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    
    start_epoch = 0
    best_acc = argparse.Namespace(top1=0, top5=0)

    # Data loading code
    traindir = args.data_dir / "train"
    valdir = args.data_dir / "val"
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    if args.train_percent in {1, 10}:
        train_dataset.samples = []
        for fname in args.train_files:
            fname = fname.decode().strip()
            cls = fname.split("_")[0]
            train_dataset.samples.append(
                (traindir / cls / fname, train_dataset.class_to_idx[cls])
            )

    #train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    kwargs = dict(
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        pin_memory=True,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle = True, **kwargs
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle= True, **kwargs)

    start_time = time.time()
    #WEIGHT FINETUNING/TRAINIG
    
    running_loss = 0.0
    checkpoints = 30
    n_total_steps = len(train_loader)
    
    for epoch in range(start_epoch, args.epochs):
        # TRAIN
        
        if args.weights == "finetune":
            model.train()
        elif args.weights == "freeze":
            model.eval()
        else:
            assert False

        for step, (images, target) in enumerate(
            train_loader, start=epoch * len(train_loader)
        ):
            
            output = model(images.cuda(gpu, non_blocking=True))
            loss = criterion(output, target.cuda(gpu, non_blocking=True))
            
            running_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (step+1) % checkpoints == 0:
            
                print (f'Epoch [{epoch+1}/{args.epochs}], Step [{step+1}/{n_total_steps}], Loss: {loss.item():.4f}')
                print('training_loss', running_loss/checkpoints, epoch * n_total_steps + step)
                running_loss = 0.0

        # EVALUATE
        
        model.eval()
        if args.rank == 0:
            top1 = AverageMeter("Acc@1")
            top5 = AverageMeter("Acc@5")
            with torch.no_grad():
                for images, target in val_loader:
                    
                    images = images.view(-1, 3, 224, 224)    
                    target = target.flatten()
                    target = target.type(torch.LongTensor)
            
                    output = model(images.cuda(gpu, non_blocking=True))
                    acc1, acc5 = accuracy(
                        output, target.cuda(gpu, non_blocking=True), topk=(1, 5)
                    )
                    top1.update(acc1[0].item(), images.size(0))
                    top5.update(acc5[0].item(), images.size(0))
            best_acc.top1 = max(best_acc.top1, top1.avg)
            best_acc.top5 = max(best_acc.top5, top5.avg)
            print('best_acc.top1', best_acc.top1)
            print('best_acc.top5', best_acc.top5)
            
        scheduler.step()
        

def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


if __name__ == "__main__":
    main()

