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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import os
import glob
from geopy.geocoders import Nominatim
import re
import pandas as pd
from torchvision.io import read_image
from sklearn.preprocessing import LabelEncoder
from torch.utils.tensorboard import SummaryWriter
import sklearn
from torchvision.transforms import InterpolationMode
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import math
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True 

import torch.distributed as dist

# In[14]:


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
torch.cuda.empty_cache()

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

from munkres import Munkres

import faiss

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.nn as nn

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

from fmow_dataloader_evaluate import CustomDatasetFromImages


#torch.multiprocessing.set_sharing_strategy('file_system')

# In[3]:


def get_y_preds(cluster_assignments, y_true, n_clusters):
    '''
    Computes the predicted labels, where label assignments now
    correspond to the actual labels in y_true (as estimated by Munkres)

    cluster_assignments:    array of labels, outputted by kmeans
    y_true:                 true labels
    n_clusters:             number of clusters in the dataset

    returns:    a tuple containing the accuracy and confusion matrix,
                in that order
    '''
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true, cluster_assignments, labels=None)
    # compute accuracy based on optimal 1:1 assignment of clusters to labels
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred, confusion_matrix 

def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))

    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:,j]) # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i,j]
            cost_matrix[j,i] = s-t
    return cost_matrix

def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    clusterLabels = np.zeros(n_clusters)
    for i in range(n_clusters):
        clusterLabels[i] = indices[i][1]
    return clusterLabels

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

    return parser

# FASTER KNN IMPLEMENTATION!!!!

class FaissKNeighbors:
    def __init__(self, k):
        self.index = None
        self.y = None
        self.k = k

    def fit(self, X, y):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))
        self.y = y

    def predict(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        votes = self.y[indices]
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes.astype(np.int64)])
        return predictions

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
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)


def main_worker(gpu, args):
    
    print('yes')
    
    train_csv = "./fmow/csv_file.csv"
    val_csv = "./fmow/csv_file_val.csv"
    
    args.rank += gpu
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    if args.rank == 0:
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
        print(" ".join(sys.argv))
        print(" ".join(sys.argv), file=stats_file)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    #supervised_model_pretrained = torchvision.models.resnet50(pretrained=True)
    #torch.save(supervised_model_pretrained.state_dict(), 'resnet50_imagenet_pretrained_supervised.pth')


# In[8]:


    backbone, embedding = resnet.__dict__['resnet50'](zero_init_residual=True)
    state_dict = torch.load('./exp/fmow_vicreg_bartu.pth', map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
        state_dict = {key.replace("module.backbone.", ""): value for (key, value) in state_dict.items()}
    backbone.load_state_dict(state_dict, strict=False)
    
    backbone.cuda(gpu)
    #backbone.requires_grad_(False)    #Otherwise DDP is redundant error
    backbone = torch.nn.parallel.DistributedDataParallel(backbone, device_ids=[gpu])
    
    print('cool')



    # Data loading code
    
    per_device_batch_size = args.batch_size // args.world_size

    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #transforms = torchvision.transforms.Compose([torchvision.transforms.RandomResizedCrop(224), torchvision.transforms.ToTensor(),normalize])
    train_transforms = transforms.Compose(
            [   transforms.Resize(224*2),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

    val_transforms = transforms.Compose(
            [
                transforms.Resize(224*2),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )

    #train_dataset = datasets.ImageFolder('./ImageNet/ILSVRC/Data/CLS-LOC/train' , train_transforms)
    #train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)                                                  
    #trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=per_device_batch_size,
                                            #sampler=train_sampler, num_workers=8, pin_memory=True)
    ##trainloader = DataLoader(dataset=train_dataset, batch_size = batch_size, shuffle = False)

    #val_dataset = datasets.ImageFolder('./ImageNet/ILSVRC/Data/CLS-LOC/val' , val_transforms)
    ##val_sampler =DistributedSampler(dataset=val_dataset, shuffle=True)                                         
    ##valloader = torch.utils.data.DataLoader(val_dataset, batch_size=per_device_batch_size,
                                            #shuffle=False, sampler=val_sampler, num_workers=10)
    #valloader = torch.utils.data.DataLoader(val_dataset, batch_size=per_device_batch_size, num_workers=8, pin_memory=True)
    ##valloader = DataLoader(dataset=val_dataset, batch_size = batch_size, shuffle = False)
    
    kwargs = dict(batch_size=args.batch_size // args.world_size, num_workers=args.workers, pin_memory=True,)
        
    train_dataset = CustomDatasetFromImages(train_csv, transform=train_transforms)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    per_device_batch_size = args.batch_size // args.world_size
    trainloader = torch.utils.data.DataLoader(
        train_dataset, sampler=train_sampler, **kwargs)
    
    val_dataset = CustomDatasetFromImages(val_csv, transform=val_transforms)
    valloader = torch.utils.data.DataLoader(val_dataset, **kwargs)
    
    print('TrainDatasetSize:', train_dataset.__len__())
    print('ValDatasetSize:', val_dataset.__len__())
    
    print('Trainloader size:', len(trainloader))
    print('Valloader size:', len(valloader))

    
    epoch = 0    
    backbone.eval()
    
    #train_sampler.set_epoch(epoch)
    
    #labels_list = []
    #embeddings_list = []

    labels_unseen_list = []
    embeddings_unseen_list = []
    
    #labels_list = torch.ones(train_dataset.__len__(), dtype=torch.int8, device='cpu')
    #embeddings_list = torch.ones([train_dataset.__len__(), embedding], dtype=torch.float32, device='cpu')
        
    start_time = time.time()
    
    embeddings_seen_arr = np.zeros((train_dataset.__len__(), 2048))
    labels_seen_arr = np.zeros(train_dataset.__len__())
    
    with torch.no_grad():
        
        counter = 0
    
        for i, (inputs, labels) in enumerate(trainloader):
            
            embedding = backbone(inputs.cuda(gpu, non_blocking=True)).cpu().detach().numpy()
            
            if embedding.shape[0] == per_device_batch_size:
                #print(len(embedding))
                embeddings_seen_arr[counter:counter+per_device_batch_size,:] = embedding
                #counter += per_device_batch_size
            else:
                #print(len(embedding))
                embeddings_seen_arr[counter:counter+embedding.shape[0],:] = embedding
                #counter += (per_device_batch_size-len(trainloader)%per_device_batch_size)

            if labels.shape[0] == per_device_batch_size:
                labels_seen_arr[counter:counter+per_device_batch_size] = labels.cpu().detach().numpy()
            else:
                labels_seen_arr[counter:counter+labels.shape[0]] = labels.cpu().detach().numpy()
                
            counter += per_device_batch_size
            
            if i%1000 == 0:
                
                print(i)
                #print(embeddings_list)
                current_time = time.time()
                elapsed_time = current_time - start_time
                print('time:',elapsed_time)

    
        #for i, (inputs, labels) in enumerate(trainloader):
        
            #if i == 50000:
                #break

            
            #embedding = backbone(inputs.cuda(gpu, non_blocking=True)).cpu()
            #print(embedding.shape)
            #embeddings_list.append(embedding)
            #embeddings_list = torch.cat((embeddings_list, embedding), dim=0)
            #labels_list.append(labels.cpu().detach().numpy())
            #print(labels.shape)
            #labels_list = torch.cat((labels_list, labels.cpu()))
            
            #if i%1000 == 0:
                
                #print(i)
                #print(embeddings_list)
                #current_time = time.time()
                #elapsed_time = current_time - start_time
                #print('time:',elapsed_time)
    
    with torch.no_grad():
    
        for i, (inputs, labels) in enumerate(valloader):
        
            #if i == 50000:
                #break
            
            if i%1000 == 0:
                print(i)
            
    
            embedding = backbone(inputs.cuda(gpu, non_blocking=True)).cpu().detach().numpy()
            embeddings_unseen_list.append(embedding)
            labels_unseen_list.append(labels.cpu().detach().numpy())
            
    
    #USE THIS IF BATCHSIZE>1
    #embeddings_seen_arr = np.zeros((len(trainloader)*per_device_batch_size-len(trainloader)%per_device_batch_size, 2048))
    #counter = 0
    #for embedding in embeddings_list:
        #if len(embedding) == per_device_batch_size:
            #print(len(embedding))
            #embeddings_seen_arr[counter:counter+baper_device_batch_sizetch_size,:] = embedding
            #counter += per_device_batch_size
        #else:
            #print(len(embedding))
            #embeddings_seen_arr[counter:counter+(per_device_batch_size-len(trainloader)%per_device_batch_size),:] = embedding
            #counter += (per_device_batch_size-len(trainloader)%per_device_batch_size)

    #print(embeddings_seen_arr)
    #labels_seen_arr = np.zeros(len(trainloader)*per_device_batch_size-len(trainloader)%per_device_batch_size)
    #counter = 0
    #for i in labels_list:
    #    if len(i) == per_device_batch_size:
    #        labels_seen_arr[counter:counter+per_device_batch_size] = i
    #    else:
    #        labels_seen_arr[counter:counter+(per_device_batch_size-len(trainloader)%per_device_batch_size)] = i
    #    counter += per_device_batch_size
    #print(np.unique(labels_seen_arr))
    
    #embeddings_seen_arr = np.array(embeddings_list)
    #labels_seen_arr = np.array(labels_list)

    embeddings_unseen_arr = np.zeros((val_dataset.__len__(), 2048))
    counter = 0
    for embedding in embeddings_unseen_list:
        embeddings_unseen_arr[counter:counter+len(embedding),:] = embedding
        counter += len(embedding)

    #print(embeddings_unseen_arr)
    labels_unseen_arr = np.zeros(val_dataset.__len__())
    counter = 0
    for i in labels_unseen_list:
        labels_unseen_arr[counter:counter+len(i)] = i
        counter += len(i)
    print(np.unique(labels_unseen_arr))

    #embeddings_seen_arr = np.delete(embeddings_seen_arr,0,axis=0)
    #labels_seen_arr = np.delete(labels_seen_arr,0)
    #embeddings_unseen_arr = np.delete(embeddings_unseen_arr,0,axis=0)
    #labels_unseen_arr = np.delete(labels_unseen_arr,0)
    


# In[14]:


    n_neighbours = [20, 200]
    for k in n_neighbours:
    
        #knn = KNeighborsClassifier(n_neighbors=k)
        knn = FaissKNeighbors(k = k)
        knn.fit(embeddings_seen_arr, labels_seen_arr)
        labels_predicted = knn.predict(embeddings_unseen_arr)
        current_time = time.time()
        elapsed_time = current_time - start_time
        print('time:',elapsed_time)

        print('Accuracy' + str(20) + ':', np.sum(labels_unseen_arr == labels_predicted)/len(labels_unseen_arr))
        #print('Accuracy:', knn_20.score(embeddings_unseen_arr, labels_unseen_arr))


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass

if __name__ == "__main__":
    main()