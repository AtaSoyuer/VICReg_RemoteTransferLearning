#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[4]:

batch_size = 24
per_device_batch_size = batch_size // 3

# In[13]:


#DDP ATTEMPT

def init_distributed():

    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
    dist_url = "env://" # default

    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            rank=rank)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)

    # synchronizes all the threads to reach this point before moving on
    dist.barrier()
    
    print('world_size:',world_size)


init_distributed()

# In[5]:


normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#transforms = torchvision.transforms.Compose([torchvision.transforms.RandomResizedCrop(224), torchvision.transforms.ToTensor(),normalize])
train_transforms = transforms.Compose(
            [   transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

val_transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )

train_dataset = datasets.ImageFolder('./ImageNet/ILSVRC/Data/CLS-LOC/train' , train_transforms)
train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)                                                  
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=per_device_batch_size,
                                            sampler=train_sampler, num_workers=10, pin_memory=True)
#trainloader = DataLoader(dataset=train_dataset, batch_size = batch_size, shuffle = False)

val_dataset = datasets.ImageFolder('./ImageNet/ILSVRC/Data/CLS-LOC/val' , val_transforms)
val_sampler =DistributedSampler(dataset=val_dataset, shuffle=True)                                         
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=per_device_batch_size,
                                            shuffle=False, sampler=val_sampler, num_workers=10)
#valloader = DataLoader(dataset=val_dataset, batch_size = batch_size, shuffle = False)


# In[6]:


#torch.cuda.current_device()
#torch.cuda.set_device(0)
#print(torch.cuda.mem_get_info(torch.cuda.current_device()))
#device = torch.cuda.current_device()


# In[7]:


supervised_model_pretrained = torchvision.models.resnet50(pretrained=True)
torch.save(supervised_model_pretrained.state_dict(), 'resnet50_imagenet_pretrained_supervised.pth')


# In[8]:


backbone, embedding = resnet.__dict__['resnet50'](zero_init_residual=True)
state_dict = torch.load('resnet50_imagenet_pretrained_supervised.pth', map_location="cpu")
if "model" in state_dict:
    state_dict = state_dict["model"]
    state_dict = {key.replace("module.backbone.", ""): value for (key, value) in state_dict.items()}
backbone.load_state_dict(state_dict, strict=False)
gpu = torch.device('cuda')
backbone.cuda(gpu)

backbone = nn.SyncBatchNorm.convert_sync_batchnorm(backbone)

local_rank = int(os.environ['LOCAL_RANK'])
backbone = nn.parallel.DistributedDataParallel(backbone, device_ids=[local_rank])


# In[9]:


print(embedding)
backbone.eval()


# In[10]:


labels_list = []
embeddings_list = []

labels_unseen_list = []
embeddings_unseen_list = []

with torch.no_grad():
    
    for i, (inputs, labels) in enumerate(trainloader):
        
        #if i == 50000:
            #break
            
        if i%1000 == 0:
            print(i)
            #print(embeddings_list)

            
        embedding = backbone(inputs.cuda(gpu, non_blocking=True)).cpu().detach().numpy()
        embeddings_list.append(embedding)
        labels_list.append(labels.cpu().detach().numpy())

    


# In[11]:


with torch.no_grad():
    
    for i, (inputs, labels) in enumerate(valloader):
        
        #if i == 50000:
            #break
            
        if i%1000 == 0:
            print(i)
            
    
        embedding = backbone(inputs.cuda(gpu, non_blocking=True)).cpu().detach().numpy()
        embeddings_unseen_list.append(embedding)
        labels_unseen_list.append(labels.cpu.detach().numpy())


# In[12]:


#USE THIS IF BATCHSIZE>1
embeddings_seen_arr = np.zeros((len(trainloader)*batch_size-len(trainloader)%batch_size, 2048))
counter = 0
for embedding in embeddings_list:
    if len(embedding) == batch_size:
        #print(len(embedding))
        embeddings_seen_arr[counter:counter+batch_size,:] = embedding
        counter += batch_size
    else:
        #print(len(embedding))
        embeddings_seen_arr[counter:counter+(batch_size-len(trainloader)%batch_size),:] = embedding
        counter += (batch_size-len(trainloader)%batch_size)

print(embeddings_seen_arr)
labels_seen_arr = np.zeros(len(trainloader)*batch_size-len(trainloader)%batch_size)
counter = 0
for i in labels_list:
    if len(i) == batch_size:
        labels_seen_arr[counter:counter+batch_size] = i
    else:
        labels_seen_arr[counter:counter+(batch_size-len(trainloader)%batch_size)] = i
    counter += batch_size
print(np.unique(labels_seen_arr))

embeddings_unseen_arr = np.zeros((len(valloader)*batch_size, 2048))
counter = 0
for embedding in embeddings_unseen_list:
    embeddings_unseen_arr[counter:counter+batch_size,:] = embedding.cpu().detach().numpy()
    counter += batch_size

print(embeddings_unseen_arr)
labels_unseen_arr = np.zeros(len(valloader)*batch_size)
counter = 0
for i in labels_unseen_list:
    labels_unseen_arr[counter:counter+batch_size] = i.detach().numpy()
    counter += batch_size
print(np.unique(labels_unseen_arr))

embeddings_seen_arr = np.delete(embeddings_seen_arr,0,axis=0)
labels_seen_arr = np.delete(labels_seen_arr,0)
embeddings_unseen_arr = np.delete(embeddings_unseen_arr,0,axis=0)
labels_unseen_arr = np.delete(labels_unseen_arr,0)


# In[11]:


#embeddings_seen_arr = np.zeros((int(3*len(dataloader)/4), 2048))
#counter = 0
#for embedding in embeddings_list:
#    embeddings_seen_arr[counter,:] = embedding.cpu().detach().numpy()
##    counter += 1

#print(embeddings_seen_arr)
#labels_seen_arr = np.zeros(int(3*len(dataloader)/4))
#counter = 0
#for i in labels_list:
#    labels_seen_arr[counter] = i.detach().numpy()
#    counter += 1
#print(np.unique(labels_seen_arr))

#embeddings_unseen_arr = np.zeros((int(len(dataloader)/4), 2048))
#counter = 0
#for embedding in embeddings_unseen_list:
#    embeddings_unseen_arr[counter,:] = embedding.cpu().detach().numpy()
#    counter += 1

##print(embeddings_unseen_arr)
#labels_unseen_arr = np.zeros(int(len(dataloader)/4))
#counter = 0
#for i in labels_unseen_list:
#    labels_unseen_arr[counter] = i.detach().numpy()
#    counter += 1
#print(np.unique(labels_unseen_arr))


# In[13]:


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


# In[14]:


n_neighbours = [20, 200]
for k in n_neighbours:
    
    #knn = KNeighborsClassifier(n_neighbors=k)
    knn = FaissKNeighbors(k = k)
    knn.fit(embeddings_seen_arr, labels_seen_arr)
    labels_predicted = knn.predict(embeddings_unseen_arr)

    print('Accuracy' + str(k) + ':', np.sum(labels_unseen_arr == labels_predicted)/len(labels_unseen_arr))
    #print('Accuracy:', knn.score(embeddings_unseen_arr, labels_unseen_arr))


# In[12]:


#JUST TRY DELETE LATER

#from sklearn.decomposition import PCA
#from sklearn.manifold import TSNE

#embeddings_unseen_arr_trial = embeddings_unseen_arr[0:180,:]
#labels_unseen_arr_trial = labels_unseen_arr[0:180]

#pca = PCA(n_components=2)
#pca.fit(embeddings_unseen_arr)
#print(pca.explained_variance_ratio_)

#embeddings_reduced = pca.transform(embeddings_unseen_arr)
#print(embeddings_reduced)

#u_labels = np.unique(labels_unseen_arr_trial)
#print(u_labels)

#embeddings_reduced = TSNE(n_components=2, learning_rate='auto',init='random', perplexity = 10.0).fit_transform(embeddings_unseen_arr_trial)

#for i in u_labels:
    
#    plt.scatter(embeddings_reduced[labels_unseen_arr_trial == i , 0] , embeddings_reduced[labels_unseen_arr_trial == i , 1] , label = i)

#plt.legend()
#plt.show()


# In[13]:


from sklearn.neighbors import KNeighborsClassifier

n_neighbours = [1, 3, 5, 10, 20, 30, 40, 50]

for k in n_neighbours:
    
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(embeddings_seen_arr, labels_seen_arr)
    labels_predicted = knn.predict(embeddings_unseen_arr)

    #print('Accuracy' + str(k) + ':', np.sum(labels_unseen_arr == labels_predicted)/len(labels_unseen_arr))
    print('Accuracy:', knn.score(embeddings_unseen_arr, labels_unseen_arr))


# In[16]:


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=1000, random_state=0).fit(embeddings_seen_arr)
labels_predicted = kmeans.predict(embeddings_unseen_arr)
print(labels_predicted)

from sklearn import metrics

print(metrics.rand_score(labels_unseen_arr, labels_predicted))

truth = labels_unseen_arr
pred=labels_predicted
print(np.sum( get_y_preds(pred, truth, 1000)[0] == truth )/len(truth) )


# # VICREG

# In[8]:


#TRY LIKE THIS
#backbone = torch.hub.load('facebookresearch/vicreg:main', 'resnet50')
#backbone.to(device)
#backbone.eval()


# In[ ]:


vicreg_model_pretrained = torch.hub.load('facebookresearch/vicreg:main', 'resnet50')
torch.save(vicreg_model_pretrained.state_dict(), 'resnet50_imagenet_pretrained_vicreg.pth')


# In[29]:


backbone, embedding = resnet.__dict__['resnet50'](zero_init_residual=True)
state_dict = torch.load('resnet50_imagenet_pretrained_vicreg.pth', map_location="cpu")
if "model" in state_dict:
    state_dict = state_dict["model"]
    state_dict = {key.replace("module.backbone.", ""): value for (key, value) in state_dict.items()}
backbone.load_state_dict(state_dict, strict=False)
backbone.cuda()

backbone = nn.SyncBatchNorm.convert_sync_batchnorm(backbone)

local_rank = int(os.environ['LOCAL_RANK'])
backbone = nn.parallel.DistributedDataParallel(backbone, device_ids=[local_rank])


# In[30]:


print(embedding)
backbone.eval()


# In[ ]:


labels_list = []
embeddings_list = []

labels_unseen_list = []
embeddings_unseen_list = []

with torch.no_grad():
    
    for i, (inputs, labels) in enumerate(trainloader):
        
        #if i == 50000:
            #break
            
        if i%100 == 0:
            print(i)
            
        
        embedding = backbone(inputs.to(device))
        embeddings_list.append(embedding)
        labels_list.append(labels)
        
        #torch.cuda.empty_cache()
            


# In[ ]:


with torch.no_grad():
    
    for i, (inputs, labels) in enumerate(valloader):
        
        #if i == 50000:
            #break
            
        if i%1000 == 0:
            print(i)
            
    
        embedding = backbone(inputs.to(device))
        embeddings_unseen_list.append(embedding)
        labels_unseen_list.append(labels)


# In[ ]:


#USE THIS IF BATCHSIZE>1
embeddings_seen_arr = np.zeros((len(trainloader)*batch_size-3, 2048))
counter = 0
for embedding in embeddings_list:
    if len(embedding) == 10:
        #print(len(embedding))
        embeddings_seen_arr[counter:counter+batch_size,:] = embedding.cpu().detach().numpy()
        counter += batch_size
    else:
        #print(len(embedding))
        embeddings_seen_arr[counter:counter+7,:] = embedding.cpu().detach().numpy()
        counter += 7

print(embeddings_seen_arr)
labels_seen_arr = np.zeros(len(trainloader)*batch_size-3)
counter = 0
for i in labels_list:
    if len(i) == 10:
        labels_seen_arr[counter:counter+batch_size] = i.detach().numpy()
    else:
        labels_seen_arr[counter:counter+7] = i.detach().numpy()
    counter += batch_size
print(np.unique(labels_seen_arr))

embeddings_unseen_arr = np.zeros((len(valloader)*batch_size, 2048))
counter = 0
for embedding in embeddings_unseen_list:
    embeddings_unseen_arr[counter:counter+batch_size,:] = embedding.cpu().detach().numpy()
    counter += batch_size

print(embeddings_unseen_arr)
labels_unseen_arr = np.zeros(len(valloader)*batch_size)
counter = 0
for i in labels_unseen_list:
    labels_unseen_arr[counter:counter+batch_size] = i.detach().numpy()
    counter += batch_size
print(np.unique(labels_unseen_arr))

embeddings_seen_arr = np.delete(embeddings_seen_arr,0,axis=0)
labels_seen_arr = np.delete(labels_seen_arr,0)
embeddings_unseen_arr = np.delete(embeddings_unseen_arr,0,axis=0)
labels_unseen_arr = np.delete(labels_unseen_arr,0)


# In[81]:


#for i in range(embeddings_seen_arr.shape[0]):
    #if len(np.unique(embeddings_seen_arr[i,:]))==1:
        #print('yes')
for i in range(embeddings_unseen_arr.shape[0]):
    if len(np.unique(embeddings_unseen_arr[i,:]))==1:
        print('yes_unseen')


# In[35]:


#embeddings_seen_arr = np.zeros((len(trainloader), 2048))
#counter = 0
#for embedding in embeddings_list:
#    embeddings_seen_arr[counter,:] = embedding.cpu().detach().numpy()
#    counter += 1

#print(embeddings_seen_arr)
#labels_seen_arr = np.zeros(len(trainloader))
#counter = 0
#for i in labels_list:
#    labels_seen_arr[counter] = i.detach().numpy()
#    counter += 1
#print(np.unique(labels_seen_arr))

#embeddings_unseen_arr = np.zeros((len(valloader), 2048))
#counter = 0
#for embedding in embeddings_unseen_list:
#    embeddings_unseen_arr[counter,:] = embedding.cpu().detach().numpy()
#    counter += 1

#print(embeddings_unseen_arr)
#labels_unseen_arr = np.zeros(len(valloader))
#counter = 0
#for i in labels_unseen_list:
#    labels_unseen_arr[counter] = i.detach().numpy()
#    counter += 1
#print(np.unique(labels_unseen_arr))


# In[82]:


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


# In[83]:


from sklearn.neighbors import KNeighborsClassifier

#n_neighbours = [1, 3, 5, 10, 20, 30, 40, 50]
n_neighbours = [20, 200]
for k in n_neighbours:
    
    #knn = KNeighborsClassifier(n_neighbors=k)
    knn = FaissKNeighbors(k = k)
    knn.fit(embeddings_seen_arr, labels_seen_arr)
    labels_predicted = knn.predict(embeddings_unseen_arr)

    print('Accuracy' + str(k) + ':', np.sum(labels_unseen_arr == labels_predicted)/len(labels_unseen_arr))
    #print('Accuracy:', knn.score(embeddings_unseen_arr, labels_unseen_arr))


# In[ ]:


#np.savetxt('embeddings_imagenet_seen_vicreg.txt', embeddings_seen_arr)
#np.savetxt('labels_imagenet_seen_vicreg.txt', labels_seen_arr, fmt='%d')


# In[23]:


#JUST TRY DELETE LATER

#from sklearn.decomposition import PCA
#from sklearn.manifold import TSNE

#embeddings_unseen_arr_trial = embeddings_unseen_arr[0:180,:]
#labels_unseen_arr_trial = labels_unseen_arr[0:180]

#pca = PCA(n_components=2)
#pca.fit(embeddings_unseen_arr)
#print(pca.explained_variance_ratio_)

#embeddings_reduced = pca.transform(embeddings_unseen_arr)
#print(embeddings_reduced)

#u_labels = np.unique(labels_unseen_arr_trial)
#print(u_labels)

#embeddings_reduced = TSNE(n_components=2, learning_rate='auto',init='random', perplexity = 10.0).fit_transform(embeddings_unseen_arr_trial)

#for i in u_labels:
    
#    plt.scatter(embeddings_reduced[labels_unseen_arr_trial == i , 0] , embeddings_reduced[labels_unseen_arr_trial == i , 1] , label = i)

#plt.legend()
#plt.show()


# In[ ]:




