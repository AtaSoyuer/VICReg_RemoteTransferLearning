import pandas as pd
import numpy as np
import warnings
import random
from glob import glob

from torch.utils.data.dataset import Dataset
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

class CustomDatasetFromImages(Dataset):
    def __init__(self, csv_path, transform):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.transforms = transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # First column contains the image paths
        #self.image_arr = np.asarray(self.data_info.iloc[:, 1])
        self.image_arr = np.asarray(pd.read_csv(csv_path, header=0).iloc[:, 1])
        # Second column is the labels
        #self.label_arr = np.asarray(self.data_info.iloc[:, 0])
        self.label_arr = np.asarray(pd.read_csv(csv_path, header=0).iloc[:, 0])
        # Calculate len
        #self.data_len = len(self.data_info.index)
        self.data_len = len(pd.read_csv(csv_path, header=0).index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        #single_image_name = self.image_arr[index]
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name)
        # Transform the image
        img_as_tensor = self.transforms(img_as_img)
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len

class CustomDatasetFromImagesTemporalClassification(Dataset):
    def __init__(self, csv_path, transform):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.transforms = transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 1])#[:16]
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 0])#[:16]
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name)
        # Transform the image
        img_as_tensor = self.transforms(img_as_img)
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return (img_as_tensor, single_image_label, single_image_name.split('/')[-2])

    def __len__(self):
        return self.data_len


class CustomDatasetFromImagesTemporal(Dataset):
    def __init__(self, csv_path, transform):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.transforms = transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 1])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 0])
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name_1 = self.image_arr[index]
        

        splt = single_image_name_1.rsplit('/', 1)
        #print(splt)
        base_path = splt[0]
        #print('Base.',base_path)
        fname = splt[1]
        #print('FNAEM:',fname)
        
        #suffix = fname[-15:]
        #print('SUFFİX:',suffix)
        #prefix = fname[:-15].rsplit('_', 1)
        #print('PREFIX:',prefix)
        
        splt_char = "_"
        K = -2
        temp = fname.split(splt_char)
        prefix = splt_char.join(temp[:K]), splt_char.join(temp[K:])
        suffix = '_rgb.jpg'
        
        
        regexp = '{}/{}_*{}'.format(base_path, prefix[0], suffix)
        #print('REGEXP:',regexp)
        temporal_files = glob(regexp)
        #print('Temporal.',temporal_files)
        temporal_files.remove(single_image_name_1)
        #print(temporal_files)
        if temporal_files == []:
            single_image_name_2 = single_image_name_1
        else:
            single_image_name_2 = random.choice(temporal_files)
        
        #print(single_image_name_1)
        #print(single_image_name_2)
        img_as_img_1 = Image.open(single_image_name_1)
        #img_as_tensor_1 = self.transforms(img_as_img_1)

        img_as_img_2 = Image.open(single_image_name_2)
        #img_as_tensor_2 = self.transforms(img_as_img_2)
        
        img_as_tensor_1, img_as_tensor_2 = self.transforms(img_as_img_1, img_as_img_2)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return img_as_tensor_1, img_as_tensor_2

    def __len__(self):
        return self.data_len
