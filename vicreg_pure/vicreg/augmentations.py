# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from PIL import ImageOps, ImageFilter
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import torch


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            #return img.filter(ImageFilter.GaussianBlur(sigma))
            transformed = transforms.functional.gaussian_blur(img, kernel_size=3, sigma=sigma)
            #print('after gaussian blur:', torch.sum(torch.isnan(transformed)) )
            return transformed
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            #return ImageOps.solarize(img)
            transformed = transforms.functional.solarize(img, threshold=0.5)
            #print('after solarization:', torch.sum(torch.isnan(transformed)) )
            return transformed
        else:
            return img


class TrainTransform(object):
    def __init__(self):
        self.transform = transforms.Compose(   #NOTE: THE PROBLEM IS THE COMBINATION IF RANDOMRESIZEDCROP AND COLORJITTER,
                                                #PERHAPS DUE TO INTERPOLATION, SO REPLACE WITH RANDOMCROP !!!!!!!!!!!!!!
            [
                transforms.RandomCrop(        #REPLACE WITH RANDOMCROP
                    224,), #interpolation=InterpolationMode.BICUBIC
                #),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.6, contrast=0.6, saturation=0.4, hue=0.2
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.3),
                GaussianBlur(p=1.0),
                Solarization(p=0.0),
                #transforms.ToTensor(),           #COMMENT OUT IF USING SN7 DATASET!!!!!!!
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.transform_prime = transforms.Compose(
            [
                transforms.RandomCrop(
                    224), #interpolation=InterpolationMode.BICUBIC
                #),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.6, contrast=0.6, saturation=0.4, hue=0.2
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=0.1),
                Solarization(p=0.3),
                #transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, sample):
        #print('sample shape', sample.shape)
        #print('shape', sample.shape)
        x1 = self.transform(sample)
        x2 = self.transform_prime(sample)
        #print('shape', x1.shape)
        #print('shape', x2.shape)
        #print('x1:', torch.sum(torch.isnan(x1)) )
        #print('x2:', torch.sum(torch.isnan(x2)) )
        return x1, x2
