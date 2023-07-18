from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class normalizeImage:
  def __call__(self,sample):
    sample=sample.numpy()
    sample=sample.astype("float32")
    sample=sample/255.0
    sample=sample*2-1
    sample=torch.from_numpy(sample)
    return sample



class ImageDataset(Dataset):
    def __init__(self,image_paths,transform=None):
      self.image_paths=image_paths
      self.transform=transform

    def __getitem__(self, index):
      # image=cv2.imread(self.image_paths[index])
      image=Image.open(self.image_paths[index])
      if self.transform is not None:
        image=self.transform(image)
      return image

    def __len__(self):
      return len(self.image_paths)

def get_my_transforms(image_size):
  my_transforms=transforms.Compose([
          transforms.Resize((image_size,image_size)),
          transforms.PILToTensor(),
          normalizeImage()

      ])
  return my_transforms

def get_train_dataset(path,image_size):
    entries = os.listdir(path)
    image_paths=[path+image_name for image_name in entries]
    my_transforms=transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.PILToTensor(),
        normalizeImage()

    ])
    dataset = ImageDataset(image_paths,my_transforms)
    return dataset


def get_dataLoader(dataset,batch_size,shuffle):
   data_loader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle)
   return data_loader


class TestImageDataset(Dataset):
    def __init__(self,image_paths,labels,transform=None):
      self.image_paths=image_paths
      self.transform=transform
      self.labels=labels

    def __getitem__(self, index):
      # image=cv2.imread(self.image_paths[index])
      image=Image.open(self.image_paths[index])
      if self.transform is not None:
        image=self.transform(image)
      label=self.labels[index]
      label=torch.tensor(label)
      return image,label

    def __len__(self):
      return len(self.image_paths)

def get_test_dataset(image_paths,labels,image_size,transform=True):
   if transform:  
    my_transforms=transforms.Compose([
          transforms.Resize((image_size,image_size)),
          transforms.PILToTensor(),
          normalizeImage()

      ])
   else:
      my_transforms=transforms.Compose([
          transforms.Resize((image_size,image_size)),
          transforms.PILToTensor(),
      ])

    
   datasets=TestImageDataset(image_paths=image_paths,labels=labels,transform=my_transforms)
   return datasets