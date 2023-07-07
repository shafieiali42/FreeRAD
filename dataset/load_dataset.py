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


class normalizeImage:
  def __call__(self,sample):
    sample=torch.numpy()
    sample=sample.astype("float64")
    sample=sample/127.5
    sample=sample-1
    sample=torch.from_numpy(sample)
    return sample



class ImageDataset(Dataset):
    def __init__(self,image_paths,transform=None):
      self.image_paths=image_paths
      self.transform=transform

    def __getitem__(self, index):
      image=cv2.imread(self.image_paths[index])
      if self.transform is not None:
        image=self.transform(image)
      return image

    def __len__(self):
      return len(self.image_paths)



def get_train_dataset(path,image_size):
    entries = os.listdir(path)
    carpet_image_paths=[path+image_name for image_name in entries]
    my_carpet_transforms=transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        normalizeImage()

    ])
    dataset = ImageDataset(carpet_image_paths,my_carpet_transforms)
    return dataset

carpet_train_dataset=get_train_dataset("carpet/train/good/")
