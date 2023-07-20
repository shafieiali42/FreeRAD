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

class ImageDataset(Dataset):
    def __init__(self, image_paths, size):
        self.image_paths = image_paths
        self.size = size
        # self.transform=transform

    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])
        # image=Image.open(self.image_paths[index])
        # if self.transform is not None:
        #   image=self.transform(image)
        image = cv2.resize(image, (self.size, self.size))
        image = image.reshape((3, self.size, self.size))
        image = image.astype("float32")
        image = image / 255.0
        image = image * 2 - 1
        image = torch.from_numpy(image)
        return image

    def __len__(self):
        return len(self.image_paths)


def get_train_dataset(path, image_size):
    entries = os.listdir(path)
    image_paths = [path + image_name for image_name in entries]
    if len(image_paths)%2==1:
        image_paths=image_paths[:-1]
    dataset = ImageDataset(image_paths, image_size)
    return dataset


def get_test_dataset(image_paths,labels, image_size):
    dataset = TestImageDataset(image_paths,labels,image_size)
    return dataset

def get_dataLoader(dataset, batch_size, shuffle):
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


class TestImageDataset(Dataset):
    def __init__(self, image_paths, labels,image_size, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.labels = labels
        self.image_size=image_size

    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])
        # image=Image.open(self.image_paths[index])
        # if self.transform is not None:
        #   image=self.transform(image)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image.reshape((3, self.image_size, self.image_size))
        image = image.astype("float32")
        image = image / 255.0
        image = image * 2 - 1
        image = torch.from_numpy(image)
        label = self.labels[index]
        label = torch.tensor(label)
        return image, label

    def __len__(self):
        return len(self.image_paths)


