import os
import sys
import pickle
import lmdb
import numpy as np
import torch

from pathlib import Path
from typing import Union, List
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision.datasets import LSUN
from torchvision.transforms import v2


class CIFAR10(Dataset):

    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]
    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]

    def __init__(
        self,
        root: Union[str, Path],
        image_size: int = 32,
        train: bool = True,
    ):
        super().__init__()

        self.data_path = os.path.join(root, 'cifar-10-batches-py')

        if train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list
        
        self.images = []
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.data_path, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.images.append(entry['data'])
        
        self.images = np.vstack(self.images).reshape(-1, 3, 32, 32)
        self.images = self.images.transpose((0, 2, 3, 1))

        self.transform = v2.Compose([
            v2.Resize((image_size, image_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])
        
    def __getitem__(self, index: int):
        image = Image.fromarray(self.images[index])
        return self.transform(image)

    def __len__(self):
        return len(self.images)


class CelebA(Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        image_size: int = 256
    ):
        super().__init__()

        self.data_path = os.path.join(root, 'CelebAMask-HQ', 'CelebA-HQ-img')
        self.images = os.listdir(self.data_path)

        self.transform = v2.Compose([
            v2.Resize((image_size, image_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])

    def __getitem__(self, index: int):
        image = Image.open(os.path.join(self.data_path, self.images[index]))
        return self.transform(image)
                              
    def __len__(self):
        return len(self.images)
        

class CustomLSUN(LSUN):
    def __init__(
        self,
        root: Union[str, Path],
        classes: Union[str, List[str]] = 'train',
        image_size: int = 256
    ):
        super().__init__(
            root=root,
            classes=classes,
            transform=None,
            target_transform=None
        )

        self.transform = v2.Compose([
            v2.Resize((image_size, image_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])

    def __getitem__(self, index: int):
        target = 0
        sub = 0
        for indice in self.indices:
            if index < indice:
                break
            target += 1
            sub = indice
        
        db = self.dbs[target]
        index = index - sub

        image, _ = db[index]

        return self.transform(image)