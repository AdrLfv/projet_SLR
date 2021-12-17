from projet_SLR.data_augmentation import Data_augmentation
from sklearn.model_selection import train_test_split
from torchvision.transforms import ToTensor, Lambda
from typing import Tuple
#from tensorflow.keras.utils import to_categorical
import random
import torch
import torch.nn as nn

import cv2
import numpy as np
import os

from projet_SLR.tuto import DATA_PATH


class Preprocess():
    def __init__(self, actions, DATA_PATH: str, nb_sequences: int, sequence_length: int, data_augmentation: bool):

        self.actions = actions
        self.DATA_PATH = DATA_PATH
        self.sequence_length = sequence_length
        self.nb_sequences = nb_sequences
        self.data_augmentation = data_augmentation

    def __getitem__(self, idx_seq: int) -> Tuple[torch.Tensor, int]:

        # le idx_seq est un index de la sequence actuelle, il est au nombre total du nombre d'actions * le nombre de sequences par action
        # l'indice de sequence que l'on cherche dans les fichiers va de 0 à 23 si le type de preprocess est train, de 0 à 3 s'il est un test,
        # de 0 à 3 s'il est un valid

        ind_action = int(idx_seq/self.nb_sequences)
        ind_seq = idx_seq % self.nb_sequences

        if(self.data_augmentation):
            # while(ind_seq>=self.nb_sequences): ind_seq =- self.nb_sequences
            # while(ind_action>=len(self.actions)): ind_action =- len(self.actions)
            x_shift = random.uniform(-0.3, 0.3)
            y_shift = random.uniform(-0.3, 0.3)
            scale = random.uniform(0.5, 1.5)

        # print("ind action : ", ind_action)
        # print("seq ind :", ind_seq)
        # print("path :", DATA_PATH)

        window = []
        for frame_num in range(self.sequence_length):
            #print("frame num : ",frame_num)
            res = np.load(os.path.join(self.DATA_PATH, self.actions[ind_action], str(
                ind_seq), "{}.npy".format(frame_num)), allow_pickle=True)
            if (self.data_augmentation): res = Data_augmentation(res, x_shift, y_shift, scale).__getitem__()
            window.append(res)

        self.X = window
        self.y = ind_action 

        flatten = nn.Flatten()
        
        # print(self.X)
        self.X = torch.tensor(self.X, dtype=torch.float)
        
        self.y = torch.tensor(self.y, dtype=torch.long)
        data = self.X, self.y
        return data

    def __len__(self):
        # Par exemple pour le train on a besoin de toutes les sequences de train concernant hello, thanks...
        # on return donc le nb_sequences concernant le train et le nombre d'actions totales
        return self.nb_sequences*len(self.actions)

    def get_data_length(self):
        return len(np.load(os.path.join(self.DATA_PATH, self.actions[0], str(0), "{}.npy".format(0)), allow_pickle=True))