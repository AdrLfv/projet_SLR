from sklearn.model_selection import train_test_split
from torchvision.transforms import ToTensor, Lambda
from typing import Tuple
#from tensorflow.keras.utils import to_categorical
import torch
import torch.nn as nn

import cv2
import numpy as np
import os


class Preprocess():
    def __init__(self, actions, DATA_PATH: str, nb_sequences: int, sequence_length: int):

        self.actions = actions
        self.DATA_PATH = DATA_PATH
        self.sequence_length = sequence_length
        self.nb_sequences = nb_sequences

    def __getitem__(self, idx_seq: int) -> Tuple[torch.Tensor, int]:

        coeff = self.nb_sequences

        #print("idx seq : ", idx_seq)
        hot_sequences, sequences = [], []
        ind_action = int(idx_seq/coeff)
        #print("ind action", ind_action)
        ind_seq = idx_seq%coeff
        # if(self.process_type == "test"):

        # elif(self.process_type == "valid"):

        window = [] 
        # print("idx_seq : ", idx_seq)       
        # print("action : ", ind_action)
        # print("ind seq : ", ind_seq)
        for frame_num in range(self.sequence_length):

            res = np.load(os.path.join(self.DATA_PATH, self.actions[ind_action], str(
                ind_seq), "{}.npy".format(frame_num)), allow_pickle=True)    
            window.append(res)

        self.X = window
        self.y = ind_action 

        flatten = nn.Flatten()
        self.X = torch.tensor(self.X, dtype=torch.float)
        self.y = torch.tensor(self.y, dtype=torch.long)
        data = self.X, self.y
        return data

    def __len__(self):

        return self.nb_sequences*len(self.actions)

    def get_data_length(self):
        return len(np.load(os.path.join(self.DATA_PATH, self.actions[0], str(0), "{}.npy".format(0)), allow_pickle=True))
