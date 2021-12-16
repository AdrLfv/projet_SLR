
from torchvision.transforms import ToTensor, Lambda
from typing import Tuple
#from tensorflow.keras.utils import to_categorical
import torch
import torch.nn as nn

import cv2
import numpy as np
import os

def calcul_transformation(data_frame, x_shift, y_shift, scale):
    
    #data_frame = list(data_frame)

    pose_landmarks = [[data_frame[ind] * scale + x_shift , data_frame[ind+1]* scale+ y_shift, data_frame[ind+2]* scale, data_frame[ind+3]]
                    for ind, _ in enumerate(data_frame[0:33*4]) if ind % 4 == 0]

    face_landmarks = [[data_frame[33*4+ind] * scale + x_shift, data_frame[33*4+ind+1]* scale+ y_shift, data_frame[33*4+ind+2]* scale]
                    for ind, _ in enumerate(data_frame[33*4:33*4+468*3]) if ind % 3 == 0]       

    left_hands_landmarks = [[data_frame[33*4+468*3+ind] * scale + x_shift, data_frame[33*4+468*3+ind+1]* scale+ y_shift, data_frame[33*4+468*3+ind+2]* scale]
                            for ind, _ in enumerate(data_frame[33*4+468*3: 33*4+468*3+21*3]) if ind % 3 == 0]

    right_hands_landmarks = [[data_frame[33*4+468*3+21*3+ind] * scale + x_shift, data_frame[33*4+468*3+21*3+ind+1]* scale+ y_shift, data_frame[33*4+468*3+21*3+ind+2]* scale] 
                            for ind, _ in enumerate(data_frame[33*4+468*3+21*3:]) if ind % 3 == 0]
    
    return pose_landmarks + face_landmarks + left_hands_landmarks + right_hands_landmarks

def transformation(self, sequences):

    new_sequences = []
    # on récupère chaque frame de chaque sequence de chaque action, on effectue les modifications sur celle-ci,
    # on sauvegarde les données de cette nouvelle frame dans une nouvelle séquence dans la même action,
    # on effectue sur la frame suivante les mêmes modifications et on sauvegarde dans la nouvelle séquence
    # lorsque l'on passe à la séquence suivante on change les paramètres de modification

    
    for action in self.actionsToAugment:
        #on ne va pas copier toutes les séquences, on prend juste la premiere de chaque action, sinon on a beaucoup trop de donnees
        #pour rajouter la prise en compte de toutes les sequences, decommenter cette ligne :
        #for sequence_num in range(self.nb_sequences):
        sequence_num = 0
        new_sequence_num = self.nb_sequences
        for scale in np.arange(0.5, 1.5, 0.5):
            for x_shift in np.arange(-0.3, 0.3, 0.1):
                for y_shift in np.arange(-0.3, 0.3, 0.1):
                    
                    try:
                        os.makedirs(os.path.join(
                            self.DATA_PATH, action, str(new_sequence_num)))
                        print("sucess")
                    except:
                        pass
                    for frame_num in range(self.sequence_length):
                        frame_datas = sequences[sequence_num][frame_num]
                        # on prends les coordonnées actuelles, on effectue un shift dans une fenetre de 0.3 sur les x
                        # de -0.3 à 0.3 avec un décalage de 0.1
                        new_frame_datas = calcul_transformation(frame_datas, x_shift, y_shift, scale)
                        # vérifier que si on trouve un 0.0 on ne modifie pas
                        npy_path = os.path.join(
                            self.DATA_PATH, action, str(new_sequence_num), str(frame_num))
                        
                        np.save(npy_path, new_frame_datas)
                        print(npy_path)
                        #print(new_sequence_num)
                        #print(new_frame_datas)
                    new_sequence_num += 1
                    #print("new sequence num ", new_sequence_num)
                        


class Data_augmentation():

    def __init__(self, DATA_PATH, actionsToAugment, sequence_length,nb_sequences):
        self.DATA_PATH = DATA_PATH
        self.actionsToAugment = actionsToAugment
        self.sequence_length = sequence_length
        self.nb_sequences = nb_sequences
        # on récupère les données stockées dans le dataset
        sequences = []
        for action in self.actionsToAugment:
            #for sequence in np.array(os.listdir(os.path.join(self.DATA_PATH, action))).astype(int):
            for sequence in range(nb_sequences):
                window = []
                for frame_num in range(self.sequence_length):
                    print(action, str(
                        sequence), "{}.npy".format(frame_num))
                    res = np.load(os.path.join(self.DATA_PATH, action, str(
                        sequence), "{}.npy".format(frame_num)))
                    
                    # window.extend(res)
                    window.append(res)
                sequences.append(window)
                # datas.append(sequences)

        transformation(self, sequences)
