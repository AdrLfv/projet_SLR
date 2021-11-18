#!pip install tensorflow==2.4.1 tensorflow-gpu==2.4.1 opencv-python mediapipe sklearn matplotlib

import numpy as np
import os
import time
import mediapipe as mp
import torch

from projet_SLR.preprocess import Preprocess
from projet_SLR.LSTM import LSTM
from projet_SLR.test import launch_test
from projet_SLR.load_LSTM import load_LSTM
from projet_SLR.dataset import CustomImageDataset
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules

def launch_NN(input_size):
    # Set device cuda for GPU if it's available otherwise run on the CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Hyperparameters of our neural network which depends on the dataset, and
    # also just experimenting to see what works well (learning rate for example).
    
    output_size = 3
    learning_rate = 0.001 #how much to update models parameters at each batch/epoch
    batch_size = 32 #number of data samples propagated through the network before the parameters are updated
    NUM_WORKERS = 4
    num_epochs = 200 #number times to iterate over the dataset
    DECAY = 1e-4
    hidden_size = 128 #number of features in the hidden state h
    num_layers = 2
    
    #print("len X_train:",len(X_train)*len(X_train[0])*len(X_train[0][0]))
    #l'input_size doit être 30*30*1662
    
    
    train_loader = DataLoader(datas_train, batch_size=batch_size, shuffle=True,num_workers=NUM_WORKERS,
    pin_memory=True)
    
    test_loader = DataLoader(datas_test, batch_size=batch_size, shuffle=True,num_workers=NUM_WORKERS,
    pin_memory=True)
    # Initialize network
    model = LSTM(input_size,  hidden_size, num_layers,output_size).to(device)
    try:
        model.load_state_dict(torch.load("actionNN.pth"))
    except:
        print("No model saved")
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=DECAY)
        # Train Network
        
        for epoch in range(num_epochs): 
            print(f"Epoch {epoch+1}\n-------------------------------")
            model.train_loop(train_loader, model, criterion, optimizer)
            
            print(f"Accuracy on training set: {model.test_loop(train_loader, model)*100:.2f}")
            print(f"Accuracy on test set: {model.test_loop(test_loader, model)*100:.2f}")
        print("Done!")
        # model = models.vgg16(pretrained=True).cuda()
        torch.save(model.state_dict(), 'actionNN.pth')
    
    return model #,logits

# on crée des dossiers dans lequels stocker les positions des points que l'on va enregistrer
# Chemin pour les données
DATA_PATH = os.path.join('MP_Data')

# Actions that we try to detect
actions = np.array(['hello', 'thanks', 'iloveyou'])

# Thirty videos worth of data
nb_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

# dataset making
#CustomImageDataset(actions, nb_sequences, sequence_length, DATA_PATH)

# reprocess
datas_train = Preprocess("train", actions,DATA_PATH, sequence_length)
datas_test = Preprocess("test", datas_train, None,None)
#print("datas_train:{}")(datas_train)
# model, res = launch_NN(datas_train, datas_test)

input_size = 1662#len(datas_train["X_train"][0])
print(input_size)

model = launch_NN(input_size)

launch_test(actions, model)
# model(np.expand_dims(datas_test[0][0], axis=0))

# Créer une classe dataset qui donne pour un indice de séquence donné, une liste de datas de 30*1662
# faire dans cette classe un __len__ qui donne le nombre total de séquences (return 90)