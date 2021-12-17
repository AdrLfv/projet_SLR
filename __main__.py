#!pip install tensorflow==2.4.1 tensorflow-gpu==2.4.1 opencv-python mediapipe sklearn matplotlib

import numpy as np
import os
import time
import mediapipe as mp
import torch


from projet_SLR.dataset import CustomImageDataset
from projet_SLR.load_LSTM import load_LSTM
from projet_SLR.LSTM import myLSTM
from projet_SLR.preprocess import Preprocess
from projet_SLR.test import launch_test
from projet_SLR.tuto import Tuto
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
# Gives easier dataset managment by creating mini batches etc.
from torch.utils.data import DataLoader
from torch import nn  # All neural network modules
from torch import optim  # For optimizers like SGD, Adam, etc.


def launch_LSTM(output_size, train):
    # Set device cuda for GPU if it's available otherwise run on the CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Hyperparameters of our neural network which depends on the dataset, and
    # also just experimenting to see what works well (learning rate for example).

    learning_rate = 0.001  # how much to update models parameters at each batch/epoch
    batch_size = 32  # number of data samples propagated through the network before the parameters are updated
    NUM_WORKERS = 4
    num_epochs = 50  # number times to iterate over the dataset
    DECAY = 1e-4
    hidden_size = 128  # number of features in the hidden state h
    num_layers = 2
    
    train_preprocess = Preprocess(actions, DATA_PATH_TRAIN, nb_sequences_train, sequence_length, make_data_augmentation)
    valid_preprocess = Preprocess(actions, DATA_PATH_VALID, nb_sequences_valid, sequence_length, False)
    test_preprocess = Preprocess(actions, DATA_PATH_TEST, nb_sequences_test, sequence_length, False)

    input_size = train_preprocess.get_data_length()

    train_loader = DataLoader(train_preprocess, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=True)

    test_loader = DataLoader(test_preprocess, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS,
                                pin_memory=True)

    valid_loader = DataLoader(valid_preprocess, batch_size=batch_size, shuffle=True,num_workers=NUM_WORKERS, 
                                pin_memory=True)
    # Initialize network
    model = myLSTM(input_size,  hidden_size, num_layers, output_size).to(device)

    if(train):  # On verifie si on souhaite reentrainer le modele
        
        model = train_launch(model, learning_rate, DECAY,
                             num_epochs, train_loader, test_loader, valid_loader)
    else:
        try:
            print("Found valid model")
            model.load_state_dict(torch.load("actionNN.pth"))
        except:
            print("Not found")
            model = train_launch(model, learning_rate, DECAY,
                                 num_epochs, train_loader, test_loader, valid_loader)

    return model  # ,logits


def train_launch(model, learning_rate, DECAY, num_epochs, train_loader, test_loader, valid_loader):
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=DECAY)
    # Train Network

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        model.train_loop(train_loader, model, criterion, optimizer)

        print(
            f"Accuracy on training set: {model.test_loop(train_loader, model, criterion)*100:.2f}")
        print(
            f"Accuracy on test set: {model.test_loop(test_loader, model, criterion)*100:.2f}")
    print("Done!")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        
        print(
            f"Accuracy on valid set: {model.test_loop(valid_loader, model, criterion)*100:.2f}")

    # model = models.vgg16(pretrained=True).cuda()
    torch.save(model.state_dict(), 'actionNN.pth')
    return model


# on crée des dossiers dans lequels stocker les positions des points que l'on va enregistrer
# Chemin pour les données
DATA_PATH_TRAIN = os.path.join('MP_Data/Train')
DATA_PATH_VALID = os.path.join('MP_Data/Valid')
DATA_PATH_TEST = os.path.join('MP_Data/Test')
RESOLUTION_X = int(1920*9/10)  # Screen resolution in pixel
RESOLUTION_Y = int(1080*9/10)
# Thirty videos worth of data
nb_sequences = 30
nb_sequences_train = int(nb_sequences*80/100)
nb_sequences_valid = int(nb_sequences*10/100)
nb_sequences_test = int(nb_sequences*10/100)
# Videos are going to be 30 frames in length
sequence_length = 30

# =====================================Parametres à modifier =====================================================
make_train =  True
make_dataset = False
make_data_augmentation = True
#=================================================================================================================

if(make_dataset): make_train = True
# dataset making : (ajouter des actions dans le actionsToAdd pour créer leur dataset)
actionsToAdd = ["empty"]  #



# Actions that we try to detect
actions = np.array(["nothing","empty", "hello", "thanks", "iloveyou", "what's up",  "my", "name","nice","to meet you"])
#, "nothing" 'hello', 'thanks', 'iloveyou', "what's up", "hey", "my", "name", "nice","to meet you"

# instances de preprocess
# on crée des instances de preprocess en leur donnant le chemin d'accès ainsi que le nombre de séquences dans chaque dossier
# en fonction de si leur type de preprocess est train, valid, test.
#
if (make_dataset): CustomImageDataset(actionsToAdd, nb_sequences, sequence_length, DATA_PATH_TRAIN, DATA_PATH_VALID, DATA_PATH_TEST).__getitem__()

# Appel du modele
model = launch_LSTM(len(actions),make_train)

for action in actions:
    if (action != "nothing" and action != "empty"):
        Tuto(actions, action, RESOLUTION_X, RESOLUTION_Y).launch_tuto()
        launch_test(actions, model, action)
        
