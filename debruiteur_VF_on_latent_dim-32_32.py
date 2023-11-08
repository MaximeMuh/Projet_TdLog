# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 16:02:38 2023

@author: maxim
"""
#%% importation des modulesss 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,random_split
from torchvision import datasets, transforms
import torch.optim as optim
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

#%%Importation des données brouillées et non brouillées  
folder_path="C:/Users/maxim/Desktop/IMI/TDLOG/Projet_TdLog/image_32_32"
folder_path_brouille="C:/Users/maxim/Desktop/IMI/TDLOG/Projet_TdLog/image_32_32_brouille"

input_dim=32 #taille des données input_dim*input_dim
batch_size=77 #nombres d'images dans le lot

transform = transforms.Compose([
    transforms.Resize((input_dim, input_dim)),
    transforms.ToTensor()
])
full_dataset = datasets.ImageFolder(root=folder_path, transform=transform)
full_dataset2 = datasets.ImageFolder(root=folder_path_brouille, transform=transform)
 
test_size = len(full_dataset) // 10  


train_set = full_dataset[:-test_size] 
test_set = full_dataset[-test_size:]  
train_set2 = full_dataset2[:-test_size] 
test_set2 = full_dataset2[-test_size:]


train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
train_loader2 = DataLoader(train_set, batch_size=batch_size, shuffle=False)
test_loader2 = DataLoader(test_set, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #pour travailler sur gpu

#%% Création de notre modèle de débrouillage
#définitions des variables de nos réseaux de  neurones
kernel_size=4
stride=2
input_conv=3 #R G B 

class Debruiteur(nn.Module):
    def __init__(self):
        super(Debruiteur,self).__init__()
        
        self.NN=nn.Sequential(
            nn.Conv2d(input_conv,8,kernel_size,stride),
            nn.ReLU(),
            nn.Conv2d(8,64,kernel_size,stride),
            nn.Sigmoid(),
            nn.ConvTranspose2d(64,8,kernel_size+1,stride),
            nn.ReLU(),
            nn.ConvTranspose2d(8,input_conv,kernel_size,stride),
            nn.Sigmoid()
            )
    def forward(self,x):       
        return self.NN(x)
    
    
def loss_function(x,x_recon):
    BCE = F.binary_cross_entropy(x_recon, x, reduction='sum')
    return BCE
def train(model, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        overall_loss = 0
        u=0
        it=iter(train_loader2)
        for batch_idx, (x_init, _) in enumerate(train_loader):
            x_bruit= next(it)[0]
            
            x_init = x_init.to(device)
            x_bruit = x_bruit.to(device)
            optimizer.zero_grad()
            x_recons = model(x_bruit)
            
            loss = loss_function(x_init, x_recons)
            overall_loss += loss.item()
            if u%5000==0:
                print(batch_idx/len(train_loader), overall_loss)
            loss.backward()
            optimizer.step()

        print("Epoch", epoch + 1, "Average Loss:", overall_loss / (batch_idx * batch_size))
    return overall_loss

#%%Création d'une instance de notre classe 
model = Debruiteur()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
#%%Entrainenement de notre modèle
train(model, optimizer, epochs=50, device=device)

#%%Sauvegarde de notre modèle
file_path ="C:/Users/maxim/Desktop/IMI/TDLOG/Projet_TdLog/modele_entraine_debruiteur_32_100e.pt"
torch.save(model, file_path)

#%%chargement d'un modèle préentrainé 
model = torch.load('C:/Users/maxim/Desktop/IMI/TDLOG/Projet_TdLog/modele_entraine_debruiteur_32_100e.pt', map_location=torch.device('cpu'))
#%% Affichage de l'image initiale et de l'image en sortie du modèle

a=random.randint(0,100)
image,_ = test_set.__getitem__(a)
image_bruit,_=test_set2.__getitem__(a)
with torch.no_grad():
    image = image.to(device)
    recon_image = model(image.unsqueeze(0))[0]
image = image.numpy()
image = np.transpose(image, (1, 2, 0))
image_bruit = image_bruit.numpy()
image_bruit = np.transpose(image_bruit, (1, 2, 0))
recon_image = recon_image.numpy()
recon_image = np.transpose(recon_image, (1, 2, 0))

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 4))
axes[0].imshow(image)
axes[0].set_title("Original")
axes[0].axis('off')
axes[1].imshow(recon_image)
axes[1].set_title("Reconstructed")
axes[1].axis('off')
axes[2].imshow(image_bruit)
axes[2].set_title("bruit")
axes[2].axis('off')
plt.show()