# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 16:02:38 2023

@author: maxim
"""

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

#%%
folder_path="C:/Users/maxim/Desktop/tdlog/Projet_TdLog/CAT3"



# Utiliser torchvision.datasets.ImageFolder pour charger les images
# et appliquer des transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


batch_size=64


train_set = datasets.ImageFolder(root=folder_path, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

#%%
input_dim=128
kernel_size=4
stride=2
input_conv=3
alpha=5
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
    def bruitage(self,x):
        list_of_index=[[i,j] for i in range(input_dim) for j in range(input_dim)]
        nbr_pix=int((input_dim**2)*alpha/100)
        for k in range (batch_size):           
            for i in range(3):
                pix_to_change=random.sample(list_of_index,nbr_pix)           
                for j in range (nbr_pix):
                    x[k][i][pix_to_change[j][0]][pix_to_change[j][1]]=random.randint(0,255)/255
        return x
        
    def forward(self,x):
        x_bruit=self.bruitage(x)
        return self.NN(x_bruit)
def loss_function(x,x_bruit):
    BCE = F.binary_cross_entropy(x_bruit, x, reduction='sum')
    return BCE
def train(model, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        overall_loss = 0
        u=0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            
            optimizer.zero_grad()
            x_recons = model(x)
            
            loss = loss_function(x, x_recons)
            overall_loss += loss.item()
            if u%5000==0:
                print(batch_idx/len(train_loader), overall_loss)
            loss.backward()
            optimizer.step()

        print("Epoch", epoch + 1, "Average Loss:", overall_loss / (batch_idx * batch_size))
    return overall_loss

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Debruiteur()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
#%%

train(model, optimizer, epochs=1, device=device)
#%%
image,_ = train_set.__getitem__(5)    
image=image.numpy()
input_dim=128
alpha=5
list_of_index=[[i,j] for i in range(input_dim) for j in range(input_dim)]
nbr_pix=int((input_dim**2)*alpha/100)

for i in range(3):
    pix_to_change=random.sample(list_of_index,nbr_pix)
    
    for j in range (nbr_pix):
        image[i][pix_to_change[j][0]][pix_to_change[j][1]]=random.randint(0,255)/255




image= np.transpose(image, (1, 2, 0))
plt.imshow(image)
plt.show()



#%%%

# convert the tensors to numpy arrays and reshape them into images
import random
image,_ = train_set.__getitem__(random.randint(0,100))
with torch.no_grad():
    image = image.to(device)
    recon_image = model(image.unsqueeze(0))
image = np.reshape(image.numpy())
image = np.transpose(image, (1, 2, 0))
recon_image = np.reshape(recon_image.numpy())
recon_image = np.transpose(recon_image, (1, 2, 0))

# display the original image and the reconstructed image side-by-side
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
axes[0].imshow(image)
axes[0].set_title("Original")
axes[0].axis('off')
axes[1].imshow(recon_image)
axes[1].set_title("Reconstructed")
axes[1].axis('off')
plt.show()