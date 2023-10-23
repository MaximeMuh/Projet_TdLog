#     # -*- coding: utf-8 -*-
#     """
#     Created on Sun Oct 22 19:03:28 2023

#     @author: maxim
#     """



#     import torch 
#     from torchvision import datasets, transforms
#     from PIL import Image
#     import torch.nn as nn

#     # Définir le chemin de votre dossier local contenant les images
#     folder_path = "C:/Users/maxim/Desktop/tdlog/CAT3"


#     # Utiliser torchvision.datasets.ImageFolder pour charger les images
#     # et appliquer des transformations
#     transform = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.ToTensor()
#     ])

#     train_dataset = datasets.ImageFolder(root=folder_path, transform=transform)

#     # Définir la taille du lot (batch size)
#     batch_size = 8

#     # Créer un DataLoader pour l'ensemble de données d'entraînement
#     train_loader = torch.utils.data.DataLoader(
#         train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
#     )




#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#     class VAE(nn.Module):

#         def __init__(self, input_dim=256*256, hidden_dim=400, latent_dim=200, device=device):
#             super(VAE, self).__init__()

#             # encoder
#             self.encoder = nn.Sequential(
#                 nn.Linear(input_dim, hidden_dim),
#                 nn.LeakyReLU(0.2),
#                 nn.Linear(hidden_dim, latent_dim),
#                 nn.LeakyReLU(0.2)
#                 )
            
#             # latent mean and variance 
#             self.mean_layer = nn.Linear(latent_dim, 2)
#             self.logvar_layer = nn.Linear(latent_dim, 2)
            
#             # decoder
#             self.decoder = nn.Sequential(
#                 nn.Linear(2, latent_dim),
#                 nn.LeakyReLU(0.2),
#                 nn.Linear(latent_dim, hidden_dim),
#                 nn.LeakyReLU(0.2),
#                 nn.Linear(hidden_dim, input_dim),
#                 nn.Sigmoid()
#                 )
        
#         def encode(self, x):
#             x = self.encoder(x)
#             mean, logvar = self.mean_layer(x), self.logvar_layer(x)
#             return mean, logvar

#         def reparameterization(self, mean, var):
#             epsilon = torch.randn_like(var).to(device)      
#             z = mean + var*epsilon
#             return z

#         def decode(self, x):
#             return self.decoder(x)
            
#         def forward(self, x):
#             mean, log_var = self.encode(x)
#             z = self.reparameterization(mean, torch.exp(0.5 * log_var)) 
#             x_hat = self.decode(z)  
#             return x_hat, mean, log_var

#     from torch.optim import Adam 
#     model = VAE().to(device)
#     optimizer = Adam(model.parameters(), lr=1e-3)


#     def loss_function(x, x_hat, mean, log_var):
#         reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
#         KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

#         return reproduction_loss + KLD



#     def train(model, optimizer, epochs, device):
#         model.train()
#         for epoch in range(epochs):
#             overall_loss = 0
#             for batch_idx, (x, _) in enumerate(train_loader):
#                 x = x.view(-1, 256*256).to(device)

#                 optimizer.zero_grad()

#                 x_hat, mean, log_var = model(x)
#                 loss = loss_function(x, x_hat, mean, log_var)
                
#                 overall_loss += loss.item()
                
#                 loss.backward()
#                 optimizer.step()

#             print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss/(batch_idx*batch_size))
#         return overall_loss

#     #train(model, optimizer, epochs=50, device=device)   

#     # import matplotlib.pyplot as plt
#     # ###
#     # z_samples = torch.randn(8, 2).to(device)

#     # # Passez les échantillons par le décodeur pour générer des images
#     # generated_images = model.decode(z_samples)

#     import matplotlib.pyplot as plt

#     # ...

#     # # Créez une figure pour afficher les images générées
#     # fig, axes = plt.subplots(1, len(generated_images), figsize=(10, 5))

#     # for i in range(len(generated_images)):
#     #     print(generated_images[i][0].cpu().detach().numpy().shape)

#     #     print("Generated image shape:", generated_images[i][0].cpu().detach().numpy().shape)

#     #     #axes[i].imshow(generated_images[i][0].cpu().detach().numpy(), cmap='jet')

#     #     axes[i].axis('off')

#     # plt.show()
#     # import matplotlib.pyplot as plt
#     # import matplotlib.pyplot as plt

#     # # Chargez un lot d'images
#     # x, _ = next(iter(train_loader))
#     # x = x.to(device)
#     # x = x.view(-1, 256*256).to(device)

#     # # Passez les images par le modèle VAE pour les reconstruire
#     # x_hat, _, _ = model(x)

#     # # Redimensionnez les images reconstruites à 256x256
#     # x_hat_reshaped = x_hat.view(-1, 1, 256, 256)

#     # # Affichez les images originales et les images reconstruites
#     # fig, axes = plt.subplots(2, 8, figsize=(16, 4))
#     # for i in range(8):
#     #     original_image = x[i].cpu().detach().numpy()
#     #     reconstructed_image = x_hat_reshaped[i][0].cpu().detach().numpy()
        
#     #     axes[0, i].imshow(original_image.reshape(256, 256))
#     #     axes[0, i].axis('off')
#     #     axes[1, i].imshow(reconstructed_image)
#     #     axes[1, i].axis('off')

#     # plt.show()

# # Chargez les 5 premières images de votre base de données
# first_five_images = []
# for i in range(5):
#     image, _ = train_dataset[i]  # Chargez l'image
#     first_five_images.append(image)  # Ajoutez l'image à la liste

# # Affichez les 5 premières images
# fig, axes = plt.subplots(1, 5, figsize=(20, 4))
# for i in range(5):
#     axes[i].imshow(first_five_images[i][0], cmap='jet')  # Affichez l'image en utilisant la colormap 'jet'
#     axes[i].axis('off')

# plt.show()

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 19:03:28 2023
@author: maxim
"""

import torch
from torchvision import datasets, transforms
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt

# Définir le chemin de votre dossier local contenant les images
folder_path = "C:/Users/maxim/Desktop/tdlog/CAT3"

# Utiliser torchvision.datasets.ImageFolder pour charger les images
# et appliquer des transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(root=folder_path, transform=transform)

# Définir la taille du lot (batch size)
batch_size = 512

# Créer un DataLoader pour l'ensemble de données d'entraînement
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VAE(nn.Module):   
    def __init__(self, input_dim=256*256, hidden_dim=400, latent_dim=200, device=device):
        super(VAE, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
        )
        
        # latent mean and variance 
        self.mean_layer = nn.Linear(latent_dim, 2)
        self.logvar_layer = nn.Linear(latent_dim, 2)
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)
        z = mean + var * epsilon
        return z

    def decode(self, x):
        return self.decoder(x)
        
    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) 
        x_hat = self.decode(z)
        return x_hat, mean, log_var

from torch.optim import Adam
model = VAE().to(device)
optimizer = Adam(model.parameters(), lr=1e-3)

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD

def train(model, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, 256 * 256).to(device)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()

        print("Epoch", epoch + 1, "Average Loss:", overall_loss / (batch_idx * batch_size))
    return overall_loss

train(model, optimizer, epochs=3, device=device)  

# Chargez les 5 premières images de votre base de données
first_five_images = []
for i in range(5):
    image, _ = train_dataset[i]  # Chargez l'image
    first_five_images.append(image)  # Ajoutez l'image à la liste

# Affichez les 5 premières images
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for i in range(5):
    axes[i].imshow(first_five_images[i][0], cmap='gray')  # Affichez l'image en utilisant la colormap 'jet'

    axes[i].axis('off')

plt.show()


# Chargez un lot d'images
x, _ = next(iter(train_loader))
x = x.to(device)
x = x.view(-1, 256*256).to(device)

    # Passez les images par le modèle VAE pour les reconstruire
x_hat, _, _ = model(x)

    # Redimensionnez les images reconstruites à 256x256
x_hat_reshaped = x_hat.view(-1, 1, 256, 256)

    # Affichez les images originales et les images reconstruites
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i in range(8):
    original_image = x[i].cpu().detach().numpy()
    reconstructed_image = x_hat_reshaped[i][0].cpu().detach().numpy()
        
    axes[0, i].imshow(original_image.reshape(256, 256))
    axes[0, i].axis('off')
    axes[1, i].imshow(reconstructed_image)
    axes[1, i].axis('off')
plt.show()                                