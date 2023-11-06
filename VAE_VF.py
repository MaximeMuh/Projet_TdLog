import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
#%% Importation des données
    
folder_path = "C:/Users/maxim/Desktop/IMI/TDLOG/Projet_TdLog/CAT3"

input_dim=128 #taille des données input_dim*input_dim
batch_size=64 #nombres d'images dans le lot

transform = transforms.Compose([
    transforms.Resize((input_dim, input_dim)),
    transforms.ToTensor()
])

train_set = datasets.ImageFolder(root=folder_path, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #pour travailler sur gpu

#%% Définition de notre modèle encodeur-décodeur
kernel_size=4
stride=2
latent_dim=1024

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Conv2d(3,32,kernel_size,stride)
        self.fc21 = nn.Linear(int(((input_dim-kernel_size)/stride+1)**2),latent_dim) # moyenne
        self.fc22 = nn.Linear(int(((input_dim-kernel_size)/stride+1)**2),latent_dim) # logvariance
        self.fc3 = nn.Linear(latent_dim,int(((input_dim-kernel_size)/stride+1)**2))
        self.fc4 = nn.ConvTranspose2d(32, 3, kernel_size, stride)

    def encode(self, x):
        h1 = F.sigmoid(self.fc1(x))
        h1 = h1.view(h1.size(0),h1.size(1),1,-1)
        return self.fc21(h1), self.fc22(h1) #retourne moyenne et logvariance

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar) 
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu) 
    
    def decode(self, z):
        return torch.sigmoid(self.fc4(z)) # retourne l'image décodé 

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        z = F.relu(self.fc3(z))
        z = z.view(z.size(0),z.size(1),int((input_dim-kernel_size)/stride+1),int((input_dim-kernel_size)/stride+1))
        return self.decode(z), mu, logvar
    


def loss_function(x,recon_x , mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train(model, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        overall_loss = 0
        u=0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            x_hat, mean, log_var = model(x)
            
            loss = loss_function(x, x_hat, mean, log_var)
            overall_loss += loss.item()
            if u%5000==0:
                print(batch_idx/len(train_loader), overall_loss)
            loss.backward()
            optimizer.step()

        print("Epoch", epoch + 1, "Average Loss:", overall_loss / (batch_idx * batch_size))
    return overall_loss
#%%Création d'une instance de notre classe VAE et définition de l'optimiseur
model = VAE()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

#%%Entrainement de notre modèle
train(model, optimizer, epochs=10, device=device)  
#%%Affichage de l'image initiale et de l'image en sortie du modèle
image,_ = train_set.__getitem__(random.randint(0,100))
with torch.no_grad():
    image = image.to(device)
    print(image.size())
    recon_image, mu, logvar = model(image.unsqueeze(0))
    print(recon_image.size())
image =image.cpu().numpy()
image = np.transpose(image, (1, 2, 0))

recon_image = recon_image.cpu().numpy()[0]
recon_image = np.transpose(recon_image, (1, 2, 0))

# display the original image and the reconstructed image side-by-side
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
axes[0].imshow(image)
axes[0].set_title("Original")
axes[1].imshow(recon_image)
axes[1].set_title("Reconstructed")
plt.show()

