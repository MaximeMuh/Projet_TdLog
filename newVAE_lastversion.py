import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import torch
import numpy as np
import matplotlib.pyplot as plt
#%%

folder_path = "C:/Users/maxim/Desktop/tdlog/Projet_Td/CAT3"

# Utiliser torchvision.datasets.ImageFolder pour charger les images
# et appliquer des transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
#%%
train_set = datasets.ImageFolder(root=folder_path, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
#%% Définir la taille du lot (batch size)
batch_size = 64
input_dim=28
kernel_size=4
stride=2
latent_dim=49
(images, labels) = iter(train_loader).next()
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Conv2d(3,32,kernel_size,stride)
#        self.fc2 = nn.Conv2d(32,64,4,2)
        self.fc21 = nn.Linear(int(((input_dim-kernel_size)/stride+1)**2),latent_dim) # mean
        self.fc22 = nn.Linear(int(((input_dim-kernel_size)/stride+1)**2),latent_dim) # variance
        self.fc3 = nn.Linear(latent_dim,int(((input_dim-kernel_size)/stride+1)**2))
        self.fc4 = nn.ConvTranspose2d(32, 3, kernel_size, stride)

    def encode(self, x):
        h1 = F.sigmoid(self.fc1(x))
        # h1 = h1.view()
        h1 = h1.view(h1.size(0),h1.size(1),1,-1)
        return self.fc21(h1), self.fc22(h1) # returns mean and variance

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu) # returns sampled latent variable z
    
    def decode(self, z):
#        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(z)) # returns reconstructed image

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

#%%

def train(model, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            x_hat, mean, log_var = model(x)
            
            loss = loss_function(x, x_hat, mean, log_var)
            overall_loss += loss.item()
            print(batch_idx/len(train_loader), overall_loss)
            loss.backward()
            optimizer.step()

        print("Epoch", epoch + 1, "Average Loss:", overall_loss / (batch_idx * batch_size))
    return overall_loss
#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VAE()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

#%%
train(model, optimizer, epochs=1, device=device)  
#%%
# convert the tensors to numpy arrays and reshape them into images
import random
image,_ = train_set.__getitem__(random.randint(0,100))
with torch.no_grad():
    image = image.to(device)
    recon_image, mu, logvar = model(image.unsqueeze(0))
image = np.reshape(image.numpy(), (128, 128))
recon_image = np.reshape(recon_image.numpy(), (128, 128))

# display the original image and the reconstructed image side-by-side
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
axes[0].imshow(image, cmap='gray')
axes[0].set_title("Original")
axes[1].imshow(recon_image, cmap='gray')
axes[1].set_title("Reconstructed")
plt.show()
#%% test visuel
k=18
i1 = images[k]
i1.numpy()
i1 = np.transpose(i1, (1, 2, 0))
plt.imshow(i1)
plt.show()


pred = model(images)[0]
p1 = pred[k]
p1 = p1.detach().numpy()
p1 = p1.transpose(1,2,0)
plt.imshow(p1)
plt.show()

