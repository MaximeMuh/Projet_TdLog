import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,random_split
from torchvision import datasets, transforms
import torch.optim as optim
import torch
import numpy as np
import matplotlib.pyplot as plt


#from google.colab import drive #à importer si google colab 
#%%
input_dim=128 #taille des données input_dim*input_dim
batch_size=64 #nombres d'images dans le lot

#google colab
#drive.mount('/content/gdrive')
#folder_path = "/content/gdrive/My Drive/vacances/CAT3"

#sinon
folder_path = "C:/Users/titou/Fichiers Infos/Python/tdlog_proj/Projet_TdLog/CAT3"

transform = transforms.Compose([
    transforms.Resize((input_dim, input_dim)),
    transforms.ToTensor()
])
full_dataset = datasets.ImageFolder(root=folder_path, transform=transform)


test_size = int(0.1 * len(full_dataset))
train_size = len(full_dataset) - test_size


train_set, test_set = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%#%% Définition de notre modèle encodeur-décodeur
kernel_size=3
stride=2
latent_dim=1024*3
input_conv=3
dropout_prob=0.05

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Conv2d(input_conv,9,kernel_size,stride)
        self.dim1=int((input_dim-kernel_size)/stride+1)
        self.fc2 = nn.Conv2d(9,81,kernel_size,stride)
        self.dim2=int((self.dim1-kernel_size)/stride+1)
        self.fc3=nn.Conv2d(81,243,kernel_size,stride)
        self.dim3=int((self.dim2-kernel_size)/stride+1)
        self.fc21 = nn.Linear(243*self.dim3**2,latent_dim) # mean
        self.fc22 = nn.Linear(243*self.dim3**2,latent_dim) # variance
        self.fc4 = nn.Linear(latent_dim,243*self.dim3**2)
        self.fc5 = nn.ConvTranspose2d(243,81,kernel_size,stride)
        self.fc6 = nn.ConvTranspose2d(81,9,kernel_size,stride)
        self.fc7 = nn.ConvTranspose2d(9, 3, kernel_size+1, stride)
#dropout, rajouter couche conv
#wait and biais 
    def encode(self, x):
        h0=F.relu(self.fc1(x))
        h1=F.relu(self.fc2(h0))
        h2 = F.sigmoid(self.fc3(h1))
        h2 = h2.view(h2.size(0),1,1,-1)
        return self.fc21(h2), self.fc22(h2) 

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu) 

    def decode(self, z):
        h4 = F.relu(self.fc5(z))
        h5 = F.relu(self.fc6(h4))
        return F.sigmoid(self.fc7(h5))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        z = F.relu(self.fc4(z))
        z = z.view(z.size(0),243,self.dim3,self.dim3)
        return self.decode(z), mu, logvar

def loss_function(x,recon_x , mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train(model, optimizer, epochs, device):
    model.train()
    loss_train_per_epoch=[]
    loss_test_per_epoch=[]
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
        average_loss=overall_loss / (batch_idx * batch_size)
        loss_train_per_epoch.append(average_loss)
        loss_test_per_epoch.append(test_loss(test_loader,model))
        print("Epoch", epoch + 1, "Average Loss:", overall_loss / (batch_idx * batch_size))
    X=[k for k in range(epochs)]
    plt.plot(X,loss_train_per_epoch)
    plt.plot(X,loss_test_per_epoch)
    plt.show()
    return overall_loss

def test_loss(test,mod):
    overall_loss = 0
    for batch_idx, (x, _) in enumerate(test):
        x = x.to(device)
        optimizer.zero_grad()
        x_hat, mean, log_var = mod(x)

        loss = loss_function(x, x_hat, mean, log_var)
        overall_loss += loss.item()
    return overall_loss/(batch_idx * batch_size)

#%%création d'une instance de notre classe de réseaux de neurones
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
#%%chargement d'un modèle pré entrainé
#model = torch.load('/content/gdrive/My Drive/vacances/vae_24ep.pt', map_location=torch.device('cpu'))#google colab
model = torch.load('C:/Users/maxim/Desktop/IMI/TDLOG/Projet_TdLog/modele_vae_2layerconv_50ep.pt', map_location=torch.device('cpu'))

    
#%% entrainement du modèle
train(model, optimizer, epochs=2,device=device)
#%%Sauvegarde du modèle 
#file_path = "/content/gdrive/My Drive/vacances/vae_2layerconv_50ep.pt" #google colab
file_path = "C:/Users/titou/Fichiers Infos/Python/tdlog_proj/Projet_TdLog/VAE_test0.pt"
torch.save(model, file_path)
#%% Affichage de l'image initiale et de l'image en sortie du modèle

import random
image,_ = test_set.__getitem__(random.randint(0,100))
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
#%%

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    