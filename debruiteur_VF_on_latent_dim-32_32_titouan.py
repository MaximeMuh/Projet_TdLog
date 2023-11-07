import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
import torch.optim as optim
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

#%%Importation des données brouillées et non brouillées  
folder_path="C:/Users/titou/Fichiers Infos/Python/tdlog_proj/Projet_TdLog/image_32_32_30prct"

input_dim=32 #taille des données input_dim*input_dim
batch_size=64 #nombres d'images dans le lot

transform = transforms.Compose([
    transforms.Resize((input_dim, input_dim)),
    transforms.ToTensor()
])

train_set = datasets.ImageFolder(root=folder_path, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #pour travailler sur gpu


# Créez un Subset contenant uniquement les éléments avec label = 0
subset_indices = [i for i, (_, label) in enumerate(train_set) if label == 0]
subset_init = Subset(train_set, subset_indices)


train_size = int(0.8 * len(subset_init))
label_subset_init_train = [i for i in range(train_size)]
label_subset_init_test = [i for i in range(train_size,len(subset_init))]
  
subset_init_test =  Subset(subset_init, label_subset_init_test)
subset_init = Subset(subset_init, label_subset_init_train)

# Créez un DataLoader2 pour ce Subset --------- images init
batch_size2 = 64  # Vous pouvez définir la taille de lot souhaitée
loader_init = DataLoader(subset_init, batch_size=batch_size2, shuffle=False)
loader_init_test = DataLoader(subset_init_test, batch_size=64, shuffle=False)

# Créez un Subset contenant uniquement les éléments avec label = 1
subset_indices = [i for i, (_, label) in enumerate(train_set) if label == 1]
subset_bruit = Subset(train_set, subset_indices)

train_size = int(0.8 * len(subset_bruit))
label_subset_bruit_train = [i for i in range(train_size)]
label_subset_bruit_test = [i for i in range(train_size,len(subset_bruit))]

subset_init_test =  Subset(subset_bruit, label_subset_bruit_test)
subset_init = Subset(subset_bruit, label_subset_bruit_train)


# Créez un DataLoader3 pour ce Subset ------ Images bruitées
batch_size3 = 64  # Vous pouvez définir la taille de lot souhaitée
loader_bruit = DataLoader(subset_bruit, batch_size=batch_size3, shuffle=False)
loader_bruit_test = DataLoader(subset_bruit_test, batch_size=64, shuffle=False)








#%% Création de notre modèle de débrouillage
#définitions des variables de nos réseaux de  neurones
kernel_size=4
stride=2
drop_out = 0.1
input_conv=3 #R G B 

class Debruiteur(nn.Module):
    def __init__(self):
        super(Debruiteur,self).__init__()
        self.NN=nn.Sequential(
            nn.Conv2d(input_conv,9,kernel_size,stride),
            nn.ReLU(),
            nn.Conv2d(9,81,kernel_size,stride),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(81,243,kernel_size,stride),
            nn.Sigmoid(),
            nn.ConvTranspose2d(243,81,kernel_size,stride),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.ConvTranspose2d(81,9,kernel_size+1,stride),
            nn.ReLU(),
            nn.ConvTranspose2d(9,input_conv,kernel_size,stride),
            nn.Sigmoid()
            )
    def forward(self,x):       
        return self.NN(x)

class Debruiteur_lin(nn.Module):
    def __init__(self):
        super(Debruiteur_lin,self).__init__()
        self.f1 = nn.Linear(input_conv*input_dim**2,input_conv*input_dim**2)
        self.f2 = nn.Linear(input_conv*input_dim**2,input_conv*input_dim**2)
    def forward(self,x):
        x = x.view(x.size(0),1,1,-1)
        x = self.f1(x)
        x = F.relu(x)
        x = self.f2(x)
        x2 = F.sigmoid(x)
        x2 = x2.view(x2.size(0),input_conv,input_dim,input_dim)
        return x2
    
    
class Debruiteur_hybride(nn.Module):
    def __init__(self):
        super(Debruiteur_hybride,self).__init__()
        self.NN = nn.Sequential(
            nn.Conv2d(input_conv,9,kernel_size,stride),
            nn.ReLU(),
            nn.Conv2d(9,81,kernel_size,stride),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(81,243,kernel_size,stride),
            nn.Sigmoid(),
            nn.ConvTranspose2d(243,81,kernel_size,stride),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.ConvTranspose2d(81,9,kernel_size+1,stride),
            nn.ReLU(),
            nn.ConvTranspose2d(9,input_conv,kernel_size,stride),
            nn.Sigmoid()
            )
        self.f1 = nn.Linear(input_conv*input_dim**2,input_conv*input_dim**2)
        self.f2 = nn.Linear(input_conv*input_dim**2,input_conv*input_dim**2)
    def forward(self,x):
        x = self.NN(x)
        
        
        x = x.view(x.size(0),1,1,-1)
        x = self.f1(x)
        x = F.relu(x)
        x = self.f2(x)
        x2 = F.sigmoid(x)
        x2 = x2.view(x2.size(0),input_conv,input_dim,input_dim)
        return x2
    
    
def loss_function(x,x_recon):
    BCE = F.binary_cross_entropy(x_recon, x, reduction='sum')
    return BCE
def train(model, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        overall_loss = 0
        u=0
        it=iter(loader_init)
        for batch_idx, (x_bruit, _) in enumerate(loader_bruit):
            x_init= next(it)[0]
            
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
model = Debruiteur_lin()
#%%
optimizer = optim.Adam(model.parameters(), lr=1e-3)
#%%Entrainenement de notre modèle
train(model, optimizer, epochs=50, device=device)

#%%Sauvegarde de notre modèle
file_path ="C:/Users/maxim/Desktop/IMI/TDLOG/Projet_TdLog/modele_entraine_debruiteur_32_100e.pt"
torch.save(model, file_path)
#%%
file_path = "C:/Users/titou/Fichiers Infos/Python/tdlog_proj/Projet_TdLog/modele_entraine_debruiteur_32_100e.pt"
model = torch.load(file_path)
#%% Affichage de l'image initiale et de l'image en sortie du modèle

a=random.randint(0,30)
image,_ = subset_init.__getitem__(a)
image_bruit,_=subset_bruit.__getitem__(a)
with torch.no_grad():
    image = image.to(device)
    recon_image = model(model(image_bruit.unsqueeze(0)))[0]
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
axes[2].imshow(recon_image)



axes[2].set_title("Reconstructed")
axes[2].axis('off')
axes[1].imshow(image_bruit)
axes[1].set_title("bruit")
axes[1].axis('off')
plt.show()

#%% 