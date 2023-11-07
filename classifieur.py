import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
#%%

folder_path = "C:/Users/titou/Fichiers Infos/Python/tdlog_proj/archive(1)/raw-img"

train_size = int(0.8 * len(train_set))  # Par exemple, 80% pour l'entraînement, 20% pour le test
test_size = len(train_set) - train_size

# Utilisez random_split pour diviser le train_set en deux ensembles
train_dataset, test_dataset = random_split(train_set, [train_size, test_size])

# Créez des DataLoaders distincts pour chaque ensemble
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

#%% CLassifieur

class Net(torch.nn.Module):
    def __init__(self): 
        super(Net, self).__init__()
        self.network = nn.Sequential(
            torch.nn.Conv2d(3,9,4,2, padding=1), 
            torch.nn.ReLU(),
            torch.nn.Conv2d(9,27,4,2, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(27,81,2,2, padding=0),           
            torch.nn.ReLU(),
            torch.nn.Flatten(1),
            torch.nn.Linear(20736,2),
            torch.nn.Softmax(dim=1)
        )
    def forward(self,im):
        return(self.network(im))
#%% Training  -  LABEL 0

def train(model, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (x, labels) in enumerate(train_loader):
            x = x.to(device)
            y = torch.zeros(64,2)
            for i in range(64):
                if labels.tolist()[i]==0:
                    y[i,0] = 1
                else:
                    y[i,1] = 1
            optimizer.zero_grad()
            prediction = model(x)
            loss = F.binary_cross_entropy(prediction, y, reduction='sum')
            overall_loss += loss.item()
            print(batch_idx/len(train_loader), loss.item())
            loss.backward()
            optimizer.step()

        print("Epoch", epoch + 1, "Average Loss:", overall_loss / (batch_idx * batch_size))
    return overall_loss

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#%%
train(model, optimizer, epochs=1, device=device) 

#%% renvoie ( proba(animal est de type 0) , proba(animal est de type different de 0))

x, labels = iter(train_loader).next()
prediction=model(x)

y = torch.zeros(64,2)
for i in range(64):
    if labels.tolist()[i]==0:
        y[i,0] = 1
    else:
        y[i,1] = 1
for i in range(64):
    print(prediction[i],y[i])