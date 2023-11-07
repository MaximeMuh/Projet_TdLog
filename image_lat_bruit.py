import random
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset
#%%
file_path = "C:/Users/titou/Fichiers Infos/Python/tdlog_proj/Projet_TdLog/modele_entraine_debruiteur_32_100e.pt"
model = torch.load(file_path)

#%%
input_dim = 128
path_init="C:/Users/titou/Fichiers Infos/Python/tdlog_proj/Projet_TdLog/image_esp_lat/a_init/image"
path_bruit="C:/Users/titou/Fichiers Infos/Python/tdlog_proj/Projet_TdLog/image_esp_lat/a_bruit/image"
path_cat="C:/Users/titou/Fichiers Infos/Python/tdlog_proj/Projet_TdLog/CAT3"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.Resize((input_dim, input_dim)),transforms.ToTensor()])
train_set = datasets.ImageFolder(root=path_cat, transform = transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=False)
#%%
alpha=0.10
input_dim=32
def bruit(x):
    y=x.copy()
    for i in range(3):
        for j in range (input_dim):
            for h in range (input_dim):
                y[i][j][h]=(random.randint(0,255)/255)*alpha+(1-alpha)*y[i][j][h]
    return(y)

#%%
u=0
for image,_ in enumerate(train_loader):
    u+=1
    image_lat = model(image)[3]
    image_lat_bruit = bruit(image_lat)    
    image_lat=image_lat.to(device)
    image_lat_bruit=image_lat_bruit.to(device) 
    image_pil = transforms.ToPILImage()(image_lat)
    image_bruit_pil = transforms.ToPILImage()(image_lat_bruit)
    image_bruit_pil.save(path_bruit+f"{u}"+".png")
    image_pil.save(path_init+f"{u}"+".png")

