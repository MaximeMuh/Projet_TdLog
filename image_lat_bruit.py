import random
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset
#%%
file_path = "C:/Users/titou/Fichiers Infos/Python/tdlog_proj/Projet_TdLog/modele_vae_2layerconv_50ep.pt"
model = torch.load(file_path, map_location=torch.device('cpu'))

#%%
input_dim = 128
path_init="C:/Users/titou/Fichiers Infos/Python/tdlog_proj/Projet_TdLog/image_esp_lat/a_init/image"
path_bruit="C:/Users/titou/Fichiers Infos/Python/tdlog_proj/Projet_TdLog/image_esp_lat/b_bruit/image"
path_cat="C:/Users/titou/Fichiers Infos/Python/tdlog_proj/Projet_TdLog/CAT3"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.Resize((input_dim, input_dim)),transforms.ToTensor()])
train_set = datasets.ImageFolder(root=path_cat, transform = transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=False)
#%%
alpha=0.10
input_dim=32
def bruit(x):
    y=torch.clone(x)
    for i in range(3):
        for j in range (input_dim):
            for h in range (input_dim):
                y[i][j][h]=(random.randint(0,255)/255)*alpha+(1-alpha)*y[i][j][h]
    return(y)

#%%
input_dim=32
u=0
for image, _ in train_loader:
    mu = model(image)[1]
    logvar = model(image)[2]
    images_lat=torch.randn_like(torch.exp(0.5*logvar)).mul(torch.exp(0.5*logvar)).add(mu)
    for image_lat in images_lat:
        u+=1
        image_lat=image_lat.to(device)
        image_lat = image_lat.view(3,input_dim,input_dim)
        image_lat_bruit=bruit(image_lat)
        image_pil = transforms.ToPILImage()(image_lat)
        image_bruit_pil = transforms.ToPILImage()(image_lat_bruit)
        image_pil.save(path_init+f"{u}"+".png")
        image_bruit_pil.save(path_bruit+f"{u}"+".png")
        









