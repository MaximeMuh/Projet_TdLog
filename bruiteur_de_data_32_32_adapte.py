import pickle
import random
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset
#%%
file = "C:/Users/titou/Fichiers Infos/Python/tdlog_proj/Projet_TdLog/image_32_32"
noisy_path="C:/Users/titou/Fichiers Infos/Python/tdlog_proj/Projet_TdLog/image_32_32_30prct/b_5prct/image"
correct_path="C:/Users/titou/Fichiers Infos/Python/tdlog_proj/Projet_TdLog/image_32_32_30prct/a_init/image"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.ToTensor()
])
train_set = datasets.ImageFolder(root=file, transform = transform)
my_dataset = DataLoader(train_set, batch_size=64, shuffle=False)

#%%fontion brouillage
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
image,_ = iter(my_dataset).next()
i12 = image[12]
bruit(i12)
image1 = np.transpose(i12, (1, 2, 0))
plt.imshow(image1)
plt.show()  
#%%
bruit(image)
from PIL import Image
image1 = Image.fromarray(image.transpose(1, 2, 0))
image.save(noisy_path)

#%%copie des images brouillées

u=0
for k in range (len(train_set)):
    image,_  = train_set.__getitem__(k)
    image_bruit=bruit(image)
    u+=1
    image=image.to(device)
    image_bruit=image_bruit.to(device)
    image_pil = transforms.ToPILImage()(image)
    image_bruit_pil = transforms.ToPILImage()(image_bruit)
    image_bruit_pil.save(noisy_path+f"{u}"+".png")
    image_pil.save(correct_path+f"{u}"+".png")
    if u>7070:
        break