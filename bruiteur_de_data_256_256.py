# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 19:45:03 2023

@author: maxim
"""

import os
import shutil
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
alpha=0.125

input_dim=256
transform = transforms.Compose([
    transforms.ToTensor()
])

def bruit(x):        
        for i in range(3):
            for j in range (x.size(1)):
                for h in range (x.size(2)):
                    x[i][j][h]=(random.randint(0,255)/255)*alpha+(1-alpha)*x[i][j][h]
                    
#%%
folder_path="C:/Users/maxim/Desktop/IMI/TDLOG/Projet_TdLog/CAT3"
train_set = datasets.ImageFolder(root=folder_path, transform=transform)
train_loader = DataLoader(train_set, batch_size=77, shuffle=False)


#%%
original_folder_path = 'C:/Users/maxim/Desktop/IMI/TDLOG/Projet_TdLog/CAT3'
noisy_folder_path = 'C:/Users/maxim/Desktop/IMI/TDLOG/Projet_TdLog/CATBRUIT'


if not os.path.exists(noisy_folder_path):
    os.makedirs(noisy_folder_path)

original_dataset = datasets.ImageFolder(root=original_folder_path, transform=transform)

for class_dir in os.listdir(original_folder_path):
    class_path = os.path.join(original_folder_path, class_dir) 
    noisy_class_path = os.path.join(noisy_folder_path, class_dir)
    
    if not os.path.exists(noisy_class_path):
        os.makedirs(noisy_class_path)
    
    for image_file in os.listdir(class_path):
        image_path = os.path.join(class_path, image_file)
        noisy_image_path = os.path.join(noisy_class_path, image_file)
        
        image = Image.open(image_path)
        
        
        image = transforms.ToTensor()(image)
        bruit(image)
        image = image.numpy()
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image.transpose(1, 2, 0))
    
        image.save(noisy_image_path)


