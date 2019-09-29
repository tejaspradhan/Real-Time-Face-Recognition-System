# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 11:09:48 2019

@author: Admin
"""
import os
import numpy as np

import cv2
import pickle
import random 

train_data = []
CATEGORIES = ['Rashi', 'Tejas']
DATADIR = 'E:/Projects/Real-Time Face Recognition System'
size = 64
def creating_train():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (size,size))
                train_data.append([new_array, class_num])
            except Exception as e:
                pass
creating_train()
X = []
y = []
for features, labels in train_data:
    X.append(features)
    y.append(labels)
X = np.array(X)

pickle_out = open('X.pickle','wb')
pickle.dump(X,pickle_out)
pickle_out.close()
pickle_out = open('y.pickle','wb')
pickle.dump(y,pickle_out)
pickle_out.close()

