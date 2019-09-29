# Developing a face recognition system using PCA and machine learning 

##STEPS :
# 1) Prepare dataset by flattening Images
# 2) Apply PCA on the dataset to reduce dimensions
# 3) Train the classifier with this dataset of reduced features
# 4) Test  

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA 
from sklearn.svm import SVC
import pickle
import cv2 


def show_orignal_images(pixels):
    #Displaying Orignal Images
    fig, axes = plt.subplots(17, 4, figsize=(11, 7),subplot_kw={'xticks':[], 'yticks':[]})
    for i, ax in enumerate(axes.flat):
	    ax.imshow(np.array(pixels)[i].reshape(64, 64), cmap='gray')
    plt.show()
    
def show_eigenfaces(pca):
	#Displaying Eigenfaces
	fig, axes = plt.subplots(3, 8, figsize=(9, 4),subplot_kw={'xticks':[], 'yticks':[]})
	for i, ax in enumerate(axes.flat):
	    ax.imshow(pca.components_[i].reshape(64, 64), cmap='gray')
	    ax.set_title("PC " + str(i+1))
	plt.show()  

# Reading the Dataset 
#df = pd.read_csv('face_data.csv')
pickle_in = open('X.pickle','rb')
pixels = pickle.load(pickle_in)
pickle_in = open('y.pickle','rb')
labels = pickle.load(pickle_in)
show_orignal_images(pixels)
#Train Test Split
x_train,x_test,y_train,y_test = train_test_split(pixels,labels,test_size = 0.33)
# Performing PCA
pca =PCA(n_components = 200).fit(x_train)
#plt.plot(pca.explained_variance_ratio_)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
show_eigenfaces(pca)

#Training the Model
new_xtrain = pca.transform(x_train)
classifier = SVC(kernel ='poly',C=1, gamma=0.01)
model = classifier.fit(new_xtrain,y_train)

new_x_test = pca.transform(x_test)
predictions = model.predict(new_x_test)
print(y_test)
print(predictions)
prev =" "
count=0

face_cascade = cv2.CascadeClassifier('E:/Projects/Real-Time Face Recognition System/cascades/haarcascade_frontalface_alt2.xml')
cap = cv2.VideoCapture(0)
while(1):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
#img = cv2.imread(r"E:\Projects\Real-Time Face Recognition System\rashi.jpeg",0)
    for (x, y, w, h) in faces:
        #print(x,y,w,h)
        roi_gray = np.array(gray[y:y+h, x:x+w])
       # cv2.imshow("Face detected",roi_gray)
        gray = cv2.resize(gray,(64,64))
        new_img = pca.transform(gray.flatten().reshape(1,-1))
        p = model.predict(new_img)
        
        if(p==0):
            l = 'Rashi'
        else:
            l = 'Tejas'
        #cv2.putText(frame,l)
        if(prev!=l):
            count+=1
            print(l)
        prev = l
        cv2.rectangle(frame,( x,y), (x+w,y+h), (255,00,00))
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()













print('Number of People:')
print(2)

