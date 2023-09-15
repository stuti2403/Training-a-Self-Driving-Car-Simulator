import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import matplotlib.image as mpimg
#matplotlib gives rgb images; opencv/cv2 gives bgr
from imgaug import augmenters as iaa
import cv2
import random
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Convolution2D,Flatten,Dense
from tensorflow.python.keras.optimizer_v2.adam import Adam

def getName(filepath):
    return filepath.split('\\')[-1]

def ImportDataInfo(path):
    columns = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']
    data = pd.read_csv(os.path.join(path, 'driving_log.csv'), names = columns)
    #### REMOVE FILE PATH AND GET ONLY FILE NAME
    #print(getName(data['Center'][0]))
    data['Center']=data['Center'].apply(getName)
    #print(data.head())
    print('Total Images Imported',data.shape[0])
    return data

def balanceData(data,display=True):
    nBins=31
    samplePerBin=100
    hist, bins= np.histogram(data['Steering'],nBins)
    #print(bins)
    if display:
        '''center=(bins[:-1]+bins[1:])*0.5
        #print(center)
        plt.bar(center,hist,width=0.06)
        plt.plot((-1,1),(samplePerBin,samplePerBin))
        plt.show()'''

    removeIndexList=[]
    for j in range(nBins):
        binDataList=[]
        for i in range(len(data['Steering'])):
            if data['Steering'][i]>=bins[j] and data['Steering'][i]<=bins[j+1]:
                binDataList.append(i)
        binDataList=shuffle(binDataList)
        binDataList=binDataList[samplePerBin:]
        removeIndexList.extend(binDataList)
    print('removed images: ',len(removeIndexList))
    data.drop(data.index[removeIndexList],inplace=True)
    print('remaining images: ',len(data))

    if display:
        '''hist, _ = np.histogram(data['Steering'], nBins)
        plt.bar(center, hist, width=0.06)
        plt.plot((-1, 1), (samplePerBin, samplePerBin))
        plt.show()'''

def loaddata(path,data):
    imagespath=[]
    steering=[]

    for i in range(len(data)):
        indexedData=data.iloc[i]
        #print(indexedData)
        imagespath.append(os.path.join(path,'IMG',indexedData[0]))
        #print(os.path.join(path,'IMG',indexedData[0]))
        steering.append(float(indexedData[3]))
    imagespath=np.array(imagespath)
    steering=np.array(steering)
    return imagespath,steering

def augmentimage(imgpath,steering):
    img=mpimg.imread(imgpath)
    #pan
    if np.random.rand()<0.5:
        pan=iaa.Affine(translate_percent={'x':(-0.1,0.1),'y':(-0.1,0.1)})
        img=pan.augment_image(img)
    #zoom
    if np.random.rand() < 0.5:
        zoom=iaa.Affine(scale=(1,1.2))
        img=zoom.augment_image(img)

    #brightness
    if np.random.rand() < 0.5:
        brightness=iaa.Multiply((0.2,1.2))
        img=brightness.augment_image(img)

    #flip
    if np.random.rand() < 0.5:
        img=cv2.flip(img,1)
        steering=-steering


    return img,steering
'''
res,st=augmentimage('D:/6th sem/nvidia self driving car project/s.jpg',0)
plt.imshow(res)
plt.show()
'''

def preprocessing(img):
    img=img[60:135,:,:]
    img=cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
    img=cv2.GaussianBlur(img,(3,3),0)
    img=cv2.resize(img,(200,66))
    img=img/255
    return img

def batchgen(imagespath,steeringlist,batchsize,trainflag):
    while True:
        imgBatch=[]
        steeringBatch=[]
        for i in range(batchsize):
            index=random.randint(0,len(imagespath)-1)
            if trainflag==True:
                img,steering=augmentimage(imagespath[index],steeringlist[index])
            else:
                img=mpimg.imread(imagespath[index])
                steering=steeringlist[index]
            img=preprocessing(img)
            imgBatch.append(img)
            steeringBatch.append(steering)
        yield(np.array(imgBatch),np.array(steeringBatch))

def createModel():


    model=Sequential()
    model.add(Convolution2D(24,(5,5),(2,2),input_shape=(66,200,3),activation='elu'))
    model.add(Convolution2D(36, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(48, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))


    model.add(Flatten())
    model.add(Dense(100,activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    model.compile(Adam(learning_rate=0.0001),loss='mse')
    return model


