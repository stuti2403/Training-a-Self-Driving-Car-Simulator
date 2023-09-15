import matplotlib.pyplot as plt
print('Setting up')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from utils import *
from sklearn.model_selection import train_test_split
#step1
path='myData'
data=ImportDataInfo(path)
#step2
balanceData(data,display=True)
#step3

imagespath,steering=loaddata(path,data)
print(imagespath[0],steering[0])
#step4
xTrain,xVal,yTrain,yVal=train_test_split(imagespath,steering,test_size=0.2,random_state=5)
print('total training images: ',len(xTrain))
print('total validation images: ',len(xVal))

#step 5

#step 6

#step 7
#step 8
model=createModel()
model.summary()

#step 9
history=model.fit(batchgen(xTrain,yTrain,100,1),steps_per_epoch=300,epochs=10,
          validation_data=batchgen(xVal,yVal,100,0),validation_steps=200)


#step 10
model.save('model.h5')
print("model saved")

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training','Validation'])
plt.ylim([0,1])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()





