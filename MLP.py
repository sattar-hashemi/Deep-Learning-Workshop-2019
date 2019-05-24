############## 37 #################################################
import os
import numpy as np
import keras


#from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense




batch_size = 128
num_classes = 10
epochs = 20

##################### Data #########################################

    ############## Load Mnist ######################################
data_dir = '/content/gdrive/My Drive/Mnist'

fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
loaded = np.fromfile(file=fd, dtype=np.uint8)
x_train = loaded[16:].reshape((60000, 28, 28))

fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
loaded = np.fromfile(file=fd, dtype=np.uint8)
y_train = loaded[8:]

fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
loaded = np.fromfile(file=fd, dtype=np.uint8)
x_test = loaded[16:].reshape((10000, 28, 28))

fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
loaded = np.fromfile(file=fd, dtype=np.uint8)
y_test = loaded[8:]
   ##################################################################



   ############### Load CIFAR #######################################
#(x_train, y_train), (x_test, y_test) = cifar10.load_data()
   ##################################################################



x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255



y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


#####################################################################


######################## Model ######################################

model = Sequential()
model.add(Dense(512, activation='sigmoid', input_shape=x_train.shape[1:]))
model.add(Dense(512, activation='sigmoid'))
model.add(Dense(num_classes, activation='sigmoid'))
#####################################################################


model.summary()


################## Compile-Fit-Evaluate #############################
model.compile(loss='mean_squared_error',
              optimizer='Adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=128,
                    epochs=epochs,
                   )


score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

score = model.evaluate(x_train, y_train)
print('Train loss:', score[0])
print('Train accuracy:', score[1])