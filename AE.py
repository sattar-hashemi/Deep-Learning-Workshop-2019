########################## 37 #########################################
import numpy as np
import keras

from keras.layers import  Dense, Input
from keras.models import Model

######################## Data ##########################################

fd = open('./data/mnist/train-images-idx3-ubyte')
loaded = np.fromfile(file=fd, dtype=np.uint8)
x_train = loaded[16:].reshape((60000, 28, 28))


fd = open('./data/mnist/t10k-images-idx3-ubyte')
loaded = np.fromfile(file=fd, dtype=np.uint8)
x_test = loaded[16:].reshape((10000, 28, 28))

x_train = np.reshape(x_train, [60000, 784])
x_test = np.reshape(x_test, [10000, 784])
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255


#########################################################################

tInputs = Input(shape=(784,))

tEncoded = Dense(128, activation='relu')(tInputs)
tEncoded = Dense(64, activation='relu')(tEncoded)
tEncoded = Dense(32, activation='relu')(tEncoded)
tEncoded = Dense(16, activation='relu')(tEncoded)

tLatent = tEncoded

tDecoded = Dense(64, activation='relu')(tLatent)
tDecoded = Dense(128, activation='relu')(tDecoded)
tDecoded = Dense(784, activation='sigmoid')(tDecoded)

tOutputs = tDecoded

autoencoder = Model(tInputs, tOutputs)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))