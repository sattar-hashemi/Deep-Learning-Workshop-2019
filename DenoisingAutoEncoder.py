########################## 37 #########################################
import numpy as np
import keras

from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model

import matplotlib.pyplot as plt




######################## Data ##########################################

fd = open('./data/mnist/train-images-idx3-ubyte')
loaded = np.fromfile(file=fd, dtype=np.uint8)
x_train = loaded[16:].reshape((60000, 28, 28))

fd = open('./data/mnist/train-labels-idx1-ubyte')
loaded = np.fromfile(file=fd, dtype=np.uint8)
y_train = loaded[8:]

fd = open('./data/mnist/t10k-images-idx3-ubyte')
loaded = np.fromfile(file=fd, dtype=np.uint8)
x_test = loaded[16:].reshape((10000, 28, 28))

fd = open('./data/mnist/t10k-labels-idx1-ubyte')
loaded = np.fromfile(file=fd, dtype=np.uint8)
y_test = loaded[8:]


x_train = np.reshape(x_train, [60000, 28, 28, 1])
x_test = np.reshape(x_test, [10000, 28, 28, 1])
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255



noise = np.random.normal(loc=0.5, scale=0.5, size=x_train.shape)
x_train_noisy = x_train + noise
noise = np.random.normal(loc=0.5, scale=0.5, size=x_test.shape)
x_test_noisy = x_test + noise



##########################################################################

####################### Meta-Parameters #################################

input_shape = (28, 28, 1)
batch_size = 128
kernel_size = 3
latent_dim = 16
filters = [32, 64]

############################################################################

####################### Encoder ############################################

   ##################### Create Layers #####################################
lConv1 = Conv2D(filters=filters[0],
           kernel_size=kernel_size,
           strides=2,
           activation='relu',
           padding='same')

lConv2 = Conv2D(filters=filters[1],
           kernel_size=kernel_size,
           strides=2,
           activation='relu',
           padding='same')

lFlatten = Flatten()

lDense = Dense(latent_dim)

    #########################################################################

tInputs = Input(shape=input_shape)

tLatent = lDense(lFlatten(lConv2(lConv1(tInputs))))

encoder = Model(tInputs, tLatent)

encoder.summary()

##############################################################################

lDense= Dense(7 * 7 * 64)
lReshape = Reshape((7, 7, 64))
lConvTranspose1 =  Conv2DTranspose(filters=filters[1],
                                   kernel_size=kernel_size,
                                   strides=2,
                                   activation='relu',
                                   padding='same')

lConvTranspose2 =  Conv2DTranspose(filters=filters[0],
                                   kernel_size=kernel_size,
                                   strides=2,
                                   activation='relu',
                                   padding='same')

lConvTranspose3 = Conv2DTranspose(filters=1,
                                  kernel_size=kernel_size,
                                  padding='same')

lActivation = Activation('sigmoid')
    #############################################################################
tLatentInputs = Input(shape=(latent_dim,))
tOutputs = lActivation(lConvTranspose3(lConvTranspose2(lConvTranspose1(lReshape(lDense(tLatentInputs))))))
decoder = Model(tLatentInputs, tOutputs)
decoder.summary()

##################################### AutoEncoder ############################################
autoencoder = Model(tInputs, decoder(encoder(tInputs)))
autoencoder.summary()

autoencoder.compile(loss='mse', optimizer='adam')


autoencoder.fit(x_train_noisy,
                x_train,
                validation_data=(x_test_noisy, x_test),
                epochs=30,
                batch_size=batch_size)


x_decoded = autoencoder.predict(x_test_noisy[0].reshape(1,28,28,1))
plt.imshow(x_test_noisy[0,:,:,0])
plt.imshow(x_decoded[0,:,:,0])