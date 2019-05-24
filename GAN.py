##################### 37 #####################################
import numpy as np
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt


###########################  Data #############################
fd = open('./data/mnist/train-images-idx3-ubyte')
loaded = np.fromfile(file=fd, dtype=np.uint8)
x_train = loaded[16:].reshape((60000, 28, 28,1))

fd = open('./data/mnist/t10k-images-idx3-ubyte')
loaded = np.fromfile(file=fd, dtype=np.uint8)
x_test = loaded[16:].reshape((10000, 28, 28,1))

X = np.concatenate((x_train, x_test), axis=0)

X = X / 127.5 - 1.
##############################################################

img_shape = (28, 28, 1)
latent_dim = 100
optimizer = Adam(0.0002, 0.5)

################ Discriminator Architecture ################################

H = Sequential()

H.add(Flatten(input_shape=img_shape))
H.add(Dense(512))
H.add(LeakyReLU(alpha=0.2))
H.add(Dense(256))
H.add(LeakyReLU(alpha=0.2))
H.add(Dense(1, activation='sigmoid'))
H.summary()

image = Input(shape=img_shape)
fake_valid_class = H(image)


discriminator = Model(image, fake_valid_class)

discriminator.compile(loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy'])

################################# Generator Architecture ################

H = Sequential()

H.add(Dense(256, input_dim=latent_dim))
H.add(LeakyReLU(alpha=0.2))
H.add(BatchNormalization(momentum=0.8))
H.add(Dense(512))
H.add(LeakyReLU(alpha=0.2))
H.add(BatchNormalization(momentum=0.8))
H.add(Dense(1024))
H.add(LeakyReLU(alpha=0.2))
H.add(BatchNormalization(momentum=0.8))
H.add(Dense(np.prod(img_shape), activation='tanh'))
H.add(Reshape(img_shape))

H.summary()

noise = Input(shape=(latent_dim,))
img = H(noise)

generator = Model(noise, img)

########################### GAN Architecture ####################################

z = Input(shape=(latent_dim,))
img = generator(z)
discriminator.trainable = False
fake_valid_class = discriminator(img)

GAN = Model(z, fake_valid_class)
GAN.compile(loss='binary_crossentropy', optimizer=optimizer)

################################### Training #####################################


epochs = 30000
batch_size = 32

valid = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

for epoch in range(epochs):

    index = np.random.randint(0, X.shape[0], batch_size)
    batch_real_images = X[index]
    noise = np.random.normal(0, 1, (batch_size, latent_dim))

    generated_images = generator.predict(noise)

    discriminator_loss_value_real = discriminator.train_on_batch(batch_real_images, valid)
    discriminator_loss_value_fake = discriminator.train_on_batch(generated_images, fake)
    discriminator_loss_value = 0.5 * np.add(discriminator_loss_value_real, discriminator_loss_value_fake)


    noise = np.random.normal(0, 1, (batch_size, latent_dim))

    generator_loss_value = GAN.train_on_batch(noise, valid)


    params = '%d accuracy.: %.2f%%  discriminator: %f  generator: %f'
    print(params % (epoch, 100 * discriminator_loss_value[1] ,discriminator_loss_value[0], generator_loss_value))


    if epoch % 100 == 0:
        noise = np.random.normal(0, 1, (36, latent_dim))
        gen_imgs = generator.predict(noise)

        gen_imgs = 0.5 * gen_imgs + 0.5

        f, a = plt.subplots(6, 6)
        f_index = 0
        for i in range(6):
            for j in range(6):
                a[i, j].imshow(gen_imgs[f_index, :, :, 0], cmap='gray')
                a[i, j].axis('off')
                f_index += 1
        f.savefig("./%d.png" % epoch)
        plt.close()