######################## 37 ###############################
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D


batch_size = 32
num_classes = 10
epochs = 100

####################### Data ##############################
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255




y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


########################## Model ###########################


Hypothesis = Sequential()
Hypothesis.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
Hypothesis.add(Activation('relu'))
Hypothesis.add(Conv2D(32, (3, 3)))
Hypothesis.add(Activation('relu'))
Hypothesis.add(MaxPooling2D(pool_size=(2, 2)))
Hypothesis.add(Dropout(0.25))

Hypothesis.add(Conv2D(64, (3, 3), padding='same'))
Hypothesis.add(Activation('relu'))
Hypothesis.add(Conv2D(64, (3, 3)))
Hypothesis.add(Activation('relu'))
Hypothesis.add(MaxPooling2D(pool_size=(2, 2)))
Hypothesis.add(Dropout(0.25))

Hypothesis.add(Flatten())
Hypothesis.add(Dense(512))
Hypothesis.add(Activation('relu'))
Hypothesis.add(Dropout(0.5))
Hypothesis.add(Dense(num_classes))
Hypothesis.add(Activation('softmax'))






############################## Compile-Fit-Evaluate ############################

Hypothesis.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

Hypothesis.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)

Hypothesis.save('keras_cifar10_trained_model.h5')

scores = Hypothesis.evaluate(x_test, y_test)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])