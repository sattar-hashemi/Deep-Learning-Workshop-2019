###################### 37 ################################################

from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers

############ Load Pre-trained Model ###########################################
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

for layer in vgg_conv.layers[:-4]:
    layer.trainable = False
 

for layer in vgg_conv.layers:
    print(layer, layer.trainable)


 
############ Add Layer to Pre-trained Model ####################################
model = models.Sequential()

model.add(vgg_conv)
 

model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))
 

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
# Train the model
history = model.fit(
      x_train,y_train,
      epochs=30,
     )
	 
	 
