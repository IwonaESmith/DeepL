# Convolutional Neural Network


# Part 1 - Building the CNN

# Importing the Keras libraries and packages Kera2 API
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
import numpy as np




# Initialising the CNN
model = Sequential([
        (Conv2D(32,(3,3),input_shape =(64,64,3), activation = 'relu')),
        (MaxPooling2D(pool_size = (2,2))),
        (Flatten()),
        (Dense(128,  activation='relu')),
        (Dense(1,  activation='sigmoid'))
        ])

model.compile(SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
             loss='binary_crossentropy',
             metrics=['accuracy'])

batch_size = 32
dataset_size = 8000

# prepare data augmentation configuration
train_data = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_data = ImageDataGenerator(rescale=1./255)

train_set = train_data.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='binary')

validation_generator = test_data.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='binary')

# Part 2 Fitting the CNN to the images 

model.fit_generator(
        train_set,
        steps_per_epoch = np.ceil(dataset_size / batch_size),
        epochs=25,
        validation_data=validation_generator,
        validation_steps= 63,
        verbose = 2)


from keras.callbacks import LambdaCallback

callbacks = callbacks=[LambdaCallback(on_batch_end=lambda batch,logs:print(logs))]

model.save_weights('first_try.h5') 





