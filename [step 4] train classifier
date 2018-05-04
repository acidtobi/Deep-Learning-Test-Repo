import glob
import re
from image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import model_from_json
#from keras.preprocessing import image
import keras
from keras import backend as K
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from multiprocessing import Pool
import os

def worker(args):
    return train_generator.next()

TRAIN_NETWORK = False

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

#Input image dimensions
input_shape = (40, 40)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(40, 40, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

#model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
#model.add(Conv2D(128, (3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.3,
        rotation_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        adaptive_equalization=True,
        horizontal_flip=True,
        vertical_flip=True)

if TRAIN_NETWORK:

    train_generator = train_datagen.flow_from_directory(
            '/tmp/chronext',
            target_size=input_shape,
            batch_size=32,
            class_mode='categorical')

    pool = Pool(processes=8)

    data = pool.map(worker, range(400))
    x_train, y_train = list(map(list, zip(*data)))
    x_train = np.vstack(x_train)
    y_train = np.vstack(y_train)
    print(x_train.shape)

    indices = np.random.permutation(x_train.shape[0])
    training_idx, test_idx = indices[:10800], indices[10800:]
    x_training, x_test = x_train[training_idx,:], x_train[test_idx,:]
    y_training, y_test = y_train[training_idx,:], y_train[test_idx,:]

    #x_train, y_train = train_generator.next()

    model.fit(x_training, y_training, epochs=10, validation_data=(x_test, y_test))

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

else:
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model.h5")
    print("Loaded model from disk")

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    train_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
            'extracted_by_haarcascade',
            target_size=input_shape,
            batch_size=10,
            shuffle=False,
            class_mode='categorical')

    score = model.predict_generator(train_generator, 5479)

    files = train_generator.filenames
    count = 0
    for filename, probas in zip(files, score):
        if probas[0] > 0.5:
            count += 1
            print("%.2f" % probas[0], filename)
            os.rename("extracted_by_haarcascade/%s" % filename, "extracted_by_haarcascade_falsepositives/%05d.jpg" % count)

sys.exit()

validation_generator = test_datagen.flow_from_directory(
        '../images',
        target_size=input_shape,
        batch_size=32,
        class_mode='categorical')

model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800)
