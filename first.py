from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import os
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Dropout, MaxPooling2D, Flatten
from tensorflow.keras.applications import VGG16, InceptionV3
from tensorflow.keras.callbacks import EarlyStopping


train_dir='./data/else/char/train/upper/'
train_datagen = ImageDataGenerator(rescale=1./255,
    rotation_range=20, shear_range=0.1,
    width_shift_range=0.1, height_shift_range=0.1,
    zoom_range=0.1, horizontal_flip=False, fill_mode='nearest')
# train_generator = train_datagen.flow_from_directory(train_dir, target_size=(84, 84)
#                     , color_mode='grayscale', batch_size=20, class_mode='categorical')
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150)
                    , batch_size=64, class_mode='categorical')

valid_dir='./data/else/char/valid/upper/'
valid_datagen = ImageDataGenerator(rescale=1./255)
# train_generator = train_datagen.flow_from_directory(train_dir, target_size=(84, 84)
#                     , color_mode='grayscale', batch_size=20, class_mode='categorical')
valid_generator = valid_datagen.flow_from_directory(valid_dir, target_size=(150, 150)
                    , batch_size=64, class_mode='categorical')


input_shape = train_generator[0][0].shape[1:]
output_shape = train_generator[0][1].shape[1]

model4 = Sequential()
conv_base = InceptionV3(weights='imagenet',
                    include_top=False,
                    input_shape=input_shape)
conv_base.trainable=False
model4.add(conv_base)
model4.add(Flatten())
model4.add(Dense(512, activation='relu'))
model4.add(Dense(output_shape, activation='softmax'))
model4.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# steps_per_epoch = len(X_train)//batch_size
# validation_steps = len(X_valid)//batch_size
steps_per_epoch = 22127//64
start = time.time()
history4 = model4.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=300,
            # callbacks=[EarlyStopping(monitor='val_loss', patience=1)],
            validation_data = valid_generator,
            validation_steps = 2340//64)

model4.save_weights('./model4.h5')

def plot_acc(h, title="accuracy"):
    plt.plot(h.history['accuracy'])
    plt.plot(h.history['val_accuracy'])
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)
    # plt.legend(['Training'], loc=0)


def plot_loss(h, title="loss"):
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)
    # plt.legend(['Training'], loc=0)

print(time.time - start)
plot_loss(history4)
plt.savefig('./loss_graph.png')
plt.clf()
plot_acc(history4)
plt.savefig('./acc_graph.png')