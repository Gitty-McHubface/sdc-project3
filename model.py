import csv
import cv2
import numpy as np
import os
from collections import namedtuple

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.layers import Cropping2D


EPOCHS = 15
VAL_SPLIT = 0.3
LEFT_CAM_CORRECTION = 0.25
RIGHT_CAM_CORRECTION = -LEFT_CAM_CORRECTION

ImageMetaFile = namedtuple('ImageMetaFile', ['path', 'center_image_only', 'flip_images'])

IMAGE_FILE_PATH = './data/IMG/'
METADATA_FILES = [ ImageMetaFile('./data/smooth_log.csv', False, True),
#                   ImageMetaFile('./data/additional_smooth_log.csv', False, True),
                   ImageMetaFile('./data/correction_log.csv', False, True),
                   ImageMetaFile('./data/tight_turn_log.csv', False, True) ]


def valid_measurement(measurement):
    return max(-1.0, min(1.0, measurement))


def load_image_data(line, img_index):
    filename = line[img_index].split('/')[-1]
    image = cv2.imread(os.path.join(IMAGE_FILE_PATH, filename))
    measurement = float(line[3])
    return (image, measurement)


def process_image(image, measurement, flip, correction=0.0):
    images = [image]
    measurements = [valid_measurement(measurement + correction)]

    if flip:
        images.append(cv2.flip(image, 1))
        measurements.append(valid_measurement(-measurement - correction))

    return (images, measurements)


def load_driving_data(metafile, center_image_only=False, flip=True):
    images = []
    measurements = []

    with open(metafile) as f:
        reader = csv.reader(f)
        for line in reader:
            center_image, center_measurement = load_image_data(line, 0)
            center_images, center_measurements = process_image(center_image, center_measurement, flip)
            images += center_images
            measurements += center_measurements

            if not center_image_only:
                left_image, left_measurement = load_image_data(line, 1)
                right_image, right_measurement = load_image_data(line, 2)

                left_images, left_measurements = process_image(left_image, left_measurement, False, LEFT_CAM_CORRECTION)
                right_images, right_measurements = process_image(right_image, right_measurement, False, RIGHT_CAM_CORRECTION)

                images += left_images + right_images
                measurements += left_measurements + right_measurements

    return (images, measurements)


def create_nvidia_model():
    model = Sequential()
    model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))
    model.add(Conv2D(24, 5, 5, activation='relu', init='he_normal', subsample=(2,2)))
    model.add(Conv2D(36, 5, 5, activation='relu', init='he_normal', subsample=(2,2)))
    model.add(Conv2D(48, 5, 5, activation='relu', init='he_normal', subsample=(2,2)))
    model.add(Conv2D(64, 3, 3, activation='relu', init='he_normal'))
    model.add(Conv2D(64, 3, 3, activation='relu', init='he_normal'))
#    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', init='he_normal'))
#    model.add(Dense(100, init='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu', init='he_normal'))
#    model.add(Dense(50, init='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu', init='he_normal'))
#    model.add(Dense(10, init='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model

    
def main():
    images = []
    measurements = []
    for meta_file in METADATA_FILES:
        i, m = load_driving_data(meta_file.path, meta_file.center_image_only, meta_file.flip_images)
        images += i
        measurements += m
        
    X_train = np.array(images)
    y_train = np.array(measurements)
    print('X_train shape:', X_train.shape)

    from keras.optimizers import Adam
    opt = Adam(lr=0.0001)
   
    model = create_nvidia_model()

    for layer in model.layers:
        print(layer.get_config())
        print()

    print()
    print(model.summary())

    model.compile(loss='mse', optimizer=opt)#'adam')
    model.fit(X_train, y_train, nb_epoch=EPOCHS, validation_split=VAL_SPLIT, shuffle=True)

    model.save('model.h5')


if __name__ == '__main__':
    main()
