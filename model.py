import csv
import cv2
import numpy as np
import os
from collections import namedtuple

from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Lambda, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers import Cropping2D

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

EPOCHS = 6
BATCH = 32
VAL_SPLIT = 0.2
LEFT_CAM_CORRECTION = 0.25
RIGHT_CAM_CORRECTION = -LEFT_CAM_CORRECTION
FLIP_IMAGES = True

ImageMetaFile = namedtuple('ImageMetaFile', ['path', 'center_image_only'])

IMAGE_FILE_PATH = './data/IMG/'
METADATA_FILES = [ ImageMetaFile('./data/smooth_log.csv', False),
#                   ImageMetaFile('./data/additional_smooth_log.csv', False),
                   ImageMetaFile('./data/correction_log.csv', False),
                   ImageMetaFile('./data/tight_turn_log.csv', False) ]

ImageRecord = namedtuple('ImageRecord', ['filename', 'steering_angle', 'flip'])


def valid_measurement(measurement):
    return max(-1.0, min(1.0, measurement))


def load_image(filename, flip):
    image = cv2.imread(os.path.join(IMAGE_FILE_PATH, filename))
    if flip:
        return cv2.flip(image, 1)
    return image


# Generator code from Udacity SDC course
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                images.append(load_image(batch_sample.filename, batch_sample.flip))
                angles.append(batch_sample.steering_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


def load_meta_records(metafiles):
    records = []
    for metafile in metafiles:
        with open(metafile.path) as f:
            reader = csv.reader(f)
            for line in reader:
                angle = float(line[3])
                records.append(ImageRecord(filename=line[0].split('/')[-1], steering_angle=angle, flip=False))
                if FLIP_IMAGES:
                    records.append(ImageRecord(filename=line[0].split('/')[-1], steering_angle=-angle, flip=True))
                if not metafile.center_image_only:
                    records.append(ImageRecord(filename=line[1].split('/')[-1], steering_angle=angle + LEFT_CAM_CORRECTION, flip=False))
                    records.append(ImageRecord(filename=line[2].split('/')[-1], steering_angle=angle + RIGHT_CAM_CORRECTION, flip=False))
                    if FLIP_IMAGES:
                        records.append(ImageRecord(filename=line[1].split('/')[-1], steering_angle=-angle - LEFT_CAM_CORRECTION, flip=True))
                        records.append(ImageRecord(filename=line[2].split('/')[-1], steering_angle=-angle - RIGHT_CAM_CORRECTION, flip=True))
    return records


def create_nvidia_model():
    model = Sequential()
    model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))
    model.add(Conv2D(24, 5, 5, activation='relu', init='he_normal', subsample=(2,2)))
    model.add(Conv2D(36, 5, 5, activation='relu', init='he_normal', subsample=(2,2)))
    model.add(Conv2D(48, 5, 5, activation='relu', init='he_normal', subsample=(2,2)))
    model.add(Conv2D(64, 3, 3, activation='relu', init='he_normal'))
    model.add(Conv2D(64, 3, 3, activation='relu', init='he_normal'))
    model.add(Flatten())
#    model.add(Dense(100, activation='relu', init='he_normal'))
    model.add(Dense(100, init='he_normal'))
    model.add(Dropout(0.5))
#    model.add(Dense(50, activation='relu', init='he_normal'))
    model.add(Dense(50, init='he_normal'))
    model.add(Dropout(0.5))
#    model.add(Dense(10, activation='relu', init='he_normal'))
    model.add(Dense(10, init='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model


def main():
    samples = load_meta_records(METADATA_FILES)
    train_samples, validation_samples = train_test_split(samples, test_size=VAL_SPLIT)
    train_generator = generator(train_samples, batch_size=BATCH)
    validation_generator = generator(validation_samples, batch_size=BATCH)

    import sys
    print('TOTAL SAMPLES:', len(samples))
    print('SAMPLE MEM (B):', sys.getsizeof(samples))
    print('TRAIN SAMPLES:', len(train_samples))
    print('VAL SAMPLES:', len(validation_samples))

    model = create_nvidia_model()
    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=EPOCHS)

    model.save('model.h5')


if __name__ == '__main__':
    main()
