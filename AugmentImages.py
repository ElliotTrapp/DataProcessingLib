'''
AugmentImages.py
Elliot Trapp
18/12/3

Utilities for augmenting image training data to create dynamic training data.
'''

import numpy as np
from keras.preprocessing.image import ImageDataGenerator


def GetTrainGenerator(load_dir, batch_size=32, target_size=(150,150), validation_split=0.0):

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=validation_split,
            #preprocessing_function=preprocess_input
            )

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data

    train_generator = train_datagen.flow_from_directory(
            directory=load_dir,  # this is the target directory
            target_size=target_size,  # all images will be resized to 150x150
            batch_size=batch_size,
            class_mode='categorical',
            subset='training')  # since we use binary_crossentropy loss, we need binary labels

    return train_generator

def GetTestGenerator(load_dir, batch_size=32, target_size=(150,150), validation_split=0.0):

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1./255,
                                    validation_split=validation_split,
                # preprocessing_function=preprocess_input
                                    )

    # this is a similar generator, for validation data
    test_generator = test_datagen.flow_from_directory(
            directory=load_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation')

    return test_generator