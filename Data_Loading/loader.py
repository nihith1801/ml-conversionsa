# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Define constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Define data loading function
def load_data(data_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, binary=True):
    if binary:
        class_mode = 'binary'
    else:
        class_mode = 'categorical'

    datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        validation_split=0.2  # 20% validation split
    )

    train_gen = datagen.flow_from_directory(
        directory=data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode=class_mode,
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        directory=data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode=class_mode,
        subset='validation'
    )

    return train_gen, val_gen

# Define data augmentation function
def augment_data():
    datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    return datagen
