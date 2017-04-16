#!/usr/bin/env python3
import concurrent.futures
import keras
import os
import requests
import shutil
import subprocess
import tarfile

from keras.layers import Activation, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from io import BytesIO


SKIP_RESIZE_AND_CROP = True

TARGET_IMAGE_SIZE = 50


class UserRecoverableException(Exception):
  def __init__(self, message, cause):
    self.message = message
    self.cause = cause

  def __str__(self):
    return "{}\n\t(Cause: {})".format(self.message, self.cause)


def resize_and_crop(source, target, threadpool_max_workers=12):
  try:
    with open(os.devnull) as devnull:
      subprocess.check_call(["convert", "--version"], stdout=devnull, stderr=subprocess.STDOUT)
  except FileNotFoundError as e:
    raise UserRecoverableException("You need to install ImageMagick to resize images.  "
                                   "https://www.imagemagick.org/script/index.php", e)

  def imagemagick_commands():
    for dirName, _, files in os.walk(source):
      for filename in files:
        yield [
          "convert", os.path.join(source, dirName, filename),
          "-resize", "{0}x{0}^".format(TARGET_IMAGE_SIZE),
          "-gravity", "Center",
          "-crop", "{0}x{0}+0+0".format(TARGET_IMAGE_SIZE),
          "+repage", target(dirName, filename)
        ]

  with open(os.devnull) as devnull:
    def run_command(cmd):
      subprocess.check_call(cmd, stdout=devnull, stderr=subprocess.STDOUT)

    with concurrent.futures.ThreadPoolExecutor(max_workers=threadpool_max_workers) as executor:
      executor.map(run_command, imagemagick_commands())

def retrieve_caltech_database(negative_source_dir):
  if not os.path.exists(negative_source_dir):
    print("Caltech101 Object Categories database unavailable, downloading...")
    objCategoriesResponse = requests.get("http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz")
    objCategoriesResponse.raise_for_status()
    print("Download complete, uncompressing...")
    tar = tarfile.open(fileobj=BytesIO(objCategoriesResponse.content),
                       mode='r:gz')
    # The tar file already contains the 101_ObjectCategories directory, so we need to ignore this.
    tar.extractall(os.path.split(negative_source_dir)[0])
  else:
    print("Caltech101 Object Categories database already available.")

def get_working_directory():
  try:
    root_dir = os.path.split(os.path.realpath(__file__))[0]
  except NameError as e:
    raise UserRecoverableException("You can only run this as a script, sorry.", e)
  print("Running in directory: {}".format(root_dir))
  return root_dir

def clean_target_directories(target_dir, negative_target_dir):
  print("Cleaning data directory: {}".format(target_dir))
  if os.path.exists(target_dir):
    shutil.rmtree(target_dir)
  os.mkdir(target_dir)
  os.mkdir(negative_target_dir)
  for i in range(0, 6):
    os.mkdir(os.path.join(target_dir, str(i)))

def main():
  print("Every bit of this is going to eat your CPU, be prepared.")

  root_dir = get_working_directory()
  source_dir = os.path.join(root_dir, "data_originals")
  # This must match the directory inside the Caltech101 gzipped file.
  negative_source_dir = os.path.join(source_dir, "101_ObjectCategories")
  positive_source_dir = os.path.join(source_dir, "positives")

  retrieve_caltech_database(negative_source_dir)

  target_dir = os.path.join(root_dir, "data")
  negative_target_dir = os.path.join(target_dir, "negatives")


  if SKIP_RESIZE_AND_CROP:
    print("Skipping resize and crop, leaving data directory intact")
  else:
    clean_target_directories(target_dir, negative_target_dir)
    print("Running resize and crop on negatives...")
    resize_and_crop(source=negative_source_dir,
                    target=lambda dir_name, filename: os.path.join(negative_target_dir,
                                                                  "{}_{}".format(os.path.split(dir_name)[-1], filename)))
    print("Running resize and crop on positives...")
    resize_and_crop(source=positive_source_dir,
                    target=lambda dir_name, filename: os.path.join(target_dir,
                                                                   os.path.split(dir_name)[-1],
                                                                   filename))

  print("Defining data generators...")

  batch_size = 16

  train_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=360,
        # channel_shift_range=0,
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1./255)

  test_datagen = ImageDataGenerator(rescale=1./255)

  train_generator = train_datagen.flow_from_directory(
        'data',  # this is the target directory
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

  # this is a similar generator, for validation data
  validation_generator = test_datagen.flow_from_directory(
        'validation',
        batch_size=batch_size,
        class_mode='binary')

  print("Defining model...")

  model = Sequential()
  model.add(Conv2D(32, (3, 3), input_shape=(3, TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZEs)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(32, (3, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(64, (3, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
  model.add(Dense(64))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(1))
  model.add(Activation('sigmoid'))

  model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

  print("Fitting model...")

  model.fit_generator(
          train_generator,
          steps_per_epoch=2000 // batch_size,
          epochs=50,
          validation_data=validation_generator,
          validation_steps=800 // batch_size)

  model.save_weights('first_try.h5')  # always save your weights after training or during training




if __name__ == "__main__":
  try:
    main()
  except UserRecoverableException as e:
    print(e)
    exit(1)
