'''
Python script to test prediction capabilities of the model
on a live camera.
'''

import numpy as np
import random
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

if __name__ == "__main__":
    # define a set "random" number generator and set it
    seed_constant = 42
    np.random.seed(seed_constant)
    random.seed(seed_constant)
    tf.random.set_seed(seed_constant)

    # Define an ImageDataGenerator object and specify what
    # parameters to randomize each new epoch.
    # Lowers or increases brightness by up to 50%
    trdata = ImageDataGenerator(brightness_range=[0.5, 1.5],
                                shear_range=0.2,  # shears the image up to 20%
                                zoom_range=0.2)  # zooms in or out up to 20%
    # width_shift_range=0.2,#shift the width up to 20%
    # height_shift_range=0.2,#shift the height up to 20%
    # rotation_range=180,#Rotate the image up to 270 degrees
    # horizontal_flip=True,#May flip the image horizontally
    # vertical_flip=True)#May flip the image vertically

    traindata = trdata.flow_from_directory(directory='PicturesofFaces',
                                           target_size=(256, 256),
                                           shuffle=True)
    saved_model = load_model("vallaccFace.h5")
    saved_loss_model = load_model("vallossFace.h5")

    # Uncomment the line below if you're only interested
    # in making predictions with previously trained models.
    model = load_model("FinalFaceModel.h5")