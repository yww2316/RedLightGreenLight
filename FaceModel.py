'''
A CNN model with an identity or projection residual module is used to train a
model on different faces. This python file will train a binary prediction
model on my face vs. any other person's face. This python file reads in images
from an inputted data set, trains a model, and uses different metrics
to automatically tune the hyperparameters.
'''
# Resnet test Building Libraries
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from keras.layers import add
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import tensorflow as tf
import numpy as np
import random
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold

# Data Visualization Libraries
import matplotlib.pyplot as plt


def relu_bn(inputs):
    '''
    Runs a tensor through relu and normalizes them.

    **Parameters**
        inputs: A keras tensor instance
            This is the input tensor that will be normalized.

    **Returns**
        bn: A keras tensor instance
            This is the output tensor that has been normalized.
    '''
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn


def create_plain_net():
    '''
    Creates a resnet model that has over 15 million trainable
    parameters. This uses Adam as the optimizer and binary_crossentropy
    as the loss function. Of note, the input shape is customizable and the
    integer parameter of the final dense layer corresponds to the number of
    desired classes.

    **Parameters**
        None

    **Returns**
        model: A keras model instance
            This is the compiled resnet model.
    '''
    inputs = Input(shape=(224, 224, 3))
    num_filters = 64

    t = BatchNormalization()(inputs)
    t = Conv2D(kernel_size=3,
               strides=1,
               filters=num_filters,
               padding="same")(t)
    t = relu_bn(t)

    num_blocks_list = [4, 10, 10, 4]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            downsample = (j == 0 and i != 0)
            t = Conv2D(kernel_size=3,
                       strides=(1 if not downsample else 2),
                       filters=num_filters,
                       padding="same")(t)
            t = relu_bn(t)
        num_filters *= 2
    t = AveragePooling2D(4)(t)
    t = Flatten()(t)
    outputs = Dense(2, activation='sigmoid')(t)
    model = Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


if __name__ == "__main__":
    # define a set "random" number generator and set it
    seed_constant = 42
    np.random.seed(seed_constant)
    random.seed(seed_constant)
    tf.random.set_seed(seed_constant)

    # Define an ImageDataGenerator object and specify what
    # parameters to randomize each new epoch.
    trdata = ImageDataGenerator(brightness_range=[0.5, 1.5],  # Brightness
                                shear_range=0.2,  # shears the image up to 20%
                                zoom_range=0.2)  # zooms in or out up to 20%
    # width_shift_range=0.2,#shift the width up to 20%
    # height_shift_range=0.2,#shift the height up to 20%
    # rotation_range=180,#Rotate the image up to 270 degrees
    # horizontal_flip=True,#May flip the image horizontally
    # vertical_flip=True)#May flip the image vertically

    traindata = trdata.flow_from_directory(directory='PicturesofFaces',
                                           target_size=(224, 224),
                                           shuffle=True)
    # Check class names
    print(traindata.class_indices)
    # load the ResNet-50 network, ensuring the head FC layer sets are left off
    print("[INFO] preparing model...")
    # summarize model
    model = create_plain_net()
    model.summary()
    # Save model with highest validation accuracy.
    mca = ModelCheckpoint("Model/vallaccFace.h5", monitor='val_accuracy',
                          verbose=1, save_best_only=True,
                          save_weights_only=False, mode='auto')
    #  Save model with the lowest validation loss.
    mcl = ModelCheckpoint("Model/vallossFace.h5", monitor='val_loss',
                          verbose=1, save_best_only=True,
                          save_weights_only=False, mode='auto')
    # Stop training after validation loss stops improving enough.
    es = EarlyStopping(monitor='val_loss', min_delta=5e-4,
                       patience=10, verbose=1, mode='auto')
    cb_list = [es, mca, mcl]
    x, y = traindata.next()
    kfold = KFold(n_splits=5, shuffle=True)
    for train, test in kfold.split(x, y):
        model_history = model.fit(x[train], y[train], validation_split=0.3,
                                  epochs=50, batch_size=32,
                                  callbacks=cb_list, shuffle=True)
    model.save("Model/FinalFaceModel.h5")
    plt.plot(model_history.history["accuracy"])
    plt.plot(model_history.history['val_accuracy'])
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title("model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Accuracy", "Validation Accuracy", "loss", "Validation Loss"])
    plt.show()
    plt.savefig('model_history.png')
