'''
A CNN model with an identity or projection residual module is used to train a
model on different faces. This python file will train a binary prediction
model on my face vs. any other person's face. This python file reads in images
from an inputted data set, trains a model, and uses different metrics
to automatically tune the hyperparameters.
'''
# Resnet test Building Libraries
from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from keras.layers import add
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import numpy as np
import random
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold

# Data Visualization Libraries
import matplotlib.pyplot as plt


# function for creating an identity or projection residual module
def residual_module(layer_in, n_filters):
    '''
    Using the resource provided by
    https://www.pyimagesearch.com/2020/04/27/fine-tuning-resnet-with-
    keras-tensorflow-and-deep-learning/,
    I implemented the resnet model define here. I then
    changed some of the activation and kernel_initializer parameters
    to bette fit my desired data.

    **Parameters**

        layer_in: *Keras Tensor Object*
            The instantiated Keras tensor. One can specify the shape,
            batch size, and other parameters as an input into the model.
        n_filters: *int*
            The number of filters within every layer except the first.

    **Returns**

        layer_out: *Keras Tensor Object*
            The output layer of the model.
    '''

    merge_input = layer_in
    # check if the number of filters needs to be increase,
    # assumes channels last format.
    if layer_in.shape[-1] != n_filters:
        merge_input = Conv2D(n_filters, (1, 1), padding='same',
                             activation='relu',
                             kernel_initializer='he_normal')(layer_in)
    # conv1
    conv1 = Conv2D(n_filters, (3, 3), padding='same', activation='relu',
                   kernel_initializer='he_normal')(layer_in)
    # conv2
    conv2 = Conv2D(n_filters, (3, 3), padding='same', activation='linear',
                   kernel_initializer='he_normal')(conv1)
    # add filters, assumes filters/channels last
    layer_out = add([conv2, merge_input])
    # activation function
    layer_out = Activation('relu')(layer_out)
    return layer_out


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
                                           target_size=(224, 224),
                                           shuffle=True)
    # Check class names
    print(traindata.class_indices)
    # load the ResNet-50 network, ensuring the head FC layer sets are left off
    print("[INFO] preparing model...")
    baseModel = ResNet50(weights="imagenet", include_top=False,
                         input_tensor=Input(shape=(224, 224, 3)))
    # construct the head of the model that will be placed on top of the
    # the base model
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(256, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(2, activation="softmax")(headModel)
    # place the head FC model on top of the base model (this will become
    # the actual model we will train)
    model = Model(inputs=baseModel.input, outputs=headModel)
    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the training process
    for layer in baseModel.layers:
        layer.trainable = False
    # summarize model
    model.summary()
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    # Save model with highest validation accuracy.
    mca = ModelCheckpoint("vallaccFace.h5", monitor='val_accuracy',
                          verbose=1, save_best_only=True,
                          save_weights_only=False, mode='auto')
    #  Save model with the lowest validation loss.
    mcl = ModelCheckpoint("vallossFace.h5", monitor='val_loss',
                          verbose=1, save_best_only=True,
                          save_weights_only=False, mode='auto')
    # Stop training after validation loss stops improving enough.
    es = EarlyStopping(monitor='val_loss', min_delta=5e-4,
                       patience=10, verbose=1, mode='auto')
    cb_list = [es, mca, mcl]
    x, y = traindata.next()
    kfold = KFold(n_splits=5, shuffle=True)
    fold_no = 1
    for train, test in kfold.split(x, y):
        model_history = model.fit(x[train], y[train], validation_split=0.3,
                                  epochs=100, batch_size=32,
                                  callbacks=cb_list, shuffle=True)
        fold_no += 1
    model.save("FinalFaceModel.h5")
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
