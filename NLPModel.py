'''
A CNN model with an identity or projection residual module is used to train a model on [].
This python file reads in images from an inputted data set, trains a model, and uses different metrics
to automatically tune the hyperparameters.
'''
from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import add
import numpy as np
import random
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

 
# function for creating an identity or projection residual module
def residual_module(layer_in, n_filters):
	'''
	Using the resource provided by 
	https://machinelearningmastery.com/how-to-implement-major-architecture-innovations-for-convolutional-neural-networks/, 
	I implemented the resnet model define here. I then changed some of the activation and kernel_initializer parameters
	to bette fit my desired data.

	**Parameters**

		layer_in: *Keras Tensor Object*
			The instantiated Keras tensor. One can specify the shape, batch size, and other parameters
			as an input into the model.
		n_filters: *int*
			The number of filters within every layer except the first.

	**Returns**

		layer_out: *Keras Tensor Object*
			The output layer of the model.
	'''
	merge_input = layer_in
	# check if the number of filters needs to be increase, assumes channels last format
	if layer_in.shape[-1] != n_filters:
		merge_input = Conv2D(n_filters, (1,1), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
	# conv1
	conv1 = Conv2D(n_filters, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
	# conv2
	conv2 = Conv2D(n_filters, (3,3), padding='same', activation='linear', kernel_initializer='he_normal')(conv1)
	# add filters, assumes filters/channels last
	layer_out = add([conv2, merge_input])
	# activation function
	layer_out = Activation('relu')(layer_out)
	return layer_out
if __name__ == "__main__":
	#define a set "random" number generator and set it
	seed_constant = 42
	np.random.seed(seed_constant)
	random.seed(seed_constant)
	tf.random.set_seed(seed_constant)

	#Define an ImageDataGenerator object and specify what parameters to randomize each new epoch. 
	trdata = ImageDataGenerator(brightness_range=[0.5,1.5], #Lowers or increases brightness by up to 50%
    shear_range=0.2,#shears the image up to 20%
    zoom_range=0.2)#,zooms in or out up to 20%
    #width_shift_range=0.2,#shift the width up to 20%
    #height_shift_range=0.2,#shift the height up to 20%
    #rotation_range=180,#Rotate the image up to 270 degrees
    #horizontal_flip=True,#May flip the image horizontally
    #vertical_flip=True)#May flip the image vertically
	traindata = trdata.flow_from_directory(directory='NewMcCormickBottles',target_size=(256,256),shuffle=True) 
	# define model input
	visible = Input(shape=(256, 256, 3))
	# add vgg module
	layer = residual_module(visible, 64)
	# create model
	model = Model(inputs=visible, outputs=layer)
	# summarize model
	model.summary()