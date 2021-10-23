'''
This Python file is used to measure the metrics of the trained
models for the purposes of hyperparameter tuning.
'''

from keras.models import load_model
from numpy import arange
from numpy import argmax
from sklearn.metrics import f1_score
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import label_binarize


def F1_threshold(directory, model):
    '''
    Calculates the F1 score to identify the optimal threshold by
    which to make predictions with.

    **Parameters:
        directory: *string
            The directory path where the data is stored.
        model: A Keras model instance
            The loaded model which we use to make predictions.
    '''
    trdata = ImageDataGenerator(brightness_range=[0.5, 1.5],
                                shear_range=0.2,  # shears the image up to 20%
                                zoom_range=0.2)  # zooms in or out up to 20%
    traindata = trdata.flow_from_directory(directory=directory,
                                           target_size=(224, 224),
                                           shuffle=True)
    # predict probabilities
    testX, testy = next(traindata)
    yhat = model.predict(testX)
    # keep probabilities for the positive outcome only
    yhat = yhat[:, 1]
    # calculate roc curves
    Y = label_binarize(testy, classes=[0, 1])
    Y = Y[:, 1]
    # search thresholds for imbalanced classification
    # apply threshold to positive probabilities to create labels

    def to_labels(pos_probs, threshold):
        '''
        Maps the positive portion of pos_probs to int
        for the purpose of creating labels.

        **Parameters**
            pos_probs: *Numpy Array
                Returns only the probability of the image being
                another person.
            threshold: *int or real
                The threshold from which we compare against.
        **Returns**
            *numpy array
                Returns a list of predictions by which the F1-score
                is calculated.
        '''
        return (pos_probs >= threshold).astype('int')

    # keep probabilities for the positive outcome only
    # define thresholds
    thresholds = arange(0, 1, 0.001)
    # evaluate each threshold
    scores = [f1_score(Y, to_labels(yhat, t)) for t in thresholds]
    # get best threshold
    ix = argmax(scores)
    # print('Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))
    threshold = thresholds[ix]
    return threshold


if __name__ == "__main__":
    # saved_model = load_model("Model/vallaccFace.h5")
    # saved_loss_model = load_model("Model/vallossFace.h5")

    # Uncomment the line below if you're only interested
    # in making predictions with previously trained models.
    model = load_model("Model/FinalFaceModel.h5")
    threshold = F1_threshold('PicturesofFaces', model)
    print(threshold)
