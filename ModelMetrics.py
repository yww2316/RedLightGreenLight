'''
This Python file is used to measure the metrics of the trained
models for the purposes of hyperparameter tuning.
'''

from keras.models import load_model

if __name__ == "__main__":
    saved_model = load_model("vallaccFace.h5")
    saved_loss_model = load_model("vallossFace.h5")

    # Uncomment the line below if you're only interested
    # in making predictions with previously trained models.
    model = load_model("FinalFaceModel.h5")
