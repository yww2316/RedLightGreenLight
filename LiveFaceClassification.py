'''
Python script to test prediction capabilities of the model
on a live camera.
'''
from keras.models import load_model
import cv2
from collections import deque
import numpy as np
import requests
import imutils

# Define a function for attaching and saving predictions based off of a video.


def predict_on_live_video(model, video_file_path, output_file_path,
                          window_size, threshold_set, timespan, url):
    '''
    Using the computer's default camera, the streamed video
    will have predictions made on it until the Escape key is pressed.

    **Parameters:
        model: A keras model instance
            This is the model that will be generating predictions
        video_file_path: *str, *int
            The desired video file path. This should be 0
            for accessing the webcam, or a directory if
            predicting on a recorded video.
        output_file_path: *str
            The desired place for the recording of the predictions
            to be saved.
        window_size: *int
            This is the parameter for how many frames should have
            their predictions averaged for a single prediction.
            This is a moving average so only the previous window_size
            frames, including the present frame, are considered.
        threshold_set: *int
            The threshold by which a prediction is made.
        timespan: *float
            This is how long the prediction should last before
            automatically terminating.
        url: *string
            The url for the IPWebcam video on your android device.
    '''

    # Initialize a Deque Object with a fixed size which will be used to
    # implement moving/rolling average functionality.
    predicted_labels_probabilities_deque = deque(maxlen=window_size)
    color = []
    predicted_class_name = 'Loading...'
    image_height, image_width = 224, 224
    # Reading the Video File using the VideoCapture Object
    video_reader = cv2.VideoCapture(video_file_path)

    # Getting the width and height of the video
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Writing the Overlayed Video Files Using the VideoWriter Object
    video_writer = cv2.VideoWriter(output_file_path,
                                   cv2.VideoWriter_fourcc('M', 'P', '4', 'V'),
                                   10, (original_video_width,
                                        original_video_height))
    total_time = 0
    while True:
        if video_file_path == 1:
            # Get The Frame
            img_resp = requests.get(url)
            img_arr = np.array(bytearray(img_resp.content))
            img = cv2.imdecode(img_arr, -1)
            frame = imutils.resize(img, width=1920, height=1080)
        else:
            # Reading The Frame
            status, frame = video_reader.read()
            if not status:
                break

        # Resize the Frame to fixed Dimensions
        resized_frame = cv2.resize(frame, (image_height, image_width))
        # Normalize the resized frame by dividing it with 255 so that each
        # pixel value then lies between 0 and 1
        normalized_frame = resized_frame / 255
        # Passing the Image Normalized Frame to the model and
        # receiving predicted Probabilities.
        predicted_labels_probabilities = model.predict(np.expand_dims(
            normalized_frame, axis=0))

        # Appending predicted label probabilities to the deque object
        predicted_labels_probabilities_deque.append(
            predicted_labels_probabilities)

        # Assuring that the Deque is completely filled before starting the
        # averaging process
        if len(predicted_labels_probabilities_deque) == window_size:

            # Converting Predicted Labels Probabilities Deque into Numpy array
            predicted_labels_probabilities_np = np.array(
                predicted_labels_probabilities_deque)

            # Calculating Average of Predicted Labels Probabilities Column Wise
            predicted_labels_probabilities_averaged = \
                predicted_labels_probabilities_np.mean(axis=0)

            # Accessing The Class Name using predicted label.
            if predicted_labels_probabilities_averaged[0][1] > threshold_set:
                predicted_class_name = 'Human'
            else:
                predicted_class_name = 'Other'

            # Overlaying Class Name Text On top of the frame, changing color to
            # reflect whether the bottle is defective or not
            if predicted_class_name == 'Human':
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            if predicted_labels_probabilities[0][1] > threshold_set:
                color1 = (0, 255, 0)
            else:
                color1 = (0, 0, 255)

            cv2.putText(frame, predicted_class_name, (200, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, color, 2)
            cv2.putText(frame,  str(predicted_labels_probabilities),
                        (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, color1, 2)
        if video_file_path == 0 or video_file_path == 1:
            total_time += 0.1
        # Writing The Frame
        video_writer.write(frame)
        if total_time > timespan:
            break
        cv2.imshow('Predicted Frames', frame)
        # Press Escape to exit the program
        key_pressed = cv2.waitKey(1)

        if key_pressed == 27:
            break

    cv2.destroyAllWindows()

    # Closing the VideoCapture and VideoWriter objects and
    # releasing all resources held by them.
    video_reader.release()
    video_writer.release()


if __name__ == "__main__":
    # Replace the below URL with your own. Make sure to add "/shot.jpg"
    # at the end. Note that this requires the download of IPWebcam onto a
    # compatible device.
    # You can download IPWebcam at
    # https://play.google.com/store/apps/details?id=com.pas.webcam&hl=en_US&gl=US
    url = "http://10.203.175.80:8080/shot.jpg"

    # Load the desired model here
    model = load_model("Model/vallossFace.h5")
    output_directory = 'ClassifiedVideo'
    video_title = 'Live_Video'
    window_size = 1
    threshold_set = .54039
    # Set input_video_file_path to 0 to use webcam or 1 to use phone
    # camera
    input_video_file_path = 0
    output_video_file_path = f'{output_directory}/{video_title}\
        {window_size}.mp4'
    predict_on_live_video(model, input_video_file_path,
                          output_video_file_path,
                          window_size, threshold_set, 15, url)
