'''
Python script to test prediction capabilities of the model
on a live camera.
'''
from keras.models import load_model
import cv2
from collections import deque
import numpy as np
import random
import turtle
# Define a function for attaching and saving predictions based off of a video.


def RedLightGreenLight(model, video_file_path, output_file_path,
                       window_size, threshold_set, penalty_diff):
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
        penalty_diff: *int
            How many instances one can move forward before losing the game.

    **Returns**

        None

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
    if video_file_path == 0:
        fps = video_reader.get(cv2.CAP_PROP_FPS)
        print(fps)
    # Writing the Overlayed Video Files Using the VideoWriter Object
    video_writer = cv2.VideoWriter(output_file_path,
                                   cv2.VideoWriter_fourcc('M', 'P', '4', 'V'),
                                   10, (original_video_width,
                                        original_video_height))
    total_time = 0
    turtle.setup(500, 300)
    wn = turtle.Screen()
    wn.bgcolor("lightgreen")
    wn.title("RedLightGreenLight")
    tess = turtle.Turtle()       # Create tess and set some attributes
    tess.penup()
    tess.setpos(-250, 0)
    tess.pendown()
    tess.shape('turtle')
    tess.color("hotpink")
    tess.pensize(5)
    initial_green = 2
    green_flag = 1
    penalty = 0
    while True:
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
                predicted_class_name = 'Jason'
                tess.forward(10)
                if green_flag == 0:
                    penalty += 1
            else:
                predicted_class_name = 'Other'

            # Overlaying class name text on top of the Frame, changing color to
            # reflect whether Jason is shown or not
            if predicted_class_name == 'Jason':
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
        total_time += 0.3
        if total_time > initial_green:
            if green_flag == 1:
                total_time = 0
                wn.bgcolor("red")
                initial_green = random.randrange(2, 5)
                green_flag = 0
            else:
                total_time = 0
                wn.bgcolor("lightgreen")
                initial_green = random.randrange(2, 5)
                green_flag = 1
        position = tess.position()
        if position[0] > 250:
            turtle.write("You win!", False, align='center',
                         font=('Arial', 40, 'normal'))
            break
        if penalty > penalty_diff:
            turtle.write("You lose!", False, align='center',
                         font=('Arial', 40, 'normal'))
            break
        # Writing The Frame
        video_writer.write(frame)
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
    # saved_model = load_model("vallaccFace.h5")
    # saved_loss_model = load_model("vallossFace.h5")

    # Uncomment the line below if you're only interested
    # in making predictions with previously trained models.
    model = load_model("Model/vallossFace.h5")
    # threshold = F1_threshold('PicturesofFaces', model)
    # print(threshold)

    output_directory = 'ClassifiedVideo'
    video_title = 'Live_Video'
    window_size = 1
    threshold_set = .54039
    Difficulty = 5
    # Set input_video_file_path to 0 to use webcam
    input_video_file_path = 0
    output_video_file_path = f'{output_directory}/{video_title}\
        {window_size}.mp4'
    RedLightGreenLight(model, input_video_file_path,
                       output_video_file_path,
                       window_size, threshold_set, Difficulty)
