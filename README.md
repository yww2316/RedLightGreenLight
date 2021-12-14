# SoftwareCarpentryFinalProject

I have developed a computer vision project where the user can play RedLightGreenLight. This requires use of your computer's native webcam.
To play the game, download this repository and download the Model folder from the onedrive link below into the repository. Then, run the RedLightGreenLight.py file. threshold_set, the threshold by which predictions are made, may need to be changed based of off your hardware and the lighting of your area.

For the game, you can set your own difficulty under the Difficulty parameter, where 1 is very hard and 10 is very easy. To go forward, put your face in front of the camera. To stop, either cover your face or cover the camera. You may have to adjust the threshold_set parameter based off of your own camera.
To train your own model, download the PictureofFaces folder into the repository and change what data is inside the subfolders. The name of the classes will correspond to the names of the folders within the subdirectories of the PictureofFaces folder.

Note: You will have to have tensorflow and playsound installed to run these python files. To do so, run this in the terminal:

```
pip install --ignore-installed --upgrade tensorflow
pip install playsound
```

Access the Model Folder through
https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/jwu130_jh_edu/EgKg9PANR9NPkpw-J_FJf7cBr2E4h9n7EHZkrMe3ZnGbxQ?e=JvPwJJ

There is also a LiveFaceClassification.py file. This is used to do just video classification, either with a recorded video, computer webcam, or a phone camera. One can change the video source through the input_video_file_path variable in this python file. Note that the computer webcam functionality requires a webcam on your computer, and the phone camera functionality requires downloading IPWebcam onto a compatible device.

Download IPWebcam through
https://play.google.com/store/apps/details?id=com.pas.webcam&hl=en_US&gl=US

I trained the Keras model with the FaceModel.py script. Below is the data used for this training.

Access the data through
https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/jwu130_jh_edu/EmafTPbSDRtAiu2GmK5XcyIB_yq2u0qEx-XdrC_iMKy9aA?e=lCTWgQ

FinalUnittest.py includes the unittests used for testing of these python files.



Inspirations for parts of the code are linked below:

Resnet Model Architecture in Keras: https://towardsdatascience.com/building-a-resnet-in-keras-e8f1322a49ba
Video Classification from a youtube video: https://learnopencv.com/introduction-to-video-classification-and-human-activity-recognition/

