# SoftwareCarpentryFinalProject

I have developed a computer vision project where the user can play RedLightGreenLight. This requires use of your native webcam.
To play the game, download this repository and download the Model folder into the repository. Then, run the RedLightGreenLight.py file.
For the game, you can set your own difficulty under the Difficulty parameter, where 1 is very hard and 5 is easy. To go forward, put your face in front of the camera. To stop, either cover your face or cover the camera. You may have to adjust the threshold_set parameter based off of your own camera.
To train your own model, download the PictureofFaces folder into the repository and change what data is inside the subfolders. The name of the classes will correspond to the names of the folders within the subdirectories of the PictureofFaces folder.

Access the data through
https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/jwu130_jh_edu/EmafTPbSDRtAiu2GmK5XcyIB_yq2u0qEx-XdrC_iMKy9aA?e=lCTWgQ

Access the trained model through
https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/jwu130_jh_edu/EgKg9PANR9NPkpw-J_FJf7cBr2E4h9n7EHZkrMe3ZnGbxQ?e=JvPwJJ

Note: You will have to have tensor flow installed. To do so, run this in the terminal:

```
pip install --ignore-installed --upgrade tensorflow
```

Inspirations for parts of the code are linked below:

Resnet Model Architecture in Keras: https://towardsdatascience.com/building-a-resnet-in-keras-e8f1322a49ba
Video Classification from a youtube video: https://learnopencv.com/introduction-to-video-classification-and-human-activity-recognition/

