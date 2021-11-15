# import unittest
from FaceModel import *
from RedLightGreenLight import *
from LiveFaceClassification import *
import unittest
import tensorflow as tf
import os


class bffTest(unittest.TestCase):
    '''
    Unit test for some of the different parts of the repository.
    '''
    def setUp(self):
        self.Tensor = tf.keras.layers.Input([10])
        self.model = load_model("Model/vallossFace.h5")
        self.video_file_path = 'wow.mp4'
        self.output_file_path = 'shakes.mp4'
        self.window_size = 25
        self.threshold_set = .5
        self.penalty_diff = 1
        self.url = "http://10.203.175.80:8080/shot.jpg"

    def testcheck_plain_net(self):
        '''
        Check to see if create_plain_net throws any errors while producing a
        model.
        '''
        self.assertTrue(create_plain_net())

    def testrelu_bn(self):
        '''
        Checks to see if relu_bn outputs a keras tensor.
        '''
        self.assertTrue(tf.keras.backend.is_keras_tensor(relu_bn(self.Tensor)))

    def testRedLightGreenLight(self):
        '''
        Checks to see if RedLightGreenLight runs, doesn't throw any errors,
        and returns nothing.
        '''
        self.assertEqual(RedLightGreenLight(self.model, self.video_file_path,
                         self.output_file_path,
                         self.window_size, self.threshold_set,
                         self.penalty_diff), None)
        os.remove(self.output_file_path)

    def testpredict_on_live_video(self):
        '''
        Checks to see if predict_on_live_video runs, doesn't throw any errors,
        and returns nothing.
        '''
        self.assertEqual(predict_on_live_video(self.model,
                         self.video_file_path,
                         self.output_file_path,
                         self.window_size, self.threshold_set,
                         self.penalty_diff, self.url), None)
        os.remove(self.output_file_path)


if __name__ == "__main__":
    unittest.main()
