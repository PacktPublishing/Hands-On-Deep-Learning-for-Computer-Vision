"""
Simple YOLO API with a bit more control over results.

Before you can use this code clone or download keras-yolo3
github repo from https://github.com/qqwweee/keras-yolo3
and set it up according to Quickstart section of the project.

Copy this file inside the project's folder (this is where
you can find yolo.py file and the rest of the source code)
"""
from yolo import YOLO
from PIL import Image
from sys import argv
import numpy as np
from yolo3.utils import letterbox_image
from keras import backend as K

class YOLOSimple(YOLO):
    """
    This class breaks down the original
    detect_image method into three smaller methods:
    prepare_image, detect and print_detected.
    The goal is to have better access to detected results.

    Keep in mind that we're igoring the last part of detect_image
    when the bounding boxes are generate for and in a output image
    and showed up. But, you can always have look at the original
    method to see how to do that.
    """
    def prepare_image(self, image):
        image=Image.open(image)
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        return image_data, image

    def detect(self, image_data, image):
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        return out_boxes, out_scores, out_classes

    def print_detected(self, out_boxes, out_scores, out_classes):
        out=[]
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}%'.format(predicted_class, score*100)
            print(label, box)
