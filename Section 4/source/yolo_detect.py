"""
Perform object detection using YOLO v3
on an image, print each detected object -
its class, probability score and bounding box.
"""
from yolo_simple import YOLOSimple
# Turn off Tensorflow debug info
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sys import argv

if __name__ == '__main__':
    # Create a new yolo object
    myyo=YOLOSimple()

    # Prepare an image to match the model.
    img = myyo.prepare_image(argv[1])

    # Detect objects.
    boxes, scores, classes = myyo.detect(*img)

    # Show results.
    myyo.print_detected(boxes, scores, classes)

    # Clean up the session afeter we've done.
    myyo.close_session()
