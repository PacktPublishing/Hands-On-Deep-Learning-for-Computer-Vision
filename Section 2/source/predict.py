# To run this script
# install the newest PIL library with:
# conda install pillow
# We can't use a standard PIL library
# since it's in conflict with one of the
# matplotlib's dependencies that we've used in Section 1
#
# Turn off Tensorflow debug info
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Import the necessary tools
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
# We have two models available:
# VGG16 and VGG19 - 16 and 19 is the number of layers
# in the model.
# More layers = bigger model = more memory required
# To use vgg16 just change vgg19->vgg16
from keras.applications.vgg19 import preprocess_input
from keras.applications.vgg19 import decode_predictions
from keras.applications.vgg19 import VGG19
from sys import argv
from pprint import pprint

# load the model
model = VGG19()

print(argv[1])

# Load image from the first script's argument.
# Resise to fit model's required input.
image = load_img(argv[1], target_size=(224, 224))

# Convert image to numpy array.
image = img_to_array(image)

# Reshape image to fit the model's requirments.
# Fist argument is the number of images we plan
# to classify using the model.
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

# Prepare the image in the same way
# that images that the model was trained on.
image = preprocess_input(image)

# Get the probablities for each class.
predictions = model.predict(image)

# Convert them to human readable labels.
labels = decode_predictions(predictions)

pprint(labels)
# Get the class that is the best match
# (has the highest probablity)
label = labels[0][0]

# Show the most probable class and it's probablity percentage.
print('%s: %.2f%%' % (label[1], label[2]*100))
