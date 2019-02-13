# import the necessary packages
from rmac import describe
import argparse
import glob
import cv2
import progressbar
import numpy as np
import pickle
from keras.preprocessing import image


# initialize the color descriptor
descriptors = {}
desc = []

bar = progressbar.ProgressBar(max_value=13).start()

i =0
# use glob to grab the image paths and loop over them
for imagePath in glob.glob('app/static/mini_data/' + "*.jpg"):
    # extract the image ID (i.e. the unique filename) from the image
    # path and load the image itself
    imageID = imagePath[imagePath.rfind("/") + 1:]
    img = image.load_img(imagePath)
    # describe the image
    features =describe(img)
    #add the list of descriptors to
    descriptors[imageID] = features
    i += 1
    bar.update(i)

with open('app/index', 'wb') as handle:
  pickle.dump(descriptors, handle)


bar.finish()


