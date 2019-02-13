from __future__ import division
from __future__ import print_function

from keras.layers import *
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.applications import VGG16
import progressbar


#from vgg16 import VGG16
from RoiPooling import RoiPooling
from get_regions import rmac_regions, get_size_vgg_feat_map

import scipy.io
import numpy as np
import utils
K.set_image_dim_ordering('th')

from sklearn.metrics.pairwise import cosine_similarity



def addition(x):
    sum = K.sum(x, axis=1)
    return sum


def weighting(input):
    x = input[0]
    w = input[1]
    w = K.repeat_elements(w, 512, axis=-1)
    out = x * w
    return out


def rmac(input_shape, num_rois):

    # Load VGG16
    vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    #vgg16_model = VGG16(utils.DATA_DIR + utils.WEIGHTS_FILE, input_shape)

    # Freeze the layers except the last 4 layers
    for layer in vgg_conv.layers[:-4]:
        layer.trainable = False

    # Check the trainable status of the individual layers
    #for layer in vgg_conv.layers:
        #print(layer, layer.trainable)

    # Regions as input
    in_roi = Input(shape=(num_rois, 4), name='input_roi')

    # ROI pooling
    x = RoiPooling([1], num_rois)([vgg_conv.layers[-5].output, in_roi])

    # Normalization
    x = Lambda(lambda x: K.l2_normalize(x, axis=2), name='norm1')(x)

    # PCA
    x = TimeDistributed(Dense(512, name='pca',
                              kernel_initializer='identity',
                              bias_initializer='zeros'))(x)

    # Normalization
    x = Lambda(lambda x: K.l2_normalize(x, axis=2), name='pca_norm')(x)

    # Addition
    rmac = Lambda(addition, output_shape=(512,), name='rmac')(x)

    # # Normalization
    rmac_norm = Lambda(lambda x: K.l2_normalize(x, axis=1), name='rmac_norm')(rmac)

    # Define model
    model = Model([vgg_conv.input, in_roi], rmac_norm)

    # Load PCA weights
    mat = scipy.io.loadmat(utils.DATA_DIR + utils.PCA_FILE)
    b = np.squeeze(mat['bias'], axis=1)
    w = np.transpose(mat['weights'])
    model.layers[-4].set_weights([w, b])

    return model

def matcher(des1, des2):
    return cosine_similarity(des1, des2)[0]


def describe(img):
    K.clear_session()
    # Resize
    scale = utils.IMG_SIZE / max(img.size)
    new_size = (int(np.ceil(scale * img.size[0])), int(np.ceil(scale * img.size[1])))
    img = img.resize(new_size)
    # Mean substraction
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = utils.preprocess_image(x)
    # Load RMAC model
    Wmap, Hmap = get_size_vgg_feat_map(x.shape[3], x.shape[2])
    regions = rmac_regions(Wmap, Hmap, 3)
    model = rmac((x.shape[1], x.shape[2], x.shape[3]), len(regions))
    # Compute RMAC vector
    RMAC = model.predict([x, np.expand_dims(regions, axis=0)])
    return RMAC

def Searcher(index, img_desc, limit = 3):
    # initialize our dictionary of results
    results = {}
   # bar = progressbar.ProgressBar(max_value=812).start()
    i = 0
    # loop over the rows in the index
    for id,image in index.items():
        # parse out the image ID and features, then compute the
        # chi-squared distance between the features in our index
        # and our query features

        dist = matcher(image, img_desc)

        # now that we have the distance
        # we can udpate the results dictionary -- the
        # key is the current image ID in the index and the
        # value is the distance we just computed, representing
        # how 'similar' the image in the index is to our query
        results[id] = dist[0]

    # sort our results, so that the bigger distances (i.e. the
    # more relevant images are at the end of the list)
    results = sorted([(v, k) for (k, v) in results.items()])
    # return our (limited) results
    return results[limit:]


#if __name__ == "__main__":

 #   # Load sample image
  #  file = utils.DATA_DIR + 'sample.jpg'
   # img = image.load_img(file)

    ## Resize
   # scale = utils.IMG_SIZE / max(img.size)
   # new_size = (int(np.ceil(scale * img.size[0])), int(np.ceil(scale * img.size[1])))
   # print('Original size: %s, Resized image: %s' %(str(img.size), str(new_size)))
   # img = img.resize(new_size)

    # Mean substraction
   # x = image.img_to_array(img)
   # x = np.expand_dims(x, axis=0)
    #x = utils.preprocess_image(x)

    # Load RMAC model
  #  Wmap, Hmap = get_size_vgg_feat_map(x.shape[3], x.shape[2])
  #  regions = rmac_regions(Wmap, Hmap, 3)
   # print('Loading RMAC model...')
   # model = rmac((x.shape[1], x.shape[2], x.shape[3]), len(regions))

    # Compute RMAC vector
   # print('Extracting RMAC from image...')
   # RMAC = model.predict([x, np.expand_dims(regions, axis=0)])
  #  print('RMAC size: %s' % RMAC.shape[1])
  #  print('Done!')
