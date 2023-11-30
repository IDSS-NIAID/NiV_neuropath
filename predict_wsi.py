import os, sys, shutil, glob, random

import numpy as np

import tensorflow as tf

from tensorflow.keras import backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Conv3DTranspose
from tensorflow.keras.layers import Dropout, SeparableConv2D, Cropping2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import Sequence

import PIL
from PIL import Image, ImageEnhance

from skimage.io import imread, imsave

Image.MAX_IMAGE_PIXELS = 200000*200000

import warnings
warnings.filterwarnings("ignore")

import time

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

NUM_GPUS = 1
BATCH_SIZE = 384 * NUM_GPUS

PATCH_SIZE = 256
PADDING = 32

"""
helper function computing dice coefficient score
"""
def f1(y_true_f, y_pred_f):
  intersection = K.sum(y_true_f * y_pred_f)
  return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

"""
compute dice coefficient score
"""
def d_coef(y_true, y_pred):
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)
  return f1(y_true_f, y_pred_f)

"""
loss function using negative dice coefficient score
"""
def d_loss(y_true, y_pred):
  return -d_coef(y_true, y_pred)

"""
UNet model definition
"""
def get_unet(lrate=1e-5):
  inputs = Input((PATCH_SIZE, PATCH_SIZE, 3))

  conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
  conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

  conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
  conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

  conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
  conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
  pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

  conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
  conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
  pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

  conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
  conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

  up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
  conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
  conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

  up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
  conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
  conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

  up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
  conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
  conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

  up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
  conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
  conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

  convLast = BatchNormalization()(conv9)

  conv10 = Conv2D(1, (1, 1), activation='sigmoid')(convLast)

  model = Model(inputs=[inputs], outputs=[conv10])

  model.compile(loss=d_loss,
                metrics=[d_coef],
                optimizer=Adam(lr=lrate))

  print(model.summary())
  return model

"""
predict a list of ROIs in a batch mode
"""
def batchPredict(img, msk, l, m):
  # break down the ROI list to trunks
  batches = [l[i:i+BATCH_SIZE] for i in range(0, len(l), BATCH_SIZE)]
  
  for b in batches:
    inputs = []
    
    for bb in b:
      patch = img[bb[1]:bb[1]+PATCH_SIZE, bb[0]:bb[0]+PATCH_SIZE]
      inputs.append( (patch/255.).astype(np.float32) )
  
    predicts = model.predict_on_batch( np.array(inputs) )
    
    for bb, pp in zip(b, predicts):
      roi = pp.reshape(PATCH_SIZE, PATCH_SIZE)[PADDING:-PADDING, PADDING:-PADDING]
      msk[bb[1]+PADDING:bb[1]+PATCH_SIZE-PADDING, bb[0]+PADDING:bb[0]+PATCH_SIZE-PADDING] += (roi*255).astype(np.uint32)
#end def batchPredict

#main program starts
if len(sys.argv) == 3:
  INPUT = sys.argv[1]
  OUTPUT = sys.argv[2]
else:
  print('need INPUT OUTPUT')
  sys.exit()
  
#prepare output folder
os.makedirs(OUTPUT, exist_ok=True)

WEIGHTS = glob.glob('weights/*.h5')

if len(WEIGHTS) == 0:
 print('did not find any trained weights, existing...')
 sys.exit()

INPUTS = glob.glob(os.path.join(INPUT, '*.svs'))

#with strategy.scope():
model = get_unet()
  
for w in INPUTS:
  wsi = openslide.OpenSlide(w)
  wsiNpy = np.array(wsi.read_region((0,0), 0, (wsi.dimensions[0], wsi.dimensions[1])))[:,:,:3]
  
  # prepare a blank mask image
  msk = np.ndarray((wsiNpy.shape[0], wsiNpy.shape[1]), dtype=np.uint32)
  msk.fill(0)
 
  # ROI coordinate list
  coords = []

  # save ROI coordinates as a list for batch prediction
  # ROIs extracted using an overlapping sliding window fashion
  tStart = time.time()    
  newPatchSize = PATCH_SIZE - PADDING * 2
  for x in range( wsiNpy.shape[1] // newPatchSize ):
    for y in range( wsiNpy.shape[0] // newPatchSize ):
      if (x*newPatchSize + PATCH_SIZE < wsiNpy.shape[1]) and (y*newPatchSize + PATCH_SIZE < wsiNpy.shape[0]):
        if np.average(wsiNpy[y*newPatchSize:y*newPatchSize+PATCH_SIZE, x*newPatchSize:x*newPatchSize+PATCH_SIZE]) < 230:
          coords.append([x*newPatchSize, y*newPatchSize])
  print('  ', 'ROI coords generation finished in:', "{:.2f}".format(time.time()-tStart), "seconds")

  for weight in WEIGHTS:
    model.load_weights(weight)
    tStart = time.time()
    batchPredict(wsiNpy, msk, coords, model)
    print('  ', weight, 'Prediction finished in:', "{:.2f}".format(time.time()-tStart), "seconds")

  #scale labels back to 255
  msk[msk != 0] = 255
 
  tStart = time.time() 
  kwargs = {'bigtiff': True}
  msk = msk.astype(np.uint8)
  imsave(os.path.join(OUTPUT, os.path.splitext(os.path.basename(w))[0] + '.tif'), msk, check_contrast=False, **kwargs)
  print('  ', 'Label image saved in:', "{:.2f}".format(time.time()-tStart), "seconds")

print("done.")
