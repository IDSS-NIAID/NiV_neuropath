"""
A simple implementation for computing density maps from prediction masks
"""
import os
import sys
import glob

import numpy as np

from skimage.io import imread, imsave

import PIL
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None

# create 1/8 downsized density map
DOWNSCALE = 64
# 1 micrometer
DIAMETER = 400
CELL_SIZE = 900

ROOT = '../../data/'

os.makedirs(os.path.join(ROOT, 'density'), exist_ok=True)

# prediction mask files
imgs = sorted(glob.glob(os.path.join(ROOT, 'results', '*.png')))
imgs = [os.path.basename(f) for f in imgs]

for img in imgs:
  imgNpy = imread(os.path.join(ROOT, 'results', img))

  densityMap = np.ndarray(( imgNpy.shape[0]//DOWNSCALE, imgNpy.shape[1]//DOWNSCALE ), dtype=np.float32)
  densityMap.fill(0)

  for y in range(densityMap.shape[0]):
    for x in range(densityMap.shape[1]):
      ox = x*DOWNSCALE
      oy = y*DOWNSCALE

      if oy+DIAMETER < imgNpy.shape[0] and ox+DIAMETER < imgNpy.shape[1]:
        roi = imgNpy[oy:oy+DIAMETER, ox:ox+DIAMETER]
        d = np.sum(roi)
        densityMap[y, x] = d/255.

  imsave(os.path.join(ROOT, 'density', os.path.splitext(img)[0] + '_' + str(DOWNSCALE) + '.tif'), 
         densityMap, check_contrast=False)
  
  densityMap /= CELL_SIZE
  densityMap = densityMap.astype(np.uint8)

  # these are experimental values
  wp = 50
  mp = 10 

  i = Image.fromarray(densityMap)
  g = ImageOps.colorize(i, black="black", white="white", mid="red", blackpoint=0, whitepoint=wp, midpoint=mp)
  g.save(os.path.join(ROOT, 'density', os.path.splitext(img)[0] + '_' + str(DOWNSCALE) + '_colored.tif'))

print('done.')
