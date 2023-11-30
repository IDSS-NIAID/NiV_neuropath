"""
Overlay segmentation mask outlines onto the original WSI
for easier visualization of segmentation results
"""
import openslide
import glob
import os
import numpy as np
import PIL
from PIL import Image, ImageEnhance, ImageFilter

import collections
import multiprocessing

import warnings
warnings.filterwarnings("ignore")

Image.MAX_IMAGE_PIXELS = 200000*200000

def processSingleImage(item):
  wsi = openslide.OpenSlide(item.w)
  wsiNpy = np.array(wsi.read_region((0, 0), 0, (wsi.dimensions[0], wsi.dimensions[1])))[:, :, :3]

  print(os.path.basename(item.w), 'loaded', flush=True)

  imgID = os.path.splitext(os.path.basename(item.w))[0]
  lbl = Image.open(os.path.join(ROOT, 'results', DATASET, imgID + '.png'))
  lbl_erosion = lbl.filter(ImageFilter.MinFilter(3))

  lblNpy = np.array(lbl)
  lbl_erosionNpy = np.array(lbl_erosion)

  lblNpy[lbl_erosionNpy > 0] = 0

  wsiNpy[lblNpy > 0] = [255, 0, 0]

  wsiImg = Image.fromarray(wsiNpy)
  wsiImg.save(os.path.join(ROOT, 'overlays', DATASET, imgID + '.png'))

  print(os.path.basename(item.w), 'saved', flush=True)
#end def processSingeImage

ROOT = 'root'
DATASET = 'dataset'

os.makedirs(os.path.join(ROOT, 'overlays', DATASET), exist_ok=True)

wsis = sorted(glob.glob(os.path.join(ROOT, 'datasets', DATASET, '*.svs')))

FolderPairs = collections.namedtuple('FolderPairs', ['w'])
pairs = []

for wsi in wsis:
  pairs.append(FolderPairs(w=wsi))

pairsTuple = tuple(pairs)

pool = multiprocessing.Pool(processes=6)
pool.map(processSingleImage, pairsTuple)
