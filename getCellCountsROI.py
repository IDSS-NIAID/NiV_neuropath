"""
Compute positive cell counts from WSIs with ROI definition
The reported positive cell counts are computed in the defined ROIs only
Used in the validation step
"""
import os
import glob
import csv
import sys
import warnings
import geojson

import numpy as np

from skimage.io import imread
from skimage.measure import label, regionprops

from PIL import Image

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = 200000*200000

# a simple utility function to get bounding box
def getBoundingBox(roi):
  maxX = 0
  maxY = 0
  minX = 99999999
  minY = 99999999

  if roi['geometry']['type'] == 'MultiPolygon':
    pls = roi['geometry']['coordinates']
    for pl in pls:
      vertices = pl[0]
      for vertex in vertices:
        x = vertex[0]
        y = vertex[1]
        minX = x if x < minX else minX
        minY = y if y < minY else minY
        maxX = x if x > maxX else maxX
        maxY = y if y > maxY else maxY
  elif roi['geometry']['type'] == 'Polygon':
    vertices = roi['geometry']['coordinates'][0]
    for vertex in vertices:
      x = vertex[0]
      y = vertex[1]
      minX = x if x < minX else minX
      minY = y if y < minY else minY
      maxX = x if x > maxX else maxX
      maxY = y if y > maxY else maxY

  return [minX, minY, maxX, maxY]

#pixel size for average cells at 40X: 30px by 30px
CELL_SIZE = 1600
CELL_MIN_SIZE = 40

if len(sys.argv) < 2:
  print('need INPUT_PATH, existing.')
  sys.exit()
else:
  INPUT = sys.argv[1]

# use the basename as output file name
# a trailing '/' will produce empty basename
OUTPUT = os.path.basename(INPUT)
if len(OUTPUT) == 0:
  print('please check INPUT_PATH, existing.')
  sys.exit()

OUTPUT = OUTPUT + '.csv'
print('saving to', OUTPUT)

# prediction mask images
pngs = sorted(glob.glob(os.path.join(INPUT, 'results', '*.png')))
# ROI annotation in geojson format
gjs = sorted(glob.glob(os.path.join(INPUT, 'rois', '*.geojson')))

cellList = [['imageID', 'cellCounts']]

for png, gj in zip(pngs, gjs):
  imgID = os.path.splitext(os.path.basename(png))[0]


  img = imread(png)

  f = open(gj)
  rois = geojson.load(f)
  
  total = 0

  for roi in rois['features']:
    if 'geometry' not in roi:
      continue
    
    bbox = getBoundingBox(roi)

    imgROI = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    print('  ROI size:', imgROI.shape, 'bbox:', bbox)

    # skip blank label area
    if imgROI.shape[0] == 0 or imgROI.shape[1] == 0:
      print('  skipping zero sized roi')
      continue
 
    lbl = label(imgROI)
    props = regionprops(lbl)
      
    counter = 0
    aggregation = 0
    for prop in props:
      if prop.area >= CELL_SIZE or prop.area < CELL_MIN_SIZE:
        aggregation += prop.area
      else:
        counter += 1

    counter += aggregation // CELL_SIZE
    print('  cell count:', counter)

    total += counter
 
  cellList.append([imgID, total])

  print('total', total, flush=True)

file = open(OUTPUT, 'w')
with file:
  writer = csv.writer(file)
  writer.writerows(cellList)

print('done.')
