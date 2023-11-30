"""
Compute positive cell counts from WSIs
"""
import os
import glob
import csv
import sys
import warnings

from skimage.io import imread
from skimage.measure import label, regionprops

from PIL import Image

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = 200000*200000

#pixel size for average cells at 40X: 30px by 30px
CELL_SIZE = 900

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

# prediction mask files
pngs = sorted(glob.glob(os.path.join(INPUT, '*.png')))

cellList = []
for png in pngs:
  imgID = os.path.splitext(os.path.basename(png))[0]

  img = imread(png)
  lbl = label(img)
  props = regionprops(lbl)

  counter = 0
  aggregation = 0
  for prop in props:
    if prop.area >= CELL_SIZE:
      aggregation += prop.area
    else:
      counter += 1

  counter += aggregation // CELL_SIZE
  cellList.append([imgID, counter])

  print(imgID, counter, flush=True)

file = open(OUTPUT, 'w')
with file:
  writer = csv.writer(file)
  writer.writerows(cellList)

print('done.')
