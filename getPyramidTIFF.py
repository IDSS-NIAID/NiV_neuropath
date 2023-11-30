"""
convert regular tiff files to tiff pyramids with aperio tile size of 240x240 px
so they can be loaded in tools such as QuPath
"""
import os, sys, pyvips, glob

if len(sys.argv) < 3:
  print('need INPUT OUTPUT')
  sys.exit()
else:
  ROOT = sys.argv[1]
  OUTPUT = sys.argv[2]

"""
Choose to create OME TIFF files or not.
For now the OME TIFF version seems to be broken
""" 
OME_TIFF = False

pngs = sorted(glob.glob(os.path.join(ROOT, '*.tif')))

if len(pngs) == 0:
  print('no image found, existing.')
  sys.exit()
  
for png in pngs:
  if OME_TIFF == False:
    img = pyvips.Image.new_from_file(png, access='sequential')
  else:
    img = pyvips.Image.new_from_file(png, access='random')

  if img.hasalpha():
    img = img[:-1]

  img_height = img.height
  img_bands = img.bands

  basename = os.path.splitext(os.path.basename(png))[0]
  if OME_TIFF == False:
    img.tiffsave(os.path.join(OUTPUT, basename+'.tif'), 
                 tile=True, 
                 compression='jpeg', 
                 bigtiff=True, 
                 pyramid=True,
                 tile_width=240,
                 tile_height=240,
                 Q=70)
  else:
    #img = pyvips.Image.arrayjoin(img.bandsplit(), across=1)
    img = img.copy()
    img.set_type(pyvips.GValue.gint_type, "page-height", img_height)
    img.set_type(pyvips.GValue.gstr_type, "interpretation", "rgb")
    img.set_type(pyvips.GValue.gstr_type, "image-description",
    f"""<?xml version="1.0" encoding="UTF-8"?>
    <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"
        xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">
        <Image ID="Image:0">
            <!-- Minimum required fields about image dimensions -->
            <Pixels DimensionOrder="XYCZT"
                ID="Pixels:0"
                SizeC="{img_bands}"
                SizeT="1"
                SizeX="{img.width}"
                SizeY="{img_height}"
                SizeZ="1"
                Type="uint8">
                <Channel ID="Channel:0:0" SamplesPerPixel="3">
                    <LightPath/>
                </Channel>
            </Pixels>
        </Image>
    </OME>""")

    img.tiffsave(os.path.join(OUTPUT, basename+'.tif'), 
                 compression="jpeg", 
                 tile=True,
                 tile_width=240,
                 tile_height=240,
                 bigtiff=True,
                 pyramid=True, 
                 subifd=True,
                 Q=70)
  print(png)
  
print('done.')
