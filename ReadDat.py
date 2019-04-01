import xml.etree.ElementTree as ET
import skimage.io as io 
import skimage.transform as tr
from matplotlib import pyplot as plt 
from matplotlib import patches
import os 

%matplotlib inline

ann_path = './dataset/Annotations/'
img_path = './dataset/JPEGImages/'

#ann_files = os.listdir(ann_path)

def parse_annotations(ann_file, im_path, labels = ['RBC'], prefix  = '.jpg'):
    
    img = {'object' : []}
    
    tree = ET.parse(ann_file)
    
    for elem in tree.iter():
        if 'filename' in elem.tag:
            img['filename'] = img_path + elem.text + prefix
        elif 'width' in elem.tag:
            img['width'] = int(elem.text)
        elif 'height' in elem.tag:
            img['height'] = int(elem.text)
        elif 'object' in elem.tag:
            obj = {}
            
            for attr in list(elem):
                if 'name' in attr.tag:
                    obj['name'] = attr.text
                    if len(labels) > 0 and obj['name'] not in labels:
                        print(obj['name'])
                        break
                    else:
                        img['object'].append(obj)
                if 'bndbox' in attr.tag:
                    for dim in list(attr):
                        if 'xmin' in dim.tag:
                            obj['xmin'] = int(round(float(dim.text)))
                        if 'ymin' in dim.tag:
                            obj['ymin'] = int(round(float(dim.text)))
                        if 'xmax' in dim.tag:
                            obj['xmax'] = int(round(float(dim.text)))
                        if 'ymax' in dim.tag:
                            obj['ymax'] = int(round(float(dim.text)))

    #converts bbox to YOLO format
    for obj in img['object']:
        obj['x'] = (obj['xmin'] + obj['xmax']) / (2 * img['width']) 
        obj['y'] = (obj['ymin'] + obj['ymax']) / (2 * img['height'])
        obj['w'] = obj['xmax'] - obj['xmin'] / img['width']
        obj['h'] = obj['ymax'] - obj['ymin']

    #detete unused keys
    del obj['xmin']
    del obj['ymin']
    del obj['xmax']
    del obj['ymax']

    return img

'''
im_anno = parse_annotations(ann_path + ann_files[20], img_path)

im = io.imread(im_anno['filename'])

fig, ax = plt.subplots()

ax.imshow(im)

for object in im_anno['object']:
    rect = patches.FancyBboxPatch([object['xmin'], object['ymin']],
                                  object['xmax'] - object['xmin'],
                                  object['ymax'] - object['ymin'],
                                  boxstyle = 'circle', fill = False, linewidth = 1.5, edgecolor = 'r')
    ax.add_patch(rect)
'''