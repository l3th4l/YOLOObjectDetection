from skimage import util
import numpy as np 
import copy
import random
import os 
import cv2 
import xml.etree.ElementTree as ET

import cfg_test
from boxUtils import BBox, iou

config = cfg_test.config

def parse_annot(ann_dir, img_dir, extension = '.jpg', labels = []):
    imgs = []

    for ann in os.listdir(ann_dir):
        img = {'object' : []}

        tree = ET.parse(ann_dir + ann)

        for el in tree.iter():
            if 'filename' in el.tag:
                img['filename'] = img_dir + el.text + extension
            if 'width' in el.tag:
                img['width'] = int(el.text)
            if 'height' in el.tag:
                img['height'] = int(el.text)
            if 'obj' in el.tag:
                obj = {}

                for attr in list(el):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text
                    
                    if 'bndbox' in attr.tag:
                        for comp in list(attr):
                            if 'xmin' in comp.tag:
                                obj['xmin'] = int(round(float(comp.text)))
                            if 'xmax' in comp.tag:
                                obj['xmax'] = int(round(float(comp.text)))
                            if 'ymin' in comp.tag:
                                obj['ymin'] = int(round(float(comp.text)))
                            if 'ymax' in comp.tag:
                                obj['ymax'] = int(round(float(comp.text)))

                    if len(labels) > 0 and obj['name'] not in labels:
                        break
                    else:
                        img['object'] += [obj] 
    
        if len(img['object']) > 0:
            imgs.append(img)
    
    return imgs

class BatchGen():

    def im_read(self, img_info, norm = True, aug = True, noise = ['gaussian', 'localvar', 'poisson', 's&p', 'speckle', None]):

        filename = img_info['filename']
        img = cv2.imread(filename)
        
        #normalize image
        if norm :
            img = img / 255.0

        h, w, _ = img.shape()
        objs = copy.deepcopy(img_info['object'])

        img = cv2.resize(img, [config['img_H'], config['img_W'], 3])

        #augment image
        if aug:
            #add noise
            try:
                img = util.random_noise(img, mode = random.choice(noise))
            except:
                pass
        
        for obj in objs:

            for x in ['xmin', 'xmax']:
                obj[x] = int(obj[x] * config['img_W'] / w)  

            for y in ['ymin', 'ymax']:
                obj[y] = int(obj[y] * config['img_H'] / h)

        return img, objs 

    def __init__(self, ann_dir, img_dir, labels = [], train_split = 0.8):
        self.ann_dir = ann_dir
        self.img_dir = img_dir
        self.labels = labels

        self.anchors = [BBox(0, 0, config['anchors'][2 * i], config['anchors'][2 * i + 1]) for i in range(int(len(config['anchors']) // 2))]
        
        self.file_list = parse_annot(self.ann_dir, self.img_dir, labels = self.labels)

        self.train_split = train_split

    def __getitem__(self, idx):
        files = self.file_list[idx * config['batch_size'] : (idx + 1) * config['batch_size']]

        x_batch = np.zeros([config['batch_size'], config['img_H'], config['img_W'], 3])
        b_batch = np.zeros([config['batch_size'], 1, 1, 1, config['max_true_boxes'], 4])
        y_batch = np.zeros([config['batch_size'], config['grid_H'], config['grid_W'], config['box'], 4 + 1 + len(config['labels'])])

        for instance, elem in enumerate(files):
            img, objs = self.im_read(elem)

            for true_ind, obj in enumerate(objs): 
                if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin'] and obj['name'] in self.labels:
                    x_center = (obj['xmin'] + obj['xmax']) * 0.5
                    y_center = (obj['ymin'] + obj['ymax']) * 0.5
                    #normalize the coordinates 
                    x_center = x_center / (float(config['img_W']) / config['grid_W'])
                    y_center = y_center / (float(config['img_H']) / config['grid_H'])

                    grid_x = int(x_center)
                    grid_y = int(y_center)

                    if grid_x < config['grid_W'] and grid_y < config['grid_H']:
                        obj_ind =  self.labels.index(obj['name'])

                        w_center = (obj['xmax'] - obj['xmin']) / (float(config['img_W']) / config['grid_W'])
                        h_center = (obj['ymax'] - obj['ymin']) / (float(config['img_H']) / config['grid_H'])

                        box = [x_center, y_center, w_center, h_center]

                        #get anchor with best iou
                        b_anchor = -1
                        max_iou = -1

                        dummy_box = BBox(0, 0, w_center, h_center)

                        for i, anchor in enumerate(self.anchors):
                            b_iou = iou(dummy_box, anchor)

                            if b_iou > max_iou:
                                b_anchor = i
                                max_iou = b_iou

                        y_batch[instance, grid_y, grid_x, b_anchor, : 4] = box
                        y_batch[instance, grid_y, grid_x, b_anchor, 4] = 1.0
                        y_batch[instance, grid_y, grid_x, b_anchor, 5 + obj_ind] = 1.0

                        b_batch[instance, 0, 0, 0, true_ind % config['max_true_boxes']] = box     

            x_batch[instance] = img
            
        return [x_batch, b_batch], y_batch                                   