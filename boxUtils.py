import numpy as np 
import tensorflow as tf 

class BBox():
    def __init__(self, xmin, ymin, xmax, ymax, c = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.c = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
        
        return self.score

def interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0 
        else:
            return min(x2, x4) - x3

def iou(box1, box2):
    intersect_w = interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersection = intersect_h * intersect_w 

    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin
    
    union = w1 * h1 + w2 * h2 - intersection

    return float(intersection) / union

def tf_iou(box_a_xy, box_a_wh, box_b_xy, box_b_wh):

    box_a_mins = box_a_xy - box_a_wh / 2
    box_a_maxs = box_a_xy + box_a_wh / 2

    box_b_mins = box_b_xy - box_b_wh / 2
    box_b_maxs = box_b_xy + box_b_wh / 2

    intersection_mins = tf.maximum(box_a_mins, box_b_mins)
    intersection_maxs = tf.minimum(box_a_maxs, box_b_maxs)
    intersection_wh = tf.maximum(intersection_maxs - intersection_mins, 0)

    intersection_area = intersection_wh[..., 0] * intersection_wh[..., 1]
    
    box_a_area = box_a_wh[..., 0] * box_a_wh[..., 1]
    box_b_area = box_b_wh[..., 0] * box_b_wh[..., 1]

    union_area = box_a_area + box_b_area - intersection_area

    return tf.divide(intersection_area, union_area)