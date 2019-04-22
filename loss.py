import numpy as np 
import tensorflow as tf
from boxUtils import tf_iou

import cfg_test

config = cfg_test.config

def loss(out, labels):
    mask_shape = tf.shape(labels)[ : 4]

    cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(config['grid_W']), config['grid_H']), [1, config['grid_H'], config['grid_W'], 1, 1]))
    cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))
    
    cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [config['batch_size'], 1, 1, 5, 1])

    coord_mask = tf.zeros(mask_shape)
    conf_mask = tf.zeros(mask_shape)
    class_mask = tf.zeros(mask_shape)

    seen = tf.Variable(0.0)
    total_recall = tf.Variable(0.0)

    #configure predictions 

    pred_box_xy = tf.sigmoid(out[..., : 2]) + cell_grid
    pred_box_wh = tf.exp(out[..., 2 : 4]) * np.reshape(config['anchors'], [1, 1, 1, config['box'], 2])
    pred_box_conf = tf.sigmoid(out[..., 4]) 
    pred_box_class = out[..., 5:]

    #configure labels
    true_box_xy = labels[..., : 2]
    true_box_wh = labels[..., 2 : 4]

    iou = tf_iou(true_box_xy, true_box_wh, pred_box_xy, pred_box_wh)

    true_box_conf = iou * labels[..., 4]
    true_box_class = tf.arg_max(labels[..., 5 : ], -1)

    #configure masks 
    coord_mask = tf.expand_dims(labels[..., 4], axis = -1)

    