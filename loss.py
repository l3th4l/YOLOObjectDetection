import numpy as np 
import tensorflow as tf

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
    