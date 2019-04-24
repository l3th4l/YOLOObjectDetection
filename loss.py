import numpy as np 
import tensorflow as tf
from boxUtils import tf_iou

import cfg_test

config = cfg_test.config

CLASS_WEIGHTS    = np.ones(config['classes'], dtype='float32')

def loss(out, labels, true_boxes):
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

    true_xy = true_boxes[..., : 2]
    true_wh = true_boxes[..., 2 : 4]

    pred_xy_expanded = tf.expand_dims(pred_box_xy, axis = 4)
    pred_wh_expanded = tf.expand_dims(pred_box_wh, axis = 4)

    iou = tf_iou(true_xy, true_wh, pred_xy_expanded, pred_wh_expanded)
    best_iou = tf.reduce_max(iou, axis = 4)

    conf_mask = conf_mask + tf.to_float(best_iou < 0.6) * (1 - labels[..., 4])
    class_mask = labels[..., 4] * tf.gather(CLASS_WEIGHTS, true_box_class) 

    #Warm up training
    no_box_mask = tf.to_float(coord_mask < 0.5)
    seen = tf.assign_add(seen, 1)

    true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, config['warm_up_epochs']),
                                                   lambda : [true_box_xy + (0.5 + cell_grid) * no_box_mask, 
                                                             true_box_wh + tf.ones_like(true_box_wh) * np.reshape(config['anchors'], [1, 1, 1, config['box'], 2]) * no_box_mask,
                                                             tf.ones_like(coord_mask)],
                                                   lambda : [true_box_xy, 
                                                             true_box_wh, 
                                                             coord_mask] )
    
    #Combine losses 
    nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0))
    nb_conf_box = tf.reduce_sum(tf.to_float(conf_mask > 0))
    nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0))

    loss_xy = tf.reduce_sum(tf.square(true_box_xy - pred_box_xy) * coord_mask) / (nb_coord_box + 1e-6) / 2
    loss_wh = tf.reduce_sum(tf.square(true_box_wh - pred_box_wh) * coord_mask) / (nb_coord_box + 1e-6) / 2
    loss_conf = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * conf_mask) / (nb_conf_box + 1e-6) / 2
    loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = true_box_class, logits = pred_box_class)
    loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)