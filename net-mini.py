import tensorflow as tf 
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Lambda, Conv2D, BatchNormalization as BatchNorm, LeakyReLU as LRelu, MaxPooling2D, Reshape
from tensorflow.keras.models import Model, load_model

config = {
        'img_H' : 416, 
        'img_W' : 416, 
        'max_true_boxes' : 50,
        'grid_H' : 13, 
        'grid_W' : 13,
        'box' : 5, 
        'classes': 1 
    }

img_H = config['img_H']
img_W = config['img_W']
grid_H = config['grid_H']
grid_W = config['grid_W']
box = config['box']
classes = config['classes']
max_boxes = config['max_true_boxes']

#Input 
x_img = Input(shape = [img_H, img_W, 3]) #tf.placeholder(tf.float32, [None, size, size, 3])
x_true_boxes = Input(shape = [1, 1, 1, max_boxes, 4]) 

#Layers (from the YOLO model)
#1
x = Conv2D(16, [3, 3], strides = [1, 1], padding = 'same', use_bias = False)(x_img)
x = BatchNorm()(x)
x = LRelu()(x)

x = MaxPooling2D([2, 2], strides = [2, 2])(x)

#2 
x = Conv2D(32, [3, 3], strides = [1, 1], padding = 'same', use_bias = False)(x)
x = BatchNorm()(x)
x = LRelu()(x)

x = MaxPooling2D([2, 2], strides = [2, 2])(x)

#3
x = Conv2D(64, [3, 3], strides = [1, 1], padding = 'same', use_bias = False)(x)
x = BatchNorm()(x)
x = LRelu()(x)

x = MaxPooling2D([2, 2], strides = [2, 2])(x)

#4
x = Conv2D(128, [3, 3], strides = [1, 1], padding = 'same', use_bias = False)(x)
x = BatchNorm()(x)
x = LRelu()(x)

x = MaxPooling2D([2, 2], strides = [2, 2])(x)

#5
x = Conv2D(256, [3, 3], strides = [1, 1], padding = 'same', use_bias = False)(x)
x = BatchNorm()(x)
x = LRelu()(x)

x = MaxPooling2D([2, 2], strides = [2, 2])(x)

#6
x = Conv2D(512, [3, 3], strides = [1, 1], padding = 'same', use_bias = False)(x)
x = BatchNorm()(x)
x = LRelu()(x)

x = MaxPooling2D([2, 2], strides = [1, 1], padding = 'same')(x)

#7
x = Conv2D(1024, [3, 3], strides = [1, 1], padding = 'same', use_bias = False)(x)
x = BatchNorm()(x)
x = LRelu()(x)

#Custom layers
x = Conv2D(512, [3, 3], strides = [1, 1], padding = 'same', use_bias = False)(x)
x = BatchNorm()(x)
x = LRelu()(x)

out = Conv2D(425, [1, 1], strides = [1, 1], padding = 'same', use_bias = True)(x)
out = Reshape([grid_H, grid_W, box, 4 + 1 + classes])(out)

# small hack to allow true_boxes to be registered when Keras build the model 
# for more information: https://github.com/fchollet/keras/issues/2790
out = Lambda(lambda a : a[0])([out, x_true_boxes])

#Make the model 
mdl = Model(inputs = [x_img, x_true_boxes], outputs = out)
mdl.summary()

def load_weights(path = './cfg/yolov2-tiny.h5'):
    
    mdl.load_weights(path, by_name = False)

    print('... Loaded parameters')

#Load pretrained model 
#yolo_model = load_model("./cfg/yolov2-tiny.h5")

#Load pretrained weights for hidden layers
'''
for new_layer, layer in zip(mdl.layers[1:], yolo_model.layers[1:]):
    new_layer.set_weights(layer.get_weights())
    new_layer.trainable = False
'''