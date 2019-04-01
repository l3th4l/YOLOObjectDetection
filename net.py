import tensorflow as tf 
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Lambda, Conv2D, BatchNormalization as BatchNorm, LeakyReLU as LRelu, MaxPooling2D
from tensorflow.keras.models import Model, load_model

size = 416

#Input 
x_in = Input(shape = [size, size, 3]) #tf.placeholder(tf.float32, [None, size, size, 3])

#Layers (from the YOLO model)
#1
x = Conv2D(16, [3, 3], strides = [1, 1], padding = 'same', use_bias = False)(x_in)
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

#Make the model 
mdl = Model(inputs = x_in, outputs = out)
mdl.summary()

#Load pretrained model 
yolo_model = load_model("./cfg/yolov2-tiny.h5")

#Load pretrained weights for hidden layers
for new_layer, layer in zip(mdl.layers[1:], yolo_model.layers[1:]):
    new_layer.set_weights(layer.get_weights())
    new_layer.trainable = False
print('... Loaded parameters')