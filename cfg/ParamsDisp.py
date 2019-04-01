import tensorflow as tf 
from tensorflow.keras.models import load_model, Model 

mdl = load_model('./yolov2-tiny.h5')
mdl.summary()

from keras.utils import plot_model
plot_model(mdl, to_file='yolo_model.png', show_shapes=True)
from IPython.display import Image
Image(filename='yolo_model.png') 

'''
tvars = tf.trainable_variables()
for tvar in tvars:
    print(tvar.name + ' ' + str(tvar.get_shape()) + '\n')
'''