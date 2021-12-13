import tensorflow as tf
import tensorflow.keras.applications.vgg19 as vgg19
from tensorflow.keras.models import Model
import skimage.io 
import cv2
import numpy as np
from util import *

import matplotlib.pyplot as plt

#load vgg model
model = vgg19.VGG19(weights=None, include_top=False)#vgg19.VGG19(weights=None, include_top=False)

#read input
raw_img = skimage.io.imread("../data/occean.jpg")
raw_img = cv2.resize(raw_img, (224,224))
raw_random = np.random.rand(1,224,224,3).astype(np.float32)*255
raw_img = np.array([raw_img]).astype(np.float32)

input_init = vgg19.preprocess_input(raw_random)/255
input_content = vgg19.preprocess_input(raw_img)/255
#try content reconstruction

#define a variable
#define a train and do minimize loss
opt = tf.keras.optimizers.Adam(learning_rate=0.6)
input_init = tf.Variable(input_init, dtype=tf.float32)

layer = "block1_conv1"
#define loss upon this variable
def loss():
    #do a model  
    block1_conv1 = model.get_layer(layer).output
    model_b1c1 = Model(inputs= model.input, outputs= block1_conv1)
    F = model_b1c1(input_init)
    P = model_b1c1(input_content)
    return tf.reduce_sum(tf.square(F-P))/2

e = 2
for i in range(200):
    opt.minimize(loss, [input_init])
    print(loss())
    print(i)
    
    if loss() < e:
        break

plt.figure()
result = np.squeeze(input_init).reshape((224,224,3))*255
plt.imshow(undo_preprocessing(result).astype(int))
skimage.io.imsave("../results/cotent_%s.jpg" % layer, undo_preprocessing(result).astype(int))

plt.figure()
orig = np.squeeze(input_content).reshape((224,224,3))*255
plt.imshow(undo_preprocessing(orig).astype(int))
plt.show()