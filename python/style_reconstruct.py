import tensorflow as tf
import tensorflow.keras.applications.vgg19 as vgg19
from tensorflow.keras.models import Model
from numpy import mod
import skimage.io 
import cv2
import numpy as np
from util import *

import matplotlib.pyplot as plt

#load vgg model
model = vgg19.VGG19(weights=None, include_top=False)#vgg19.VGG19(weights=None, include_top=False)

#read input style
raw_img1 = skimage.io.imread("../data/vg-hr.jpg")
raw_img1 = cv2.resize(raw_img1, (224,224))
raw_img1 = np.array([raw_img1]).astype(np.float32)
input_style = vgg19.preprocess_input(raw_img1)/255

#initial random 
raw_random = np.random.rand(1,224,224,3).astype(np.float32)*255
input_init = vgg19.preprocess_input(raw_random)/255

plt.figure()
orig = np.squeeze(input_style).reshape((224,224,3))*255
plt.imshow(undo_preprocessing(orig).astype(int))

#try style reconstruction

#define a variable
#define a train and do minimize loss
opt = tf.keras.optimizers.Adam(learning_rate=0.8)
input_init = tf.Variable(input_init, dtype=tf.float32)

A1, _, _ = calc_gram(model, input_style, "block1_conv1")
A2, _, _ = calc_gram(model, input_style, "block2_conv1")
A3, _, _ = calc_gram(model, input_style, "block3_conv1")
A4, _, _ = calc_gram(model, input_style, "block4_conv1")
A5, _, _ = calc_gram(model, input_style, "block5_conv1")
#print("A", A.shape)

#define loss upon this variable
#I did this long loss function because the result is better if the loss function
#is written as a whole than written as several helper functions. Not sure what
#causes the issue but for the sake of appealing reconstruction the loss function
#is written as a whole
def loss():
    #do a model  
    
    G1, M1, N1 = calc_gram(model, input_init, "block1_conv1")
    G2, M2, N2 = calc_gram(model, input_init, "block2_conv1")
    G3, M3, N3 = calc_gram(model, input_init, "block3_conv1")
    G4, M4, N4 = calc_gram(model, input_init, "block4_conv1")
    G5, M5, N5 = calc_gram(model, input_init, "block5_conv1")
    
    e1 = (1/(4*M1*N1))*tf.reduce_sum(tf.square(G1-A1))
    e2 = (1/(4*M2*N2))*tf.reduce_sum(tf.square(G2-A2))
    e3 = (1/(4*M3*N3))*tf.reduce_sum(tf.square(G3-A3))
    e4 = (1/(4*M4*N4))*tf.reduce_sum(tf.square(G4-A4))
    e5 = (1/(4*M5*N5))*tf.reduce_sum(tf.square(G5-A5))
    
    return  e1 +e2 + e3 + e4 + e5 #div * tf.reduce_sum(tf.square(G-A))

e = 0
for i in range(200):
    opt.minimize(loss, [input_init])
    #print(loss())
    print(i)
    if i % 10 == 0:
        print(loss())
    
    if loss() < e:
        break

plt.figure()
result = np.squeeze(input_init).reshape((224,224,3))*255
plt.imshow(undo_preprocessing(result).astype(int))
plt.show()

skimage.io.imsave("../results/style_block1-4_conv1.jpg", undo_preprocessing(result).astype(int))