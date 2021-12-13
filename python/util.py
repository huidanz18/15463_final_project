import numpy as np
from keras.models import Model
import tensorflow as tf

def calc_gram(model, input, layer):
    
    layerConv = model.get_layer(layer).output
    layerModel = Model(inputs= model.input, outputs= layerConv)
    
    P = layerModel(input)
    P = P[0]
    M = P.shape[0]*P.shape[1]
    N = P.shape[2]
    
    P = tf.reshape(P, (M, N))
    P_T = tf.transpose(P) 
    A = tf.matmul(P_T,P)
    
    return A, M, N
    
#code from stackoverflow to undo what preprocessing does to the image
#https://stackoverflow.com/questions/55987302/reversing-the-image-preprocessing-of-vgg-in-keras-to-return-original-image/55987630
def undo_preprocessing(img):
    x = np.copy(img)
    mean = [103.939, 116.779, 123.68]

    x[:, :, 0] += mean[0]
    x[:, :, 1] += mean[1]
    x[:, :, 2] += mean[2]
    
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    
    return x