import tensorflow as tf
import numpy as np

def generator(z,c):
    #z 100 dimensionale
    #c onehot encoding delle categorie ,
    with tf.variable_scope("GAN/generator", reuse=False):
        x = tf.concat([z,c],axis=-1)
        x = tf.layers.dense(x,128,activation=tf.tanh)
