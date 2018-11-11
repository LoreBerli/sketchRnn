import tensorflow as tf
import numpy as np
import tensorflow.contrib.rnn as tfn
import tensorflow.contrib as tfc
import tensorflow.contrib.seq2seq as s2s
import utils
import sketcher
import models
import os
import time

EPOCHS=16
trunc_back=10
BATCH=128
leng=70
latent=100

dataset="dataset/shuffled_bikecar"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

#TODO leng variabile
#TODO TOKEN fine sequenza <EOS> ===> -1 ?
#TODO MASK LOSS per punti dopo <EOS>


def get_coord_drawings_z_axis():
    gg = utils.get_one_hot_data("",leng)
    while True:
        x_batched = np.zeros([BATCH,leng,5])
        y_batched = np.zeros([BATCH,leng,5])
        for b in range(BATCH):
            x = next(gg)
            ll = len(x)
            x_batched[b,:,:]=x
            y_batched[b,:,:]=x

        yield x_batched,y_batched

def train_mlp():


    #mod = models.mlp_ae(latent, BATCH)
    mod=models.vae(latent,BATCH)
    sess = tf.Session()
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    mod.train(sess)

train_mlp()

