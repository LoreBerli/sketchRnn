import tensorflow as tf
import numpy as np
import utils
import sketcher
import models
import os
import time



dataset="dataset/shuffled_bikecar"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#TODO leng variabile
#TODO TOKEN fine sequenza <EOS> ===> -1 ?
#TODO MASK LOSS per punti dopo <EOS>




def train_mlp():
    load=True
    BATCH = 100
    rates= [0.0002]
    encs=[16,32,64,128]

    accel=[1.2]
    latents = [32,64,128,256]
    activ=[None]
    e_l={256:[128]}
    EPOCHS=32
    for e in e_l:
        for l in e_l[e]:
            for a in activ:
                for ccel in accel:
                    tf.reset_default_graph()
                    sess = tf.Session()

                    mod = models.vae(l, BATCH, e, 0.001, EPOCHS,ccel,a)
                    init_op = tf.global_variables_initializer()

                    sess.run(init_op)
                    mod.train(sess)


train_mlp()

