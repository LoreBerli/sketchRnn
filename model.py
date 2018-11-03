import tensorflow as tf
import numpy as np
import tensorflow.contrib.rnn as tfn

BATCH=64
leng=70

def model(x,z=None):
    enc_size=128
    dec_size=enc_size
    llz = tf.constant(value=BATCH, dtype=tf.int32, shape=[BATCH])
    cell_dec = tfn.LSTMCell(dec_size, name="dec")
    #state_ll=tf.placeholder(dtype=tf.float32,shape=[BATCH,enc_size])
    with tf.variable_scope("bet_mod",reuse=tf.AUTO_REUSE):
        if(z==None):
            x=tf.tile(x,(1,leng,1))
            cell_fw=tfn.MultiRNNCell([tfn.LSTMCell(enc_size,name="fw"+str(i)) for i in range(0,3)])
            cell_bw=tfn.MultiRNNCell([tfn.LSTMCell(enc_size,name="bw"+str(i)) for i in range(0,3)])
            outputs,state = tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,inputs=x,dtype=tf.float32,sequence_length=llz,time_major=False,scope="encoder")
            #outputs,state = tf.nn.dynamic_rnn(cell_fw,inputs=x,dtype=tf.float32,sequence_length=llz,time_major=False,scope="encoder")
            latent=tf.concat([state[0].c,state[1].c],axis=-1)
            state_fw= state
            print("STATE",state_fw)
            sigma=tf.layers.dense(latent,64)
            mu=tf.layers.dense(latent,64)
        else:
            middle=z
            state_ll=tfn.MultiRNNCell.zero_state(cell_dec,BATCH,dtype=tf.float32)
        #middle=tf.transpose(middle,[1,2,0])
        ####
        #state_ll=state_fw
        print("middle",middle)
        print("STATE_FW",state_ll)
        #res=tf.expand_dims(middle,axis=-1)
        res=tf.zeros([BATCH,128,3])
        print("Second INPUT",res)

        dec_outs,dec_state= tf.nn.dynamic_rnn(
        cell_dec,
        res,
        initial_state=state_ll,
        time_major=False,
        dtype=tf.float32,
        scope='RNN')

        ###
        acti=None
        flat_outs=tf.layers.flatten(dec_outs[:,:,0:2])
        states=tf.layers.flatten(dec_outs[:,:,2])

        print(flat_outs)
        last = tf.layers.dense(flat_outs,leng*2,activation=tf.tanh)
        last_state=tf.layers.dense(states,leng,activation=tf.tanh)
        #last=tf.expand_dims(last,-1)
        #last_state=tf.expand_dims(last_state,-1)

        last=tf.reshape(last,[BATCH,leng,2])
        #last_state=tf.reshape(last_state,[BATCH,leng,1])
        print("LAST",last)
        print("LAST_STATE",last_state)
        #total=tf.concat([last,last_state],axis=-1)
        #total=tf.reshape(total,[BATCH,leng,3])
        #last=tf.reshape(last,[BATCH,leng,3])

        return last,last_state,middle
