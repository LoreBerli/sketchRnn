import tensorflow as tf
import numpy as np
import tensorflow.contrib.rnn as tfn
import utils
import sketcher
import os
import time

EPOCHS=8
trunc_back=10
BATCH=64
leng=72
dataset="dataset/ambu_car"


def get_drawings():
        gg=utils.get_simplified_data(leng)
        while True:

            x_batched = []
            y_batched = []
            for b in range(BATCH):
                x=next(gg)
                ll = len(x)
                x_batched.append(x)
                y_batched.append(x)

            yield np.array(x_batched).reshape([BATCH, leng * 2]), np.array(y_batched).reshape([BATCH, leng * 2])


def get_coord_drawings():
    gg = utils.get_simplified_data(dataset,leng)
    while True:

        x_batched = []
        y_batched = []
        for b in range(BATCH):
            x = next(gg)
            ll = len(x)
            x_batched.append(x)
            y_batched.append(x)

        yield np.array(x_batched), np.array(y_batched)


def rnn_model_dense(x):
    mu = tfn.MultiRNNCell([tfn.LSTMCell(leng),tfn.LSTMCell(leng)])
    outs, out_fw, out_bk = tf.nn.static_bidirectional_rnn(mu,mu, inputs=[x],
                                                          dtype=tf.float32)

    print(out_bk[-1])
    ss=tf.concat([out_bk[-1],out_fw[-1]],axis=0)
    # print(ss.shape)
    # #ss = tf.reshape(ss,[BATCH,leng*4])
    print(ss.shape)
    # den = tf.layers.dense(ss, 64)

    # forw_out,forw_state = tf.nn.static_rnn(tfn.LSTMCell(num_units=leng*2,name="forward"),inputs=[tf.concat([outf,outb],axis=-1)],dtype=tf.float32)
    mu = tfn.MultiRNNCell([tfn.LSTMCell(leng,name="forw0"),tfn.LSTMCell(leng,name="forw1")])
    forw_out, forw_state = tf.nn.static_rnn(mu,
                                            inputs=[ss], dtype=tf.float32)
    pred = forw_out[-1]
    return pred

def simple_model(x,z=None):
    #x=tf.unstack(x,axis=0)
    with tf.variable_scope("scp",reuse=tf.AUTO_REUSE):
        ll = tf.constant(value=leng, dtype=tf.int32, shape=[leng])
        llz = tf.constant(value=BATCH, dtype=tf.int32, shape=[BATCH])
        if(z==None):
            state_size=128
            cell=tfn.LSTMCell(state_size,name="diomerda")
            cell3 = tfn.LSTMCell(state_size, name="diomerda3")


            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell,
                cell_bw=cell3,
                inputs=x,

                dtype=tf.float32,
                time_major=False,
                scope="rnn_1"
         )


            fl = tf.reshape(outputs[-1],shape=[BATCH,-1,2])

            flat=tf.layers.flatten(fl)
            middle = tf.layers.dense(flat,leng*2)
        else:
            middle=z

        print(middle.shape,"flat")
        pred = tf.reshape(middle, shape=[BATCH, leng, 2])
        # pred =tf.layerfl.dense(fl,128)
        # pred = tf.layers.dense(pred,leng)
        # print(pred.shape)
        #pred = tf.expand_dims(pred,axis=-1)
        print(pred.shape)
        #pred = tf.unstack(pred,axis=0)
        cell2=tfn.LSTMCell(2,name="diomerda2")

        outputss, states = tf.nn.dynamic_rnn(
            cell=cell2,
            inputs=pred,
            dtype=tf.float32,
            sequence_length=llz,
            scope="rnn_2")
        print("outputs",outputss.shape)
        #preds = tf.nn.tanh(outputss)
        preds=outputss
        return preds,middle


def rnn_model_raw(x):
    outs, out_fw, out_bk = tf.nn.static_bidirectional_rnn(tfn.LSTMCell(num_units=leng * 2),
                                                          tfn.LSTMCell(num_units=leng * 2), inputs=[x],
                                                          dtype=tf.float32)

    # forw_out,forw_state = tf.nn.static_rnn(tfn.LSTMCell(num_units=leng*2,name="forward"),inputs=[tf.concat([outf,outb],axis=-1)],dtype=tf.float32)
    forw_out, forw_state = tf.nn.static_rnn(tfn.LSTMCell(num_units=leng * 2, name="forward"),
                                            inputs=[outs[-1]], dtype=tf.float32)
    pred = forw_out[-1]
    return pred





def gen_interpolation(x,y):
    STEPS=15
    out=[]
    for s in range(STEPS+1):
        inter=x*(float(STEPS-s)/float(STEPS))+y*(float(s)/float(STEPS))
        out.append(inter)
    return out

def test_dense():
    dr=str(time.time())
    os.mkdir("out/"+dr)
    os.mkdir("out/"+dr+"/imgs")
    os.mkdir("out/"+dr+"/gene")
    x_in = tf.placeholder(tf.float32, shape=[BATCH, leng, 2])
    print(x_in)
    y_in = tf.placeholder(tf.float32, shape=[BATCH, leng , 2])
    print(y_in)
    z_in=tf.placeholder(tf.float32,shape=[BATCH,leng*2])
    pred,latent = simple_model(x_in)
    z,_=simple_model(x_in,z_in)
    loss=tf.losses.mean_squared_error(y_in,pred)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0015)
    minimize = optimizer.minimize(loss)

    tf.summary.scalar("mse", loss)
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(dr+"/",
                                         sess.graph)
    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    idx = 0
    for i in range(EPOCHS):
        gene = get_coord_drawings()
        print(":...............")

        merge = tf.summary.merge_all()
        for x,y in gene:

            #x,y =next(gene)

            sess.run(minimize,{x_in:x,y_in: y})
            if(idx%100==0):
                lo,summary=sess.run([loss,merge],{x_in: x, y_in: y})
                print("::",lo)
                train_writer.add_summary(summary,idx)

            if(idx%100==0):
                #diff = sess.run(loss, {x_in: x, y_in: y})
                tt = sess.run(pred, {x_in: x, y_in: y})
                np.random.seed(i)
                img = np.random.randint(0,BATCH)
                #sketcher.save_tested(list((0.5 + tt[img])*256),"denseR",str(i)+str(idx))
                tot=sketcher.save_batch_diff(list((1+ tt[0:16]) * 128),list((1 + y[0:16]) * 128), "out/"+dr+"/imgs", str(i) + str(idx))
                #sketcher.save_batch(list((1 + y[0:16]) * 128), dr + "/imgs", str(i) + str(idx)+"gt")
                tot=np.array(tot)
                tot=np.expand_dims(tot,0)
                tf.summary.image(str(idx%500),np.array(tot))

            if(idx%1000==0):
                inter=sess.run(latent, {x_in:x})
                intepolations=[gen_interpolation(inter[i],inter[i+1]) for i in range(0,4)]
                ltn=[]
                for r in intepolations:
                    ltn.extend(r)
                ltn=np.array(ltn)
                tt = sess.run(z, {x_in:x,z_in: ltn})
                sketcher.save_batch(list((1+ tt[0:16]) * 128),"out/"+dr+"/gene", str(i) + str(0))
                sketcher.save_batch(list((1 + tt[16:32]) * 128), "out/"+dr+ "/gene", str(i) + str(1))
                sketcher.save_batch(list((1 + tt[32:48]) * 128), "out/"+dr+ "/gene", str(i) + str(2))
                sketcher.save_batch(list((1 + tt[48:64]) * 128), "out/"+dr+ "/gene", str(i) + str(3))

            idx += 1


def test():
    test_dense()
test()