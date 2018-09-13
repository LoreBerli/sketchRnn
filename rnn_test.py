import tensorflow as tf
import numpy as np
import tensorflow.contrib.rnn as tfn
import tensorflow.contrib as tfc
import utils
import sketcher
import os
import time

EPOCHS=16
trunc_back=10
BATCH=512
leng=80
dataset="dataset/shuffled_bikecar"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

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

def get_coord_drawings_z_axis():
    gg = utils.get_slightly_less_simplified_data(dataset,leng)
    while True:
        x_batched = np.zeros([BATCH,leng,3])
        y_batched = np.zeros([BATCH,leng,3])
        for b in range(BATCH):
            x = next(gg)
            ll = len(x)
            x_batched[b,:,:]=x
            y_batched[b,:,:]=x

        yield x_batched,y_batched


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


def better_model(x,z=None):
    enc_size=128
    dec_size=enc_size
    llz = tf.constant(value=BATCH, dtype=tf.int32, shape=[BATCH])
    cell_dec = tfn.MultiRNNCell([tfn.LSTMCell(dec_size, name="dec") for i in range(0,3)])
    #state_ll=tf.placeholder(dtype=tf.float32,shape=[BATCH,enc_size])
    with tf.variable_scope("bet_mod",reuse=tf.AUTO_REUSE):
        if(z==None):
            x=tf.tile(x,(1,leng,1))
            print("INPUT:"+str(x))
            cell_fw=tfn.MultiRNNCell([tfn.LSTMCell(enc_size,name="fw"+str(i)) for i in range(0,3)])
            cell_bw=tfn.MultiRNNCell([tfn.LSTMCell(enc_size,name="bw"+str(i)) for i in range(0,3)])
            #outputs,state = tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,inputs=x,dtype=tf.float32,sequence_length=llz,time_major=False,scope="encoder")
            outputs,state = tf.nn.dynamic_rnn(cell_fw,inputs=x,dtype=tf.float32,sequence_length=llz,time_major=False,scope="encoder")

            state_fw= state
            print("STATE",state_fw)
            tot =state_fw# tf.concat([state_fw,state_bw],axis=1)
            #middle=tf.layers.flatten(tot)
            #middle=tf.layers.dense(middle,dec_size,activation=tf.tanh,name="lastOne")
            middle=tot
            state_ll = tot
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
        last = tf.layers.dense(flat_outs,leng*2,activation=acti)
        last_state=tf.layers.dense(states,leng,activation=acti)
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
    _,z=simple_model(x_in,z_in)
    loss=tf.losses.mean_squared_error(y_in,pred)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0015)
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
            print(x.shape)
            print(y.shape)
            #x,y =next(gene)

            sess.run(minimize,{x_in:x,y_in: y})
            if(idx%100==0):
                lo,summary=sess.run([loss,merge],{x_in: x, y_in: y})
                print("::",lo)
                train_writer.add_summary(summary,idx)

            if(idx%5==0):
                print(idx)
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


def test_better_model():
    #qui uso dati :[x,y,pen_down]
    dr=str(time.time())
    os.mkdir("out/"+dr)
    os.mkdir("out/"+dr+"/imgs")
    os.mkdir("out/"+dr+"/gene")
    x_in = tf.placeholder(tf.float32, shape=[BATCH, leng, 3])
    print(x_in)
    y_in = tf.placeholder(tf.float32, shape=[BATCH, leng , 3])
    print(y_in)
    z_in=tf.placeholder(tf.float32,shape=[BATCH,leng*3])
    pred_pos,pred_state,latent = better_model(x_in)
    
    _,_,z=better_model(x_in,z_in)

    loss=tf.losses.mean_squared_error(y_in[:,:,0:2],pred_pos)
    cro = tf.losses.mean_squared_error(y_in[:,:,2],pred_state)
    final = 10.0*loss+cro
    optimizer = tf.train.AdamOptimizer(learning_rate=0.002)
    minimize = optimizer.minimize(final)

    tf.summary.scalar("mse", loss)
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(dr+"/",
                                         sess.graph)
    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    idx = 0
    for i in range(EPOCHS):
        gene = get_coord_drawings_z_axis()
        print(":.......EPOCH "+str(i)+"........")

        merge = tf.summary.merge_all()
        for x,y in gene:
            #print("idx: "+str(idx))
            sess.run(minimize,{x_in:x,y_in: y})
            if(idx%10==0):
                cros,lo,tots,summary=sess.run([cro,loss,final,merge],{x_in: x, y_in: y})

                print("##################")
                print("states",cros)
                print("positions",lo)
                print("total",tots)
                train_writer.add_summary(summary,idx)

            if(idx%50==0):
                print("Saving images...")
                #diff = sess.run(loss, {x_in: x, y_in: y})
                cords,states = sess.run([pred_pos,pred_state], {x_in: x, y_in: y})
                total=np.concatenate([cords,np.reshape(states,[BATCH,leng,1])],-1)
                tt=total
                print(y[0])
                print(cords[0])
                print(states[0])
                print("##################")
                np.random.seed(i)
                img = np.random.randint(0,BATCH)
                #sketcher.save_tested(list((0.5 + tt[img])*256),"denseR",str(i)+str(idx))
                tot=sketcher.save_batch_diff_z_axis(list((1+ tt[0:16]) * 128),list((1 + y[0:16]) * 128), "out/"+dr+"/imgs", str(i) + str(idx))
                #sketcher.save_batch(list((1 + y[0:16]) * 128), dr + "/imgs", str(i) + str(idx)+"gt")
                tot=np.array(tot)
                tot=np.expand_dims(tot,0)
                tf.summary.image(str(i)+"_"+str(idx%500),np.array(tot))

            # if(idx%1000==0):
            #     inter=sess.run(latent, {x_in:x})
            #     intepolations=[gen_interpolation(inter[i],inter[i+1]) for i in range(0,4)]
            #     ltn=[]
            #     for r in intepolations:
            #         ltn.extend(r)
            #     ltn=np.array(ltn)
            #     tt = sess.run(z, {x_in:x,z_in: ltn})
            #     sketcher.save_batch_z_axis(list((1+ tt[0:16]) * 128),"out/"+dr+"/gene", str(i) + str(0))
            #     sketcher.save_batch_z_axis(list((1 + tt[16:32]) * 128), "out/"+dr+ "/gene", str(i) + str(1))
            #     sketcher.save_batch_z_axis(list((1 + tt[32:48]) * 128), "out/"+dr+ "/gene", str(i) + str(2))
            #     sketcher.save_batch_z_axis(list((1 + tt[48:64]) * 128), "out/"+dr+ "/gene", str(i) + str(3))

            idx += 1

def test():
    test_better_model()
test()
