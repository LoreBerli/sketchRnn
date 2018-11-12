import tensorflow as tf
import numpy as np
import tensorflow.contrib.rnn as tfn
import utils
import sketcher
import time
import os

from tensorflow.contrib.legacy_seq2seq import basic_rnn_seq2seq



leng=120
EPOCHS=40

def build_paths():
    dr = str(time.time())
    os.mkdir("out/" + dr)
    os.mkdir("out/" + dr + "/imgs")
    os.mkdir("out/" + dr + "/gene")
    return dr


class vae:
    def __init__(self,latent_size,batch_size):
        self.batch_size = batch_size
        self.leng = 120
        self.target=tf.placeholder(tf.float32,[self.batch_size,self.leng,5])
        self.input=tf.placeholder(tf.float32,[self.batch_size,self.leng,5])
        self.input_size=tf.placeholder(tf.int32,[self.batch_size])
        self.alpha=0.1
        self.latent_size=latent_size
        self.enc_size=16

        self.dr=build_paths()

        self.cell_enc=tfn.LSTMCell
        self.cell_dec = tfn.LSTMCell
        self.last = self.build()
        self.lat_loss=self.latent_loss()
        self.rec_loss=self.reconstruction_loss()
        self.loss=self.alpha*self.lat_loss+self.rec_loss
        tf.summary.scalar("latent_loss",self.lat_loss)
        tf.summary.scalar("reconstruction_loss", self.rec_loss)
        tf.summary.scalar("total_loss",self.loss)
        self.merged=tf.summary.merge_all()



        self.optimizer=tf.train.RMSPropOptimizer(learning_rate=0.001)
        # gvs = self.optimizer.compute_gradients(self.loss)
        # capped_gvs = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in gvs]
        self.minimize = self.optimizer.minimize(self.loss)


    def latent_loss(self):
        #lat = 0.5*tf.reduce_sum(tf.square(self.z_mean) + tf.square(self.z_std) - tf.log(tf.square(self.z_std)) - 1, 1)
        lat=- 0.5 * tf.reduce_sum(1 + self.z_std - tf.square(self.z_mean) - tf.exp(self.z_std), axis=-1)
        lat = tf.reduce_mean(lat)
        return lat
    def reconstruction_loss(self):
            return tf.losses.mean_squared_error(self.target, self.last)
            #return tf.losses.mean_squared_error(self.target[:,:,0:2],self.last[:,:,0:2])+tf.losses.softmax_cross_entropy(self.target[:,:,2:5],self.last[:,:,2:5])



    def build(self):
        with tf.variable_scope("vae_model", reuse=tf.AUTO_REUSE):
            x = self.input
            print("pre",x)
            x = tf.tile(x, (1, 1, leng))
            print("post",x)
            cell_fw = self.cell_enc(self.enc_size, name="fw")
            cell_bw = self.cell_enc(self.enc_size, name="bw")
            outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs=x, dtype=tf.float32,
                                                             sequence_length=self.input_size, time_major=False, scope="encoder")
            print(state)
            if(self.cell_enc==tfn.LSTMCell):
                latent_h=tf.concat([state[0].h,state[1].h],axis=-1)
                latent_c = tf.concat([state[0].c, state[1].c], axis=-1)
                print("LATENT",latent_c)
            else:
                latent=tf.concat([state[0],state[1]],axis=-1)
                print("ELSE LATENT ",latent)
            self.z_mean=tf.layers.dense(latent_h,self.latent_size)
            self.z_std=tf.layers.dense(latent_h,self.latent_size)
            mu=self.z_mean
            sigma=self.z_std
            samples = tf.random_normal([self.batch_size,self.latent_size], 0, 1, dtype=tf.float32)
            sampled_z = mu + tf.exp(sigma/2)* samples

            self.z_mean_c=tf.layers.dense(latent_c,self.latent_size)
            self.z_std_c=tf.layers.dense(latent_c,self.latent_size)
            mu_c=self.z_mean_c
            sigma_c=self.z_std_c
            samples_c = tf.random_normal([self.batch_size,self.latent_size], 0, 1, dtype=tf.float32)
            sampled_z_c = mu_c + tf.exp(sigma_c/2)* samples_c
            print("SAMPLED ",sampled_z)
            if (self.cell_dec == tfn.LSTMCell):
                latent_state=tfn.LSTMStateTuple(sampled_z,sampled_z_c)#,tfn.LSTMStateTuple(mu,sigma))
            else:
                latent_state=sampled_z
            res=tf.expand_dims(sampled_z,axis=-1)
            res =tf.tile(res,[1,1,leng])
            res = tf.transpose(res,[0,2,1])
            print("RES",res)
            coord_outs, dec_state = tf.nn.dynamic_rnn(
                tfn.OutputProjectionWrapper(self.cell_dec(self.latent_size,name="dec"), 2,activation=tf.nn.tanh),

                res,
                initial_state=latent_state,
                sequence_length=self.input_size,
                time_major=False,
                dtype=tf.float32,
                scope='RNN_cord')

            state_outs, dec_state = tf.nn.dynamic_rnn(
                tfn.OutputProjectionWrapper(self.cell_dec(self.latent_size, name="dec_state"),3),

                res,
                initial_state=latent_state,
                sequence_length=self.input_size,
                time_major=False,
                dtype=tf.float32,
                scope='RNN')

            print("OUT : ",coord_outs)
            print("OUT : ", state_outs)
            dec_outs=tf.concat([coord_outs,state_outs],axis=-1)
            #
            #
            # out=tf.reshape(dec_outs,[self.batch_size,leng*self.latent_size])
            # out = tf.layers.dense(out,leng*5,activation=tf.nn.tanh)
            # out=tf.reshape(out,[self.batch_size,leng,5])
            return dec_outs

    def train(self,sess):
        import time
        y=tf.placeholder(tf.float32,[self.batch_size,leng,5])
        MAX_ITER=500
        tiled_leng=np.tile(self.leng,self.batch_size)
        self.writer = tf.summary.FileWriter("./test/"+str(self.dr), sess.graph)

        for e in range(0,EPOCHS):
            g = utils.get_coord_drawings_z_axis(self.batch_size, leng)
            for i in range(0,MAX_ITER):

                x_,y_ = next(g)

                #self.input=X
                #cros,lo,tots,summary=sess.run([cro,loss,final,merge],{x_in: x, y_in: y})
                ls,_=sess.run([self.loss,self.minimize],feed_dict={self.input:x_,self.target:y_,self.input_size:tiled_leng})
                print("LOSS:",ls)

                if (i % 2 == 0):
                    summary=sess.run(self.merged,
                             feed_dict={self.input: x_, self.target: y_, self.input_size: tiled_leng})
                    self.writer.add_summary(summary, i+(e*MAX_ITER))

                #tot = sketcher.save_batch_diff_z_axis(list((1 + tt[0:16]) * 128), list((1 + y_draw[0:16]) * 128),"out/" + str(self.dr) + "/imgs", str(e))
                if(i%100==0):

                    tt = sess.run(self.last, feed_dict={self.input: x_, self.input_size: tiled_leng})
                    print(y_[0,:,0:2])
                    print(tt[0,:,0:2])
                    tot = sketcher.save_batch_diff_z_axis(list((1 + tt[0:16]) * 128), list((1 + y_[0:16]) * 128),"out/" + str(self.dr) + "/imgs",str(e)+"_"+str(i))




class mlp_ae:
    def __init__(self,latent,BATCH):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.BATCH=BATCH
        self.latent=latent
        self.dr = build_paths()
        self.input = tf.placeholder(dtype=tf.float32, shape=[BATCH, leng* 5])
        self.output=self.build()
        self.loss=self.ae_loss(self.output,self.input)
        self.min=self.optimizer.minimize(self.loss)
        #?????


    def build(self):
        #crude autoencoder
        x=self.input
        input_size=leng*5
        acti=tf.nn.relu
        print(x)
        x = tf.layers.dense(x,512,activation=acti)

        x = tf.layers.dense(x,128,activation=tf.nn.sigmoid)
        middle = tf.layers.dense(x,self.latent, activation=acti)
        x = tf.layers.dense(middle,128,activation=acti)
        x = tf.layers.dense(x,input_size,activation=tf.nn.tanh)

        #NONONONONONONO
        self.last= x
        self.center,self.std=tf.nn.moments(middle,axes=[0,1])


        self.sampler=middle
        return x

    def sample(self,sess,md,std):

        seeds = np.random.normal(md,std,[BATCH,self.latent])
        out = sess.run(self.last, feed_dict={self.sampler: seeds})
        tt = np.reshape(out, [BATCH, leng, 5])
        tot = sketcher.save_batch_diff_z_axis(list((1 + tt[0:16]) * 128), list((1 + tt[0:16]) * 128),
                                              "out/" + str(self.dr) + "/imgs", "SAMPLING")

    def ae_loss(self,y_pred,y_true):
        return tf.losses.mean_squared_error(y_true,y_pred)

    def train(self,sess):


        for e in range(0,EPOCHS):
            g = utils.get_coord_drawings_z_axis(BATCH, leng)
            for i in range(0,1000):
                x_,y_ = next(g)
                x_=np.reshape(x_,[BATCH,-1])
                y_=np.reshape(y_,[BATCH,-1])
                ls=sess.run([self.loss,self.min],feed_dict={self.input:x_})

            out,m,avg,std=sess.run([self.output,self.sampler,self.center,self.std],feed_dict={self.input:x_})

            print(avg,std)
            tt=np.reshape(out,[BATCH,leng,5])
            y_draw=np.reshape(y_,[BATCH,leng,5])
            self.sample(sess,avg,std)
            tot = sketcher.save_batch_diff_z_axis(list((1 + tt[0:16]) * 128), list((1 + y_draw[0:16]) * 128),"out/" + str(self.dr) + "/imgs", str(e))

