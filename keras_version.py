from keras.layers import Input, LSTM, RepeatVector,Softmax
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
import utils
from lstm_vae import create_lstm_vae
import numpy as np
import sketcher

batch_size=64
timesteps=120
latent_dim=128
input_dim=3


def get_coord_drawings_z_axis():
    leng=timesteps
    #gg = utils.get_slightly_less_simplified_data("",leng)
    gg = utils.get_one_hot_data("", leng)
    while True:
        x_batched = np.zeros([batch_size, leng, 5])
        y_batched = np.zeros([batch_size, leng, 5])
        for b in range(batch_size):
            x = next(gg)

            ll = len(x)
            x_batched[b,:,:]=x
            y_batched[b,:,:]=x

        yield x_batched,y_batched

def main():
    x_input=[]
    gen=get_coord_drawings_z_axis()
    for i in range(0,2048):
        x,y=next(gen)
        x_input.extend(x)

    x_input=np.asarray(x_input)
    print(x_input.shape)
    y_input=x_input
    input_dim = 5 # 13

    vae, enc, gen = create_lstm_vae(input_dim,
        timesteps=timesteps,
        batch_size=batch_size,
        intermediate_dim=32,
        latent_dim=100,
        epsilon_std=1.)

    for i in range(4):
        vae.fit(x_input, x_input, epochs=4,batch_size=batch_size)

        preds = vae.predict(x_input,batch_size=batch_size)

        # pick a column to plot.
        print("[plotting...]")
        print("x: %s, preds: %s" % (x_input.shape, preds.shape))
        tt=preds
        l=np.exp(tt[:,:,2:5])
        ld=np.sum(np.exp(tt[:,:,2:5]),axis=-1,keepdims=True)
        l=l/ld
        print(l[0])
        print(tt[0])
        print(y_input[0])
        tot = sketcher.save_batch_diff_z_axis(list((1 + tt[0:16]) * 128), list((1 + y_input[0:16]) * 128), "./",
                                              str(i) + "FINAL")
    # sketcher.save_batch(list((1 + y[0:16]) * 128), dr + "/imgs", str(i) + str(idx)+"gt")
    # tot = np.array(tot)
    # tot = np.expand_dims(tot, 0)


main()
