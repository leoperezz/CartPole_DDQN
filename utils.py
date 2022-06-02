import numpy as np
import random
import tensorflow as tf


def frame_process(o):
    
    o=o[1:176:2, ::2]
    o=tf.divide(tf.cast(o,dtype=tf.float32),tf.constant(255.0))
    o=tf.image.rgb_to_grayscale(o)
    return o


class Buffer:

    def __init__(self,b_size):

        self.b_size=b_size
        self.storage=[]

    def add(self,*args):
        if len(self.storage)>=self.b_size:
            self.storage.pop(0)
        self.storage.append(args)

    def encode_sample(self,idxes):

        b_o,b_a,b_r,b_o_,b_d=[],[],[],[],[]

        for i in idxes:
            o,a,r,o_,d,w=self.storage[i]
            b_o.append(o)
            b_a.append(a)
            b_r.append(r)
            b_o_.append(o_)
            b_d.append(d)
        return(
            np.stack(b_o).astype('float32'),
            np.stack(b_a).astype('int32'),
            np.stack(b_r).astype('float32'),
            np.stack(b_o_).astype('float32'),
            np.stack(b_d).astype('int32')
        )    


    def get_sample(self,batch_size,n,e):
        indexes=range(len(self.storage))
        o,a,r,o_,d,w=zip(*self.storage)
        w=list(w)
        w=tf.constant(w)
        w=(tf.math.abs(w)+e)**n
        w=tf.nn.softmax(w).numpy()
        idxes=random.choices(range(len(self.storage)),weights=w,k=batch_size)
        return self.encode_sample(idxes)         
