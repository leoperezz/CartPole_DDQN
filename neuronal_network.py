import tensorflow as tf
from tensorflow.keras.layers import Conv2D,Dense,Flatten,Input,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import Adam,RMSprop
import numpy as np
import random
from utils import *

def QFunc(n_actions):
    #For images
    input_layer=Input((88,80,1))
    x=Conv2D(32,kernel_size=(8,8),strides=(4,4),padding='same',activation='relu')(input_layer)
    x=Conv2D(64,kernel_size=(4,4),strides=(2,2),padding='same',activation='relu')(x)
    x=Conv2D(64,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu')(x)
    x=Flatten()(x)
    x=Dense(512,activation='relu')(x)
    x=Dense(512,activation='relu')(x)
    x=Dense(n_actions,activation='linear')(x)
    model=Model(input_layer,x)
    return model

    
def QFunc2(n_actions):
    input_layer=Input((4))
    x=Dense(512,activation='relu',kernel_initializer='he_normal')(input_layer)
    x=Dense(256,activation='relu',kernel_initializer='he_normal')(x)
    x=Dense(64,activation='relu',kernel_initializer='he_normal')(x)
    x=Dense(n_actions,activation='linear')(x)
    model=Model(input_layer,x)
    return model

class DQN(tf.keras.Model):

    def __init__(self,n_actions,lr,tao,epsilon,batch_size):

        super(DQN,self).__init__()
        self.n_actions=n_actions #number of actions
        self.batch_size=batch_size
        self.tao=tao
        self.epsilon=epsilon
        self.optimizer=RMSprop(lr=lr,rho=0.95,epsilon=0.01)
        self.lr=lr
        self.Q_network=QFunc2(n_actions)
        self.T_network=QFunc2(n_actions)
    

    def sync(self):
        weights=self.Q_network.get_weights()
        self.T_network.set_weights(weights) 

    def boltzman_policy(self,state):
        state=tf.expand_dims(state,axis=0)
        values=self.Q_network(state)[0]
        values=tf.divide(values,tf.constant(self.tao,dtype=tf.float32))
        values=tf.nn.softmax(values).numpy()
        action=random.choices(list(range(self.n_actions)),weights=values,k=1)
        return action[0]

    def epsilon_policy(self,state):
        state=tf.expand_dims(state,axis=0)        
        r=np.random.sample()

        if r<self.epsilon:
            return random.choices(list(range(self.n_actions)),k=1)[0]
        else:
            return self.Q_network(state).numpy().argmax()    

    def update_tao(self,tao_decay):
        
        if self.tao>0.01:
            self.tao-=tao_decay

    def update_epsilon(self,eps_decay):
        if self.epsilon>0.1:
            self.epsilon-=eps_decay        

    def update_lr(self,lr_decay):
        if self.lr>1e-4:
            self.lr-=lr_decay
        self.optimizer=Adam(lr=self.lr)        


    def error_train(self,b_o,b_a,b_r,b_o_,b_d,type=True):
        gamma=0.95
        b_a_ = tf.one_hot(tf.argmax(self.Q_network(b_o_), 1), self.n_actions)
        b_q_=tf.cast((1-b_d),tf.float32)*tf.reduce_sum(self.T_network(b_o_)*b_a_,1)
        b_q = tf.reduce_sum(self.Q_network(b_o) * tf.reshape(tf.one_hot(b_a, self.n_actions),(-1,self.n_actions)), 1)
        yi=b_r+gamma*b_q_
        loss=mean_squared_error(yi,b_q)
        #loss=b_q-(b_r+gamma*b_q_)**2
        #loss=tf.divide(loss**2,tf.constant(self.batch_size,dtype=tf.float32))
        return loss    

    @tf.function
    def train(self,b_o,b_a,b_r,b_o_,b_d):

        with tf.GradientTape() as tape:
            loss=self.error_train(b_o,b_a,b_r,b_o_,b_d)
        grads=tape.gradient(loss,self.Q_network.trainable_weights)
        self.optimizer.apply_gradients(zip(grads,self.Q_network.trainable_weights))
        
        return loss

    def get_weights(self,o,a,r,o_,d):
        o=tf.cast(tf.expand_dims(o,axis=0),dtype='float32')        
        a=tf.cast(tf.expand_dims(a,axis=0),dtype='int32')        
        r=tf.cast(tf.expand_dims(r,axis=0),dtype='float32')        
        o_=tf.cast(tf.expand_dims(o_,axis=0),dtype='float32')        
        d=tf.cast(tf.expand_dims(d,axis=0),dtype='int32')        
        loss=self.error_train(o,a,r,o_,d)
        return loss.numpy()        



        
        


                    




        




