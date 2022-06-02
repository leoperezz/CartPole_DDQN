from neuronal_network import *
from utils import *
import tensorflow as tf
from tensorflow.keras.models import load_model
import gym
import numpy as np
import time
time.clock=time.time
def main():
    
    env = gym.make('CartPole-v1')

    state_shape=env.observation_space.shape
    n_actions=env.action_space.n
    lr=0.001
    #tao=6
    tao=0.1
    batch_size=32
    epsilon=0.5
    network=DQN(n_actions,lr,tao,epsilon,batch_size)

    network.Q_network=load_model('Cart_Pole_v3_boltzman.h5')

    num_episodes=100

    for i in range(num_episodes):

        reward=0

        o=env.reset()

        while 1:
            o=tf.expand_dims(o,axis=0)
            a=np.argmax(network.Q_network(o)[0])
            o_,r,done,info=env.step(a)
            env.render()
            reward+=r
            o=o_

            if done:
                break

        print(f"episode:{i} reward:{reward}")    

main()
