from utils import *
from neuronal_network import *
import tensorflow as tf
from tensorflow.keras.models import save_model
import matplotlib.pyplot as plt
import numpy as np
import gym
import time



time.clock=time.time

def main():

    x=input(f'What policy?: \na)epsilon\nb)boltzman\n\nAnswer: ')
    if x=='a':
        x=1
    elif x=='b':
        x=0    


    env = gym.make('CartPole-v1')

    n_actions=env.action_space.n

    #Hyperparameters
    time_steps=1
    num_episodes=5000
    lr=0.00025
    lr_decay=1e-6-5e-9
    tao=9
    tao_decay=0.00098
    epsilon=0.4
    epsilon_decay=0.0001
    b_size=1000
    batch_size=32
    train_freq=1
    sync_steps=100
    save_episodes=50
    n=3
    e=0.01


    #Buffer and Network instance
    network=DQN(n_actions,lr,tao,epsilon,batch_size)
    buffer=Buffer(b_size)

    #network.Q_network.load_weights('C:\ARCHIVOS DE LEO\Inteligencia Artificial\Reinforcement Learning\Codigo\RL springer\Capitulo 4\DQN\CartPole.h5')

    total_rewards=[]

    for episode in range(num_episodes):

        o=env.reset()
        reward=0
        loss_t=0
        count=0
        while 1:
            if x:
                a=network.epsilon_policy(o)
            else:
                a=network.boltzman_policy(o)    
            o_,r,done,info=env.step(a)
            env.render()
            reward+=r

            if done:
                #r=-100
                d=1
            else:
                #r=r
                d=0
            w=network.get_weights(o,a,r,o_,d)
            buffer.add(o,a,r,o_,d,w)
            
            if time_steps>32 and time_steps%train_freq==0:
                sample=buffer.get_sample(batch_size,n,e)
                loss=network.train(*sample)
                loss_t+=loss

            if time_steps%sync_steps==0:
                network.sync()    
                
            o=o_  
            time_steps += 1


            if done:
                total_rewards.append(reward)
                break
            count+=1
        loss_t/=count   

        print(f'EPISODIO: {episode+1} REWARD:{reward} loss:{loss_t}' )
        if reward>450:
            if x:
                save_model(network.Q_network,'Cart_Pole_v3_epsilon.h5')
            else:
                save_model(network.Q_network,'Cart_Pole_v3_boltzman.h5')

        network.update_tao(tao_decay)
        network.update_epsilon(epsilon_decay) 
        #network.update_lr(lr_decay)

    total_rewards=np.array(total_rewards)
    plt.plot(total_rewards)
    plt.show()     

if __name__ == '__main__':
    main()