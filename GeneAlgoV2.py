# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 17:59:47 2019

@author: Pranav Devarinti
"""

import gym
import time
import keras
from keras.models import Sequential
from keras.layers import *
import numpy as np
import matplotlib.pyplot as plt
import time
from operator import itemgetter
import numpy.random as random
from sklearn.preprocessing import MinMaxScaler
# In[]
env = gym.make('Atlantis-v0')
a=env.reset()
env_shape = a.shape
env.reset()
# In[]
total = 0
env.reset()
b = False
while b != True:
    env.render()
    #time.sleep(.03)
    r = env.step(np.random.randint(0,4))
    total = total+r[1]
    print(total)
    b = r[2]
 # In[]

class Species():
    def __init__(self):
        self.Condenser = Sequential()
        self.Condenser.add(Lambda(lambda x: x/255))
        self.Condenser.add(Conv2D(5,(5,5),strides=(2,2),kernel_initializer='random_uniform',bias_initializer='random_uniform'))
        self.Condenser.add(Conv2D(10,(5,5),strides=(2,2),activation='relu',kernel_initializer='random_uniform',bias_initializer='random_uniform'))
        self.Condenser.add(Conv2D(15,(5,5),strides=(2,2),kernel_initializer='random_uniform',bias_initializer='random_uniform'))
        self.Condenser.add(Conv2D(10,(5,5),strides=(2,2),activation='relu',kernel_initializer='random_uniform',bias_initializer='random_uniform'))
        self.Condenser.add(Flatten())
        self.Condenser.add(Dense(600,kernel_initializer='random_uniform',bias_initializer='random_uniform'))
        self.Condenser.add(Dense(500,activation='relu',kernel_initializer='random_uniform',bias_initializer='random_uniform'))
        self.Condenser.add(Dense(300,kernel_initializer='random_uniform',bias_initializer='random_uniform'))
        self.Condenser.add(Dense(4,activation='softmax',kernel_initializer='random_uniform',bias_initializer='random_uniform'))
        self.Condenser.compile(loss='sparse_categorical_crossentropy',optimizer='adam')
        total = 0
        env.reset()
        b = False
        game_state_list = []
        action_state_list = []
        while b != True:
            env.render()
            #time.sleep(.03)
            action_state_list.append(np.random.randint(0,4))
            r = env.step(action_state_list[-1])
            game_state_list.append(r[0])
            total = total+r[1]
            print(total)
            b = r[2]
        self.a = action_state_list
        self.Condenser.fit(np.array(game_state_list).reshape(-1,210,160,3),np.array(action_state_list),epochs=1)
    def getW(self):
        self.W_list = []
        self.W_list.append(self.Condenser.get_weights())
        return self.W_list
    
    def SetW(self,W_list):
        self.Condenser.set_weights(W_list[0])
        
    def randomW(self):
        self.NW = []
        self.W=self.getW()
        for i in self.W:
            nl = []
            for a in i:
                shape = a.shape
                nl.append(((np.random.sample(size=(shape)))*2))
            self.NW.append(nl)
        self.Condenser.set_weights(self.NW[0])
    
    def randomW2(self):
        self.Condenser.fit(np.random.normal(size=(400,env_shape[0],env_shape[1],env_shape[2])),np.random.randint(0,4,size=(400,1)),epochs=10)
        return True

        
    def test_model(self,environment):
        self.e = environment
        self.MemList = np.zeros((1,100))
        self.counter = 0
        screen_state = environment.reset()
        self.val = False
        self.score = 0
        while self.val != True:
            pred = self.Condenser.predict_classes(screen_state.reshape(1,210,160,3))
            r = env.step(pred)
            env.render()
            print(pred)
            screen_state = r[0]
            self.score = self.score+r[1]
            self.val = r[2] 
            self.counter = self.counter+1
            if self.counter>1000:
                self.val=True
            
        return self.score

# In[]
class Evolution():
    def __init__(self,mt_chance,mt_amount,speciessize,environment,epochs):
        self.species = []
        self.losslist = []
        self.bestW=[]
        self.environment = environment
        self.S = Species()
        self.epochs = epochs
        # Mt_amount for one standard deviation
        self.Scaler = MinMaxScaler()
        self.mt_amount = mt_amount
        self.mt_chance = mt_chance
        self.sp_size = speciessize
        for i in range(0,speciessize):
            self.S = Species()
            self.S.randomW()
            self.losslist.append(self.S.test_model(self.environment))
            self.species.append(self.S.getW())
            print(self.losslist[-1])
        print('Started Models')
    def mutate(self,chance,amount,Wheights):
        FWL = []
        for a in Wheights:
            Per_Cat_W = []
            for i in a:
                LW = []
                return_shape = i.shape
                i = i.reshape(-1)
                for k in i:
                    if np.random.random()>=self.mt_chance:
                        LW.append(k)
                    else:
                        LW.append(np.random.normal(self.mt_amount)+k)
                Per_Cat_W.append(np.array(LW).reshape(return_shape))
            FWL.append(Per_Cat_W)
        return FWL
    def sort(self,genome,Scores,base_to_add):
        new_list = []
        total = 0
        for i in range(len(Scores)):
            new_list.append([genome[i],Scores[i]])
            total = total+Scores[i]+base_to_add
        st = sorted(new_list,key=itemgetter(1),reverse=True)
        self.sortlist = []
        for i in range(0,self.sp_size):
            self.sortlist.append(st[0])
        return st
    def ranking_func(self):
        number_of_species = self.sp_size
        value_list = []
        for i in range(number_of_species):
            i = i+1
            r = (3/4)**i
            value_list.append(r)
        value_list = np.array(value_list).reshape(-1,1)
        scaled = self.Scaler.fit_transform(value_list)
        scaled = scaled.reshape(-1)
        ttl = 0
        for i in scaled:
            ttl = ttl+i
        scaled = scaled/ttl
        return scaled
    
    def generate_new_rand(self,Sorted_list):
        '''rand = np.random.random()
        stop = False
        self.to_pick = 0
        for a in range(len(Percents)):
            i = Percents[a]
            if i >= rand and stop == False:
                stop = True
                self.to_pick = a'''
        ranks = self.ranking_func()
        pick = []
        for i in range(0,self.sp_size):
            pick.append(i) 
        return np.random.choice(pick,p=ranks)
        
        return self.to_pick
    def Crossover(self,p_list):
        p1,p2 = p_list
        print(p1)
        W1 = self.sortlist[p1-1]
        W2 = self.sortlist[p2-1]
        self.W3 = W1
        for a in range(1):
            sample = W1[a]
            self.W3_part = []
            for i in range(len(sample)):
                random = np.random.random()
                if random >= .5:
                    self.W3_part.append(W1[a][i])
                else:
                    self.W3_part.append(W2[a][i])
            self.W3.append(self.W3_part)            
        return self.W3
    
    def crossoverv2(self,x, y):
        t_list = []
        x1 = self.species[x-1]
        y1 = self.species[y-1]
        
        for i in range(1):
            offspring_x = x1[i]
            offspring_y = y1[i]
    
            for a in range(0, len(offspring_x)):  # 10
                a_layer = offspring_x[a]
                for b in range(0, len(a_layer)):  # 8
                    b_layer = a_layer[b]
                    if not isinstance(b_layer, np.ndarray):
                        if random.choice([True, False]):
                            offspring_x[a][b] = offspring_y[a][b]
                            offspring_y[a][b] = offspring_x[a][b]
                        continue
                    for c in range(0, len(b_layer)):  # 8
                        c_layer = b_layer[c]
                        if not isinstance(c_layer, np.ndarray):
                            if random.choice([True, False]):
                                offspring_x[a][b][c] = offspring_y[a][b][c]
                                offspring_y[a][b][c] = offspring_x[a][b][c]
                            continue
                        for d in range(0, len(c_layer)):  # 4
                            d_layer = c_layer[d]
                            for e in range(0, len(d_layer)):  # 32
                                if random.choice([True, False]):
                                    try:
                                        offspring_x[a][b][c][d][e] = offspring_y[a][b][c][d][e]
                                        offspring_y[a][b][c][d][e] = offspring_x[a][b][c][d][e]
                                    except:
                                        pass
            t_list.append(offspring_x)           
        return t_list
            

    def Epoch(self):
        self.sorted_w_percents = self.sort(self.species,self.losslist,.75)
        self.bestW.append(self.sorted_w_percents[0])
        parents_list = []
        for i in range(self.sp_size*2):
            parents_list.append(self.generate_new_rand(self.sorted_w_percents[-1]))
        parents_list = np.array(parents_list).reshape(-1,2)
        self.K = parents_list
        Kid_W = []
        for i in parents_list:
            Kid_W.append(self.mutate(.001,.1,self.crossoverv2(i[0],i[1])))
        losslist = []
        for i in Kid_W:
            self.S.SetW(i)
            global env
            losslist.append(self.S.test_model(env))
        self.losslist = losslist
        print(max(self.losslist))
        self.species = Kid_W
        
# In[]
#Spec = Species()



# In[]
E = Evolution(.01,1,25,env,1)
# In[]
for i in range(0,100):
    E.Epoch()