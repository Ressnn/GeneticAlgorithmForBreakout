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
import matplotlib.pyplot as plt
import pandas as pd
#import keyboard

# In[]
env = gym.make('Breakout-ram-v0')
a=env.reset()
env_shape = a.shape
env.reset()
#env = gym.wrappers.Monitor(env,'recording')
# In[]
'''
counter = 0
total = 0
env.reset()
b = False

for i in range(0,100):
    counter = 0
    while b != True:
        #env.render()
        #time.sleep(.03)
        r = env.step(np.random.randint(0,4))
        total = total+r[1]
        print(total)
        b = r[2]
        #time.sleep(.01)
        if counter>1000:
            b=True
        else:
            pass
    env.reset()
    b=False
env.close()

# In[]
env.reset()
while True:  
    env.render()
    try:  
        if keyboard.is_pressed('a'): 
            env.step(3)# finishing the loop
            time.sleep(.1)
        elif keyboard.is_pressed('d'):
            env.step(2)
            time.sleep(.1)
        elif keyboard.is_pressed('w'):
            env.step(1)
            time.sleep(.1)
    except:
        break  
        '''
 # In[]

class Species():
    def __init__(self):

        self.Condenser = Sequential()
        self.Condenser.add(Lambda(lambda x: x/255))
        self.Condenser.add(Dense(100,activation='relu'))
        self.Condenser.add(Dense(75,activation='relu'))
        self.Condenser.add(Dense(50,activation='relu'))
        self.Condenser.add(Dense(25,activation='tanh'))
        self.Condenser.compile(loss='mse',optimizer='adam')
        
        self.MemFixer = Sequential()
        self.MemFixer.add(Dense(50,activation='relu'))
        self.MemFixer.add(Dense(40,activation='relu'))
        self.MemFixer.add(Dense(35,activation='relu'))
        self.MemFixer.add(Dense(25,activation='tanh'))
        self.MemFixer.compile(loss='mse',optimizer='adam')
        
        self.Decider = Sequential()
        self.Decider.add(Dense(25,activation='relu'))
        self.Decider.add(Dense(50,activation='relu'))
        self.Decider.add(Dense(25,activation='relu'))
        self.Decider.add(Dense(4,activation='sigmoid'))
        self.Decider.compile(loss='sparse_categorical_crossentropy',optimizer='adam')
        
        self.Condenser.fit(np.zeros((1,128)),np.zeros((1,25)))
        self.MemFixer.fit(np.zeros((1,50)),np.zeros((1,25)))
        self.Decider.fit(np.zeros((1,25)),np.zeros((1,1)))
        
    def getW(self):
        self.W_list = []
        self.W_list.append(self.Condenser.get_weights())
        self.W_list.append(self.MemFixer.get_weights())
        self.W_list.append(self.Decider.get_weights())
        return self.W_list
    
    
    def SetW(self,W_list):
        self.Condenser.set_weights(W_list[0])
        self.MemFixer.set_weights(W_list[1])
        self.Decider.set_weights(W_list[2])
        
        
    def randomW(self):
        self.NW = []
        self.W=self.getW()
        for i in self.W:
            nl = []
            for a in i:
                shape = a.shape
                nl.append(np.random.uniform(low=-1,high=1,size=(shape)))
            self.NW.append(nl)
            
        self.Condenser.set_weights(self.NW[0])
        self.MemFixer.set_weights(self.NW[1])
        self.Decider.set_weights(self.NW[2])
        
        
    def randomW2(self):
        self.Condenser.fit(np.random.normal(size=(4,env_shape[0],env_shape[1],env_shape[2])),np.random.normal(size=(4,100)),epochs=10)
        self.MemFixer.fit(np.random.normal(size=(8,200)),np.random.normal(size=(8,100)),epochs=10)
        self.Decider.fit(np.random.rand(16,100),np.random.randint(0,4,(16,1)),epochs=10)
        return True
    
    
    def randomW3(self):
        total = 0
        env.reset()
        b = False
        game_state_list = []
        action_state_list = []
        while b != True:
            #env.render()
            #time.sleep(.03)
            action_state_list.append(np.random.randint(0,4))
            r = env.step(action_state_list[-1])
            game_state_list.append(r[0])
            total = total+r[1]
            #print(total)
            b = r[2]
        v = np.random.normal(size=(len(action_state_list),100))
        r = np.random.normal(size=(len(action_state_list),100))
        self.Condenser.fit(np.array(game_state_list).reshape(-1,210,160,3),r,verbose=False)
        self.MemFixer.fit(np.concatenate((v,r),1),v,epochs=10,verbose=False)
        self.Decider.fit(v,np.random.randint(0,4,(len(action_state_list),1)),verbose=False)
        
        
    def test_model(self,environment):
        sc = []
        self.action_list = []
        self.e = environment
        for z in range(0,1):
            IMG_list = []
            self.MemList = np.zeros((1,25))
            self.counter = 0
            screen_state = environment.reset()
            self.val = False
            self.score = 0
            self.acu = []
            while self.val != True:
                #time.sleep(.01)
                r = self.Condenser.predict(screen_state.reshape(1,-1))
                self.CList = np.concatenate((r.reshape(-1),self.MemList.reshape(-1))).reshape(1,-1)
                pred = self.MemFixer.predict(self.CList)
                p2 = self.Decider.predict_classes(pred)
                self.action_list.append(p2.tolist()[0])
                if self.counter%100 == 0:
                    k = environment.step(p2)
                else:
                    k = environment.step(1)
                screen_state = k[0]
                IMG_list.append(screen_state)
                self.score = self.score+k[1]
                self.val = k[2]
                self.counter = self.counter+1
                if self.counter==2000:
                    self.val=True
                #self.e.render()
                x = 0
                if self.counter>210:
                    nx_list = self.action_list[-150:-1]
                    first = nx_list[0]
                    for i in nx_list:
                        if i == first:
                            x = x+1
                    if x>140:
                        sc.append(0)
            sc.append(self.score)
            
            if self.score>= 15:
                #time.sleep(.01)
                IMG_list = np.array(IMG_list)
                np.save('5',IMG_list)
                for z in range(0,6):
                    IMG_list = []
                    self.MemList = np.zeros((1,25))
                    self.counter = 0
                    screen_state = environment.reset()
                    self.val = False
                    self.score = 0
                    while self.val != True:
                        r = self.Condenser.predict(screen_state.reshape(1,-1))
                        self.CList = np.concatenate((r.reshape(-1),self.MemList.reshape(-1))).reshape(1,-1)
                        pred = self.MemFixer.predict(self.CList)
                        p2 = self.Decider.predict_classes(pred)
                        if self.counter%10 != 0:
                            k = environment.step(p2)
                        else:
                            k = environment.step(1)
                        screen_state = k[0]
                        IMG_list.append(screen_state)
                        self.score = self.score+k[1]
                        self.val = k[2]
                        self.counter = self.counter+1
                        if self.counter==400:
                            self.val=True
                        self.e.render()
                    sc.append(self.score)
            if self.score>=15:
                IMG_list = np.array(IMG_list)
                np.save('9',IMG_list)
        return np.mean(sc)

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
            self.S.randomW()
            self.losslist.append(self.S.test_model(self.environment))
            self.species.append(self.S.getW())
        print(self.losslist)
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
                    if np.random.random()>= (np.random.random()*self.mt_chance):
                        LW.append(k)
                    else:
                        LW.append(np.random.uniform(-self.mt_amount,self.mt_amount))
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
        self.score_list2 = []
        for i in range(0,self.sp_size):
            self.sortlist.append(st[i][0])
        return st
    def sort2(self,genomes,Scores,base):
        subset_score_list = Scores.copy()
        num_list = np.ones(len(Scores))
        final_sorted = []
        counter = 0
        fcl = []
        for i in genomes:
            fdl = []
            for a in genomes:
                dist_list = []
                for idx in range(3):
                    for z in range(len(i[idx])):
                        c = np.array(a[idx][z]).reshape(-1)
                        v = np.array(i[idx][z]).reshape(-1)
                        dist_list.append(np.linalg.norm(c-v))
                fdl.append(np.mean(dist_list))
            fcl.append(np.mean(fdl))
            self.fcl = fcl
            counter = 0
            fdsort = []
            
            for i in range(len(fdl)):
                fdsort.append([fdl[i],i])
            fdsort = sorted(fdsort,key=itemgetter(0),reverse=False)

            # This part shares the score with others in a similar vectorspace
            subset_score_list = Scores.copy()
            for i in range(30):
                selection = fdsort[i]
                close, selection = selection
                subset_score_list[selection] += Scores[counter]
                num_list[selection] += 1
            final_sorted.append(np.divide(subset_score_list,num_list))
            counter = counter+1
        final_sorted = np.array(final_sorted)
        self.fs = final_sorted
        final_sorted = np.mean(final_sorted,0)
        self.st = final_sorted
        sortme = []
        self.sortlist = []
        for i in range(len(Scores)):
            sortme.append([genomes[i],final_sorted[i]])
        sortme = sorted(sortme,key=itemgetter(1),reverse=True)
        for i in sortme:
            self.sortlist.append(i[0])
        
        return sortme
    
    def ranking_func(self):
        number_of_species = self.sp_size
        value_list = []
        for i in range(number_of_species):
            i = i+1
            r = (75/100)**i
            value_list.append(r)
        value_list = np.array(value_list).reshape(-1,1)
        scaled = self.Scaler.fit_transform(value_list)
        scaled = scaled.reshape(-1)
        ttl = 0
        for i in scaled:
            ttl = ttl+i
        scaled = scaled/ttl
        return scaled
    
    def generate_new_rand(self):
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
        W1 = self.sortlist[p1]
        W2 = self.sortlist[p2]
        self.W3 = W1
        for a in range(100):
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
    

    # Thanks to https://towardsdatascience.com/atari-solving-games-with-ai-part-2-neuroevolution-aac2ebb6c72b for this peice of code
    def crossoverv2(self,x, y):
        t_list = []
        x1 = self.sortlist[x]
        y1 = self.sortlist[y]
        
        for i in range(3):
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
        self.sorted_w_percents = self.sort2(self.species,self.losslist,.1)
        if self.sorted_w_percents[0][1]>=4:
            self.bestW.append(self.sorted_w_percents[0])
        parents_list = []
        for i in range((self.sp_size)):
            parents_list.append(self.generate_new_rand())
        parents_list = np.array(parents_list).reshape(-1,2)
        self.K = parents_list
        Kid_W = []
        for i in range(int(self.sp_size/2)):
            Kid_W.append(self.mutate(.06,1,self.crossoverv2(parents_list[i][0],parents_list[i][1])))
            Kid_W.append(self.sorted_w_percents[i][0])
        losslist = []
        for i in Kid_W:
            self.S.SetW(i)
            global env
            losslist.append(self.S.test_model(env))
        self.losslist = losslist
        print(max(self.losslist))
        self.species = Kid_W
        return self.losslist
    
# In[]
#Spec = Species()
# In[]
ll = []
E = Evolution(.01,1,50,env,1)
# In[]
for i in range(0,10000):
    ll.append(E.Epoch())
    print("printing")
    print(ll[-1])



# In[]

ln = np.array(ll)
nl = []
print(np.max(ln))
m = []
mx = []
mn = []
for i in ln:
    m.append(np.average(i))
    mx.append(np.max(i))
    mn.append(np.min(i))
    
    
plt.plot(m)
# In[]

arr = np.concatenate((np.array(m).reshape(-1,1),np.array(mx).reshape(-1,1),np.array(mn).reshape(-1,1)),1)[1:]
plt.xlabel("Generations")
plt.xlabel("Score")
plt.plot(arr[:,0],label="Mean")
plt.plot(arr[:,1],label="Max")
# In[]
def movingaverage(l,s):
    avg_list = []
    for i in range(len(l)-s):
        avg = l[i:i+s]
        avg_list.append(np.average(avg))
    return avg_list
plt.plot(movingaverage(m,10))
# In[]
S2 = Species()
testr = []
for i in range(100):
    L2 = LE[i][0]
    S2.SetW(L2)
    testr.append(S2.test_model(env))