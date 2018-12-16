import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from keras import backend as k
from keras.backend import clear_session
from collections import deque
import random
import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import gc

#import keras.backend.tensorflow_backend
#from keras.backend import clear_session


class NumberData:
    def __init__(self, value, x, y):
        self.yCoord = y
        self.value = value
        self.xCoord = x


class NumberCatchingAI:

    def __init__(self, turns):
        self.maxTurns = turns
        self.currentTurn = 0

        self.score = 0
        self.numberInfo = deque()

        self.playerLocation = 8

        self.numCols = 15
        self.numRows = 25

        self.discountFactor = 0.9
        
        self.epsilon = 0.1

        for i in range(5):
            self.numberInfo.append(NumberData(random.randint(1,9), random.randint(0,14), i * 5))




        exInput = tf.convert_to_tensor(self.getState())

        self.policyFunction = keras.Sequential([
            keras.layers.Dense(64, input_shape=exInput.shape, activation=tf.nn.tanh),
            keras.layers.Dense(64, activation=tf.nn.tanh),
            keras.layers.Dense(3, activation=tf.nn.softmax)
        ])
        #self.policyFunction.summary()

        self.policyFunction.compile(loss='mse',optimizer=keras.optimizers.SGD(lr=1e-5, clipnorm=10))




        self.valueFunction = keras.Sequential([
            keras.layers.Dense(64, input_dim=16, activation=tf.nn.tanh),
            keras.layers.Dense(64, activation=tf.nn.tanh),
            keras.layers.Dense(1)
        ])

        self.valueFunction.compile(loss='mse', optimizer=keras.optimizers.SGD(lr=1e-4, clipnorm=10))

        self.policyFunction.save("currentPolicy.h5")
        self.policyFunction.save("oldPolicy.h5")




    
    def getState(self):
        ret = []
        for i in range(len(self.numberInfo)):
            ret.append(self.numberInfo[i].value)
            ret.append(self.numberInfo[i].xCoord)
            ret.append(self.numberInfo[i].yCoord)
        ret.append(self.playerLocation)
        return ret

    def getBestAction(self):
        start = time.time()
        
        #self.policyFunction.load_weights("oldPolicy.h5")
        keras.backend.clear_session()
        self.policyFunction = keras.models.load_model("oldPolicy.h5")
        

        #keras.backend.clear_session()
        #if keras.backend.sess:
            #tf.reset_default_graph()
            #keras.backend._SESSION.close()
            #keras.backend._SESSION = None
        
        state = self.getState()
        p = []
        for i in range(16):
            p.append(1)

        inp = tf.convert_to_tensor(p)

        inp = tf.stack([inp,tf.convert_to_tensor(state)])

        out = self.policyFunction.predict(inp, steps=1)
        
        n = random.random()
        #print "Time for one prediction = " + str(time.time() - start)
        index = 0
        for i in np.nditer(out):
            if(n < i):
                return index - 1
            else:
                n -= i

            index += 1
        return -1
    

    def listToTensor(self, list):
        p = []
        for i in range(len(list)):
            p.append(1)
        
        r = tf.convert_to_tensor(p)

        r = tf.stack([r, tf.convert_to_tensor(list)])

        return r


    def drawGame(self):

        mat = [[0 for x in range(self.numCols)] for y in range(self.numRows)]

        for i in range(len(self.numberInfo)):
            val = str(self.numberInfo[i].value)
            mat[self.numberInfo[i].yCoord][self.numberInfo[i].xCoord] = val
        
        #player location
        mat[self.numRows - 1][self.playerLocation] = "U"


        for r in range(self.numRows):
            sys.stdout.write("|")
            for c in range(self.numCols):
                if (mat[r][c]) == 0:
                    sys.stdout.write(" |")
                else:
                    sys.stdout.write(str(mat[r][c]) + "|")
            print ""
        
        print "Score: " + str(self.score)
        print "Turn " + str(self.currentTurn) + "/" + str(self.maxTurns)
        print "Best action = " + str(self.getBestAction())



    def setState(self, state):
        # number of numbers in the queue
        self.numberInfo.clear()
        
        #print "length = " + str(len(state))
        for i in range(0, 15, 3):
            data = NumberData(state[i], state[i + 1], state[i + 2])
            self.numberInfo.append(data)

        self.playerLocation = state[ len(state) - 1]



    def rewardFunction(self, state, action):
        invalidMoveReward = -5

        #keep old data here for later use
        oldState = self.getState()
        oldScore = self.score
        oldTurns = self.currentTurn
        self.setState(state)
        r = 0

        #check for bad moves
        if (self.playerLocation == 14 and action == 1):
            r += invalidMoveReward
        elif(self.playerLocation == 0 and action == -1):
            r += invalidMoveReward
        
        #now check to see if any of the numbers are caught or not
        #remember, the last item in number info is the closest to the bottom

        if(self.playerLocation == 14 and action == 1):
            self.playerLocation = 14
        elif(self.playerLocation == 0 and action == -1):
            self.playerLocation = 0
        else:
            self.playerLocation += action

        for i in range(len(self.numberInfo)):
            self.numberInfo[i].yCoord +=1
        

        if(self.numberInfo[len(self.numberInfo) - 1].yCoord == 24):
            if(self.numberInfo[len(self.numberInfo) - 1].xCoord == self.playerLocation):
                #number caught
                r += self.numberInfo[len(self.numberInfo) - 1].value
            else:
                #number not caught
                r -= self.numberInfo[len(self.numberInfo) - 1].value
            
          
        
        #reset state
        self.setState(oldState)
        self.score = oldScore
        self.currentTurn = oldTurns
        return r
            

    def playGameHuman(self):

        while( self.currentTurn <= self.maxTurns):
            #draw game
            self.drawGame()
            #ask for input
            i = raw_input()
            action = 0

            if(i == "a"):
                action = -1
            elif(i == "d"):
                action = 1


            #perform the action
            self.score += self.rewardFunction(self.getState(), action)
            self.performAction(action)

            



    def performAction(self, action):

        self.score += self.rewardFunction(self.getState(), action)
        #move player
        if(self.playerLocation == 14 and action == 1):
            self.playerLocation = 14
        elif(self.playerLocation == 0 and action == -1):
            self.playerLocation = 0
        else:
            self.playerLocation += action

        #update numbers
        for i in range(len(self.numberInfo)):
            self.numberInfo[i].yCoord += 1
            if(self.numberInfo[i].yCoord == 24):
                #remove number and add a new one
                self.numberInfo.pop()

                #create new number and stick it on the front of the list
                n = NumberData(random.randint(1,9), random.randint(0,14), 0)
                self.numberInfo.appendleft(n)

        self.currentTurn += 1


    def probRatio(self, state, action):
        #keras.backend.clear_session()
        #load old
        self.policyFunction = keras.models.load_model("oldPolicy.h5")

        stateTensor = self.listToTensor(state)
            

        out = self.policyFunction.predict(stateTensor, steps=1)
        #print tf.shape(out)
        probOld = out[1][action + 1]

        #keras.backend.clear_session()
        #load current
        self.policyFunction = keras.models.load_model("currentPolicy.h5")
        out = self.policyFunction.predict(stateTensor, steps=1)
        probNew = out[1][action + 1]

        return probNew / probOld



    def trainPPO(self, state, action, reward):
        ratio = self.probRatio(state, action)

        #how well did the policy work compared to before? (advantage)
        advantage = reward - self.valueFunction.predict(self.listToTensor(state), steps=1)[0][0]

        if(ratio > 1 + self.epsilon):
            #always demote action
            obj = [1,1,1]
            obj[action + 1] = 0

        elif(ratio < 1 - self.epsilon):
            #always promote action
            obj = [0,0,0]
            obj[action + 1] = 1

        elif(advantage > 0):
            #better results than before -> promote
            obj = [0,0,0]
            obj[action + 1] = 1

        else:
            #worse results than before -> demote
            obj = [1,1,1]
            obj[action + 1] = 0

        #train with new objective
        stateTensor = self.listToTensor(state)
        probTensor = self.listToTensor(obj)
        self.policyFunction.fit(stateTensor, probTensor, epochs=1, steps_per_epoch=1, verbose=0)
        self.policyFunction.save("currentPolicy.h5")




    def trainAI(self, iterations, timesteps):
        
        

        #for each iteration
        for iter in range(iterations):
            startTime = time.time()
            rewards = []
            actions = []
            states = [[0 for x in range(16)] for y in range(1)]
            values = [[0 for x in range(1)] for y in range(1)]
            #run for timesteps
            print "Collecting data..."
            for t in range(timesteps):
                #run old policy
                states.append(self.getState())
                actions.append(self.getBestAction())
                rewards.append(self.rewardFunction(states[t], actions[t]))
                self.performAction(actions[t])
            
            #data collected, now to be analyzed
            #get values
            for t in range(timesteps):
                values.append([self.cumulativeReward(t, rewards)])
            
            #train the policy function
            print "Training policy..."
            for i in range(200):
                rand = random.randint(0, len(actions) - 1)
                self.trainPPO(states[rand], actions[rand], values[rand][0])
            
            
            
            #train the value function
            self.valueFunction.fit(np.array(states), np.array(values), epochs=1, steps_per_epoch= 30 * timesteps)

            #print out stats
            print "Iteration: " + str(iter)
            print "AI score: " + str(self.runGameAI())
            self.policyFunction.save("oldPolicy.h5")
            print "Iteration took " + str((time.time() - startTime)) + " seconds"

            gc.collect()

            

    def runGameAI(self):
        self.score = 0
        for i in range(200):
            keras.backend.clear_session()
            self.policyFunction = keras.models.load_model("oldPolicy.h5")
            action = self.getBestAction()
            self.performAction(action)
        
        return self.score
            

    def cumulativeReward(self, timestep, rewards):
        ret = 0

        for i in range(timestep, len(rewards)):
            ret += (self.discountFactor ** (i - timestep)) * rewards[i]
        
        return ret






n = NumberCatchingAI(200)
n.trainAI(100, 1000)
