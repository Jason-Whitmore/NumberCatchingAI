import os
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from collections import deque
import random
import sys
import numpy as np
import math
import matplotlib.pyplot as plt 


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



        #PPO hyperparameters
        self.discountFactor = 0.95

        self.horizon = 50

        self.GAELambda = 0.95
        
        self.epsilon = 0.2

        for i in range(5):
            self.numberInfo.append(NumberData(random.randint(1,9), random.randint(0,14), i * 5))


        #Other hyperparameters
        
        #Always 2 hidden layers
        layerSize = 64

        valueLR = 3e-4
        policyLR = 3e-2



        self.policyFunction = Sequential()

        self.policyFunction.add(Dense(layerSize, activation='tanh', input_dim=16))
        
        self.policyFunction.add(Dense(layerSize, activation='tanh'))

        self.policyFunction.add(Dense(3))
        self.policyFunction.add(Activation('softmax'))


        self.policyFunction.compile(optimizer=keras.optimizers.SGD(lr=policyLR), loss="mse")





        self.valueFunction = Sequential()

        self.valueFunction.add(Dense(64, activation='tanh', input_dim=16))
        self.valueFunction.add(Dense(64, activation='tanh'))
        self.valueFunction.add(Dense(1))
        self.valueFunction.add(Activation('linear'))

        self.valueFunction.compile(loss='mse',optimizer=keras.optimizers.SGD(lr=valueLR))



    
    def getState(self):
        ret = []
        for i in range(len(self.numberInfo)):
            ret.append(self.numberInfo[i].value)
            ret.append(self.numberInfo[i].xCoord)
            ret.append(self.numberInfo[i].yCoord)
        ret.append(self.playerLocation)
        return ret

    def getBestAction(self):
        
        state = self.getState()

        out = self.policyFunction.predict(np.array([state]))
        
        n = random.random()
        index = 0
        for i in np.nditer(out):
            if(n < i):
                return index - 1
            else:
                n -= i

            index += 1
        return -1
    

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

        #keep old data here for later use
        oldState = self.getState()
        oldScore = self.score
        oldTurns = self.currentTurn
        self.setState(state)
        r = 0

        #check for bad moves
        
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
                r = self.numberInfo[len(self.numberInfo) - 1].value
            else:
                #number not caught
                r = -1 * self.numberInfo[len(self.numberInfo) - 1].value
            

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

        #self.score += self.rewardFunction(self.getState(), action)
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


    def runGameAI(self):
        self.score = 0
        for i in range(200):
            action = self.getBestAction()
            self.score += self.rewardFunction(self.getState(), action)
            self.performAction(action)
            
        
        return self.score
    
    def runGameAISamples(self, numSamples):
        scores = []
        for i in range(numSamples):
            scores.append(self.runGameAI())

        return np.mean(scores)

    def cumulativeReward(self, timestep, rewards):
        ret = 0

        for t in range(timestep, min(len(rewards), len(rewards) + self.horizon)):
            ret += (self.discountFactor ** (t - timestep)) * rewards[t]
        
        return ret


    def getValue(self, state):
        return (self.valueFunction.predict(np.array([state])))[0][0]

    def calculateGAEDeltas(self, rewards, states):
        ret = []

        for t in range(len(rewards) - 1):
            s = rewards[t] + self.discountFactor * self.getValue(states[t + 1]) - self.getValue(states[t])
            ret.append(s)
        
        return ret

    def calculateAdvantage(self, timestep, GAEDeltas):
        r = 0
        for t in range(timestep, len(GAEDeltas)):
            r += (self.GAELambda * self.discountFactor)**(t - timestep) * GAEDeltas[t]

        return r


    def calculateSoftmaxTarget(self, oldSoftmax, action, advantage):
        ret = [0,0,0]

        probNonActions = 0

        for i in range(len(oldSoftmax)):
            if i != action + 1:
                probNonActions += oldSoftmax[i]
            
            ret[i] = oldSoftmax[i]

        
        if advantage > 0:
            delta = (oldSoftmax[action + 1] * (1 + self.epsilon)) - oldSoftmax[action + 1]

            ret[action + 1] = oldSoftmax[action + 1] + delta

            for i in range(len(ret)):
                if i != action + 1:
                    ret[i] -= (oldSoftmax[i] / probNonActions) * delta

        else:
            delta = oldSoftmax[action + 1]  -  (oldSoftmax[action + 1] * (1 - self.epsilon))

            ret[action + 1] = oldSoftmax[action + 1] - delta

            for i in range(len(ret)):
                if i != action + 1:
                    ret[i] += (oldSoftmax[i] / probNonActions) * delta
        
        return ret


    def getProbability(self, state, action):
        return self.policyFunction.predict(np.array([state]))[0][action + 1]


    def getAvgProbability(self, states, actions):
        r = 0

        for i in range(len(states)):
            r += self.getProbability(states[i], actions[i])

        return r / len(states)

    def getRunningAverage(self, data, n):
        if n >= len(data):
            r = 0

            for i in range(len(data)):
                r += data[i]
            return float(r) / len(data)

        r = 0

        for i in range(len(data) - n, len(data)):
            r += data[i]

        return float(r) / len(data)


    def getAvgAbsAdvantage(self, advantages):
        r = 0

        for i in range(len(advantages)):
            r += abs(advantages[i])

        return r / len(advantages)

    def playGameAI(self, policyName):

        self.policyFunction = keras.models.load_model(policyName)

        for t in range(200):

            self.drawGame()
            i = raw_input()

            action = self.getBestAction()
            self.score += self.rewardFunction(self.getState(), action)
            self.performAction(action)
            

    def trainAIPPO(self, iterations, timesteps):
        scores = []
        maxScore = -250

        #for each iteration
        for iter in range(iterations):
            rewards = []
            actions = []
            states = []
            values = []
            advantages = []

            #run for timesteps
            for t in range(timesteps):
                #run old policy
                states.append(np.array(self.getState()))
                actions.append(self.getBestAction())
                rewards.append(self.rewardFunction(states[t], actions[t]))
                self.performAction(actions[t])
            
            #data collected, now to be analyzed

            #calculate advantages
            deltaGAE = self.calculateGAEDeltas(rewards, states)

            for t in range(len(deltaGAE)):
                advantages.append(self.calculateAdvantage(t, deltaGAE))


            avgProb = self.getAvgProbability(states, actions)

            #calculate policy adjustments
            softmaxTargets = []

            for t in range(len(advantages)):

                currentSoftmax = self.policyFunction.predict(np.array([states[t]]))[0].tolist()
                softmaxTargets.append(np.array(self.calculateSoftmaxTarget(currentSoftmax,actions[t],advantages[t])))

            states.pop()
            self.policyFunction.fit(x=np.array(states), y=np.array(softmaxTargets), batch_size=64, epochs=30, verbose=0)


            #calculate cumulative rewards for new value function updates
            values = []
            for t in range(len(states)):
                values.append(self.cumulativeReward(t, rewards))

            
            #train the value function
            self.valueFunction.fit(x=np.array(states), y=np.array(values), batch_size=64, epochs=30, verbose=0)

            #print out stats
            print "Iteration: " + str(iter)
            s = self.runGameAISamples(100)

            if s > maxScore:
                maxScore = s
                self.policyFunction.save("BestPolicy.h5")

            print "AI score: " + str(s)
            print "Avg probability: " + str(avgProb)
            print "Avg advantage: " + str(self.getAvgAbsAdvantage(advantages))

            f = open("performance.csv", "a")
            f.write(str(iter) + "," + str(s) + "," + str(avgProb) + "\n")
            f.close()

            print ""





n = NumberCatchingAI(200)
n.playGameAI("BestPolicy.h5")
#n.playGameHuman()
#n.trainAIPPO(1000, 10000)
