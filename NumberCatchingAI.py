from collections import deque
import random
import sys
import tensorflow as tf
from tensorflow import keras

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

        self.policyFunction = keras.Sequential([
            keras.layers.Dense(64, input_shape=(16,), activation=tf.nn.tanh),
            keras.layers.Dense(64, activation=tf.nn.tanh),
            keras.layers.Dense(3, activation=tf.nn.softmax)
        ])

        self.policyFunction.compile(loss=keras.losses.mean_absolute_error,optimizer=keras.optimizers.SGD)


        self.valueFunction = keras.Sequential([
            keras.layers.Dense(64, input_shape=(16,), activation=tf.nn.tanh),
            keras.layers.Dense(64, activation=tf.nn.tanh),
            keras.layers.Dense(1)
        ])

        self.valueFunction.compile(loss=keras.losses.mean_absolute_error, optimizer=keras.optimizers.SGD)


        for i in range(5):
            self.numberInfo.append(NumberData(random.randint(1,9), random.randint(0,14), i * 5))

    
    def getState(self):
        ret = []
        for i in range(len(self.numberInfo)):
            ret.append(self.numberInfo[i].value)
            ret.append(self.numberInfo[i].xCoord)
            ret.append(self.numberInfo[i].yCoord)
        ret.append(self.playerLocation)
        return ret



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



    def setState(self, state):
        # number of numbers in the queue
        self.numberInfo.clear()
        

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

    def trainAI(self, iterations, timesteps, learningRate):
        rewards = []
        actions = []
        state = []

        #for each iteration
        #for iter in range(iterations):
            #run for timesteps
            #for t in range(timesteps):







n = NumberCatchingAI(200)
n.playGameHuman()