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

trainingIterations = 1000
timestepsPerIteration = 10000


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

        for i in range(5):
            self.numberInfo.append(NumberData(random.randint(1,9), random.randint(0,14), i * 5))

        #PPO hyperparameters
        self.discountFactor = 0.95

        self.horizon = 50

        self.GAELambda = 0.95
        
        self.epsilon = 0.2


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
        """
        Records current game state. Note: State does not include information about turns or score.

        Returns:
            The state as a list
        """
        ret = []
        for i in range(len(self.numberInfo)):
            ret.append(self.numberInfo[i].value)
            ret.append(self.numberInfo[i].xCoord)
            ret.append(self.numberInfo[i].yCoord)
        ret.append(self.playerLocation)
        return ret

    def getBestAction(self):
        """
        Chooses best action based on weighted values from softmax outputs of the policy function

        Returns:
            The action as in integer (-1 for left, 0 for stay, 1 for right)
        """
        
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
        """
        Draws the game so that user can either play or watch the game.
        """

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
            print("")
        
        print("Score: " + str(self.score))
        print("Turn " + str(self.currentTurn) + "/" + str(self.maxTurns))



    def setState(self, state):
        """
        Sets the game to a specified state. Note: State does not include information about turn number or score

        Args:
            state: the state to set to
        """
        # number of numbers in the queue
        self.numberInfo.clear()
        
        for i in range(0, 15, 3):
            data = NumberData(state[i], state[i + 1], state[i + 2])
            self.numberInfo.append(data)

        self.playerLocation = state[ len(state) - 1]



    def rewardFunction(self, state, action):
        """
        Measures reward given by being in a state and taking an action.

        Args:
            state: The state prior to the action being taken

            action: The action that is taken to observe reward

        Returns:
            The reward
        """

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
        """
        Prints out game to the terminal so you can play!
        """

        while( self.currentTurn <= self.maxTurns):
            #draw game
            self.drawGame()
            #ask for input
            i = input()
            action = 0

            if(i == "a"):
                action = -1
            elif(i == "d"):
                action = 1


            #perform the action
            self.score += self.rewardFunction(self.getState(), action)
            self.performAction(action)

            



    def performAction(self, action):
        """
        Performs action and progresses game.

        Args:
            action: The action that will be applied to the game
        """
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
        """
        Measures agent performance on one game

        Returns:
            The agent's score
        """
        self.score = 0
        for i in range(200):
            action = self.getBestAction()
            self.score += self.rewardFunction(self.getState(), action)
            self.performAction(action)
            
        
        return self.score
    
    def runGameAISamples(self, numSamples):
        """
        Measures performance of agent by averaging scores of independent games.

        Args:
            numSamples: The number of samples to collect. Small numbers calculate fast, but with poor estimation of the mean.

        Returns:
            Average score of the games.
        """
        scores = []
        for i in range(numSamples):
            scores.append(self.runGameAI())

        return np.mean(scores)

    def cumulativeReward(self, timestep, rewards):
        """
        Calculates the cumulative discounted reward. Used in training the value function

        Args:
            timestep: The timestep to calculate rewards from.

            rewards: A list of rewards collected.

        Returns:
            Returns the cumulative discounted reward
        """
        ret = 0

        for t in range(timestep, min(len(rewards), len(rewards) + self.horizon)):
            ret += (self.discountFactor ** (t - timestep)) * rewards[t]
        
        return ret


    def getValue(self, state):
        """
        Uses value function to predict estimated rewards from a state

        Args:
            state: The state input

        Returns:
            Returns the estimated rewards from the input state
        """
        return (self.valueFunction.predict(np.array([state])))[0][0]

    def calculateGAEDeltas(self, rewards, states):
        """
        Calculates the GAEDeltas used in calculating advantages. Equation from the PPO paper

        Args:
            rewards: A list of rewards (each being a scalar)
            
            states: A list of states (each being a list)

        Returns:
            Returns a list of the GAE deltas to be used to calculate advantages
        """
        ret = []

        for t in range(len(rewards) - 1):
            s = rewards[t] + self.discountFactor * self.getValue(states[t + 1]) - self.getValue(states[t])
            ret.append(s)
        
        return ret

    def calculateAdvantage(self, timestep, GAEDeltas):
        """
        Calculates advantage using the Generalized Advantage estimation found in the PPO paper

        Args:
            timestep: The timestep to calculate from

            GAEDeltas: A list of precomputed GAEDeltas

        Returns:
            The advantage calculated
        """
        r = 0
        for t in range(timestep, len(GAEDeltas)):
            r += (self.GAELambda * self.discountFactor)**(t - timestep) * GAEDeltas[t]

        return r


    def calculateSoftmaxTarget(self, oldSoftmax, action, advantage):
        """
        Calculates the new softmax output based on whether or not an advantage is positive or negative.

        Args:
            oldSoftmax: The current softmax values.
            
            action: The action that was taken as an integer

            advantage: Scalar value representing how well that action did

        Returns:
            New Softmax output that only deviates from old softmax by the model's epsilon hyperparameter
        """
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
        """
        Returns the probability of the action given the state using the policy function

        Args:
            state: A list representing the state

            action: The action that is taken as an integer

        Returns:
            Probability P( action | state )
        """
        return self.policyFunction.predict(np.array([state]))[0][action + 1]


    def getAvgProbability(self, states, actions):
        """
        Computes the average probability of actions in a recorded trajectory. Good metric to see how "confident" the policy is

        Args:
            states: a list of the states in the trajectory

            actions: a list of the actions in the trajectory.

        Returns:
            The average probability
        """
        r = 0

        for i in range(len(states)):
            r += self.getProbability(states[i], actions[i])

        return r / len(states)

    def getFloatingAverage(self, data, n):
        """
        Returns the floating average of a sequence of numbers

        Args:
            data: a list of numbers (scalars)
            
            n: The period to average over

        Returns:
            The floating average.
        """
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
        """
        Debugging function to see how the absolute value of the advantages changes over time

        Args:
            advantages: a list of advantanges (scalar)

        Returns:
            Average absolute value of the advantages
        """
        r = 0

        for i in range(len(advantages)):
            r += abs(advantages[i])

        return r / len(advantages)

    def playGameAI(self, policyName):
        """
        Agent plays game by using policy parameters

        Args:
            policyName: The name of the policy to load. Should be in the same folder as this file
        """

        self.policyFunction = keras.models.load_model(policyName)

        for t in range(200):

            self.drawGame()
            i = input()

            action = self.getBestAction()
            self.score += self.rewardFunction(self.getState(), action)
            self.performAction(action)
            

    def trainAIPPO(self, iterations, timesteps):
        """
        Trains the agent using the Proximal Policy Optimization algorithm

        Args:
            iterations: Number of training iterations to perform
            
            timesteps: Number of timesteps to collect data per training iteration

        """
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
            print("Iteration: " + str(iter))
            s = self.runGameAISamples(100)

            if s > maxScore:
                maxScore = s
                self.policyFunction.save("BestPolicy.h5")

            print("AI score: " + str(s))
            print("Avg probability: " + str(avgProb))

            f = open("performance.csv", "a")
            f.write(str(iter) + "," + str(s) + "," + str(avgProb) + "\n")
            f.close()

            print("")







"""
Command line args description:
t - train the AI. Warning: Takes forever
p - play the game as a human
w - watch the AI play with the policy "BestPolicy.h5" in the folder
"""

if len(sys.argv) is not 2:
    print("Need 1 command line arg. Options are t (train), p (play), w (watch)")
    exit()

n = NumberCatchingAI(200)

if sys.argv[1] == 't':
    n.trainAIPPO(trainingIterations, timestepsPerIteration)
elif sys.argv[1] == 'p':
    print("Press a,s,d then enter to move bucket.")
    n.playGameHuman()
elif sys.argv[1] == 'w':
    print("Press enter to advance the game.")
    n.playGameAI("BestPolicy.h5")
else:
    print("Bad command line arg. Options are t (train), p (play), w (watch)")