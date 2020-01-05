# Joao Porto 89472 ; Tiago Le 89550 ; Grupo 24

import random
import math
import numpy as np
import matplotlib.pyplot as plt
import visualizer as vis

# LearningAgent to implement
# no knowledeg about the environment can be used
# the code should work even with another environment


class LearningAgent:

    # nS maximum number of states
    # nA maximum number of action per state
    def __init__(self, nS, nA):
        self.nS = nS
        self.nA = nA

        # initialize Q-table
        self.Q = np.zeros((nS, nA), dtype=float)
        self.statePlayNum = [-1] * nS
        self.epsilon = 0.2
        self.alpha = 0.6
        self.gamma = 0.8

    # Select one action, used when learning
    # st - is the current state
    # aa - is the set of possible actions
    # for a given state they are always given in the same order
    # returns
    # a - the index to the action in aa
    def selectactiontolearn(self, st, aa):
        # print("select one action to learn better")

        if self.statePlayNum[st] == -1:
            self.statePlayNum[st] = len(aa)

        a = -1
        if random.uniform(0, 1) < self.epsilon:
            a = random.randrange(0, len(aa))
        else:
            a = np.argmax(self.Q[st][:len(aa)])
        return a

    # Select one action, used when evaluating
    # st - is the current state
    # aa - is the set of possible actions
    # for a given state they are always given in the same order
    # returns
    # a - the index to the action in aa
    def selectactiontoexecute(self, st, aa):
        # Select action with max q value
        a = np.argmax(self.Q[st][:len(aa)])
        return a

    # this function is called after every action
    # st - original state
    # nst - next state
    # a - the index to the action taken
    # r - reward obtained
    def learn(self, ost, nst, a, r):
        # Caclulate q value for state-action pair
        bestNextQ = -math.inf
        for q in self.Q[nst][:self.statePlayNum[nst]]:
            if q > bestNextQ:
                bestNextQ = q

        qTarget = r + self.gamma * bestNextQ
        qDelta = qTarget - self.Q[ost][a]
        self.Q[ost][a] += self.alpha * qDelta
        return
