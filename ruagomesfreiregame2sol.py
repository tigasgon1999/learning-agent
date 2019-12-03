import random
import math
import numpy as np
import matplotlib.pyplot as plt
import visualizer as vis

# LearningAgent to implement
# no knowledeg about the environment can be used
# the code should work even with another environment


class LearningAgent:

    # init
    # nS maximum number of states
    # nA maximum number of action per state
    def __init__(self, nS, nA):
        # define this function
        self.nS = nS
        self.nA = nA
        # initialize Q-table
        self.Q = np.zeros((nS, nA))
        self.epsilon = 0.2
        self.alpha = 0.5
        self.gamma = 0.9
        self.moveCounter = 0
        self.lastState = -1
        # self.vis = Visualizer(600, 1000, 10, 4)

    # Select one action, used when learning
    # st - is the current state
    # aa - is the set of possible actions
    # for a given state they are always given in the same order
    # returns
    # a - the index to the action in aa
    def selectactiontolearn(self, st, aa):
        # define this function
        # print("select one action to learn better")
        # self.moveCounter += 1
        # if not (self.moveCounter % 15):
        #     self.epsilon *= 0.9
        a = -1
        if random.uniform(0, 1) < self.epsilon:
            a = random.randrange(0, len(aa))
            while aa[a] is self.lastState:
                a = random.randrange(0, len(aa))
        else:
            # TODO: change this to numpy.max
            a = np.argmax(self.Q[st][:len(aa)])
            while aa[a] is self.lastState:
                a = np.argmax(np.delete(self.Q[st][:len(aa)], a))        
        return a

    # Select one action, used when evaluating
    # st - is the current state
    # aa - is the set of possible actions
    # for a given state they are always given in the same order
    # returns
    # a - the index to the action in aa
    def selectactiontoexecute(self, st, aa):
        vis.saveTable(self.Q)
        # self.printQ()
        self.moveCounter = 0
        self.epsilon = 1
        # define this function
        a = np.argmax(self.Q[st][:len(aa)])
        # print("select one action to see if I learned")
        return a

    # this function is called after every action
    # st - original state
    # nst - next state
    # a - the index to the action taken
    # r - reward obtained
    def learn(self, ost, nst, a, r):
        # define this function
        # print("learn something from this data")
        bestNextQ = -math.inf
        for q in self.Q[nst]:
            if q > bestNextQ:
                bestNextQ = q

        qTarget = r + self.gamma*bestNextQ
        qDelta = qTarget - self.Q[ost][a]
        self.Q[ost][a] += self.alpha*qDelta
        # self.vis.update(ost, a, self.Q[ost][a])

        return

    def printQ(self):
        i = 0
        for st in self.Q:
            print(i, ":", st)
            i += 1
