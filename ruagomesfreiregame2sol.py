import random
import math

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
        self.Q = [[0 for i in range(nA)] for j in range(nS)]
        self.epsilon = 1
        self.alpha = 0.4
        self.gamma = 0.5
        self.moveCounter = 0

    # Select one action, used when learning
    # st - is the current state
    # aa - is the set of possible actions
    # for a given state they are always given in the same order
    # returns
    # a - the index to the action in aa
    def selectactiontolearn(self, st, aa):
        # define this function
        # print("select one action to learn better")
        self.moveCounter += 1
        self.epsilon = 0.4
        a = -1
        if random.random() < self.epsilon:
            return random.randrange(0, len(aa))
        else:
            # TODO: change this to numpy.max
            bestQ = -math.inf
            for act in range(len(aa)):
                q = self.Q[st][act]
                if q > bestQ:
                    bestQ = q
                    a = act
        return a

    # Select one action, used when evaluating
    # st - is the current state
    # aa - is the set of possible actions
    # for a given state they are always given in the same order
    # returns
    # a - the index to the action in aa
    def selectactiontoexecute(self, st, aa):
        self.moveCounter = 0
        self.moveCounter += 1
        # define this function
        a = -1
        bestQ = -math.inf
        for act in range(len(aa)):
            q = self.Q[st][act]
            if q > bestQ:
                bestQ = q
                a = act
        # print("select one action to see if I learned")
        return a

    # this function is called after every action
    # st - original state
    # nst - next state
    # a - the index to the action taken
    # r - reward obtained
    def learn(self, ost, nst, a, r):
        # define this function
        #print("learn something from this data")
        # print("Move:", self.moveCounter)        
        #self.alpha = 60/(59+self.moveCounter)
        bestNextQ = -math.inf
        for q in self.Q[nst]:
            if q > bestNextQ:
                bestNextQ = q

        qTarget = r + self.gamma*bestNextQ
        qDelta = qTarget - self.Q[ost][a]
        self.Q[ost][a] += self.alpha*qDelta

        return
