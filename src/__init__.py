#!/usr/bin/python
"""
Author: Jeremy M. Stober
Program: TD.PY
Date: Thursday, April 26 2011
Description: TD V function approximation using simple tabular state representation.
"""

import os, sys, getopt, pdb
import random as prandom
import numpy as np
import pickle
from cmac import TraceCMAC

class TDCmac(object):
    def __init__(self, nactions, beta, gamma, ld):
        self.cmac = []
        for a in range(nactions):
            self.cmac.append(TraceCMAC(32,0.1,beta, ld, gamma))

    def value(self, action, vector):
        return self.cmac[action].eval(vector)

    def train(self, action, pvector, reward, vector, next_action):
        # This has an error. Needs to include the next action.
        self.cmac[action].train(vector, pvector, reward)

    def reset(self):
        for cmac in self.cmac:
            cmac.reset()

    def policy(self,vector):
        values = []
        for cmac in self.cmac:
            values.append(cmac.eval(vector))
        return np.argmax(values)

class TD(object):
    """
    Discrete value function approximation via temporal difference learning.
    """

    def __init__(self, nstates, alpha, gamma, ld):
        self.V = np.zeros(nstates)
        self.e = np.zeros(nstates)
        self.nstates = nstates
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount
        self.ld = ld # lambda

    def value(self, state):
        return self.V[state]

    def delta(self, pstate, reward, state):
        """
        This is the core error calculation. Note that if the value
        function is perfectly accurate then this returns zero since by
        definition value(pstate) = gamma * value(state) + reward.
        """
        return reward + (self.gamma * self.value(state)) - self.value(pstate)

    def train(self, pstate, reward, state):
        """
        A single step of reinforcement learning.
        """

        delta = self.delta(pstate, reward, state)

        self.e[pstate] = 1.0 # replacing traces

        #for s in range(self.nstates):
        self.V += self.alpha * delta * self.e
        self.e *= (self.gamma * self.ld)

    def learn(self, nepisodes, env, policy):
        # learn for niters episodes with resets
        for i in range(nepisodes):
            self.reset()
            t = env.single_episode(policy) # includes env reset
            for (previous, action, reward, state, next_action) in t:
                self.train(previous, reward, state)

    def reset(self):
        self.e = np.zeros(self.nstates)

class TDQ(object):
    """
    Discrete action-value function approximation via temporal difference learning.
    """
    def __init__(self, nactions, nstates, alpha, gamma, ld):
        self.V = np.zeros((nactions, nstates))
        self.e = np.zeros((nactions, nstates))
        self.nactions = nactions
        self.nstates = nstates
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount
        self.ld = ld # lambda

    def value(self, action, state):
        return self.V[action, state]

    def delta(self, pstate, paction, reward, state, action):
        """
        This is the core error calculation. Note that if the value
        function is perfectly accurate then this returns zero since by
        definition value(pstate) = gamma * value(state) + reward.
        """
        return reward + (self.gamma * self.value(action,state)) - self.value(paction,pstate)

    def train(self, pstate, paction, reward, state, action):
        """
        A single step of reinforcement learning.
        """

        delta = self.delta(pstate, paction, reward, state, action)

        self.e[paction,pstate] = 1.0 # replacing traces

        self.V += self.alpha * delta * self.e
        self.e *= (self.gamma * self.ld)

    def learn(self, nepisodes, env, policy, episodic = True):
        # learn for nepisodes with resets
        if episodic:
            for i in range(nepisodes):
                self.reset()
                t = env.single_episode(policy) # includes env reset
                for (previous, paction, reward, state, action) in t:
                    self.train(previous, paction, reward, state, action)
        else:
            for i in range(nepisodes):
                t = env.trace(nepisodes, policy)
                for (previous, paction, reward, state, action) in t:
                    self.train(previous, paction, reward, state, action)

    def reset(self):
        self.e = np.zeros((self.nactions, self.nstates))


class TDLinear(TD):
    """
    A more general linear value function representation.
    """

    def __init__(self, nfeatures, alpha, gamma, ld):
        self.params = np.zeros(nfeatures)
        self.e = np.zeros(nfeatures)
        self.nfeatures = nfeatures
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount
        self.ld = ld # lambda

    def value(self,features):
        return np.dot(self.params,features)

    def train(self, pfeatures, reward, features):
        delta = self.delta(pfeatures, reward, features)
        self.e = self.gamma * self.ld * self.e + pfeatures
        self.params = self.params + self.alpha * delta * self.e

    def reset(self):
        self.e = np.zeros(self.nstates)
