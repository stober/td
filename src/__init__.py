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

    def train(self, action, pvector, reward, vector):
        self.cmac[action].train(vector, pvector, reward)

    def reset(self):
        for cmac in self.cmac:
            cmac.reset()

    def policy(self,vector):
        values = []
        for cmac in self.cmac:
            values.append(cmac.eval(vector))
        return np.argmax(values)

class TDLinear(object):
    """
    The same training algorithm as lstd.td but packaged differently
    with a slightly more flexible interface (can call train on single
    steps for instance).
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
        # basically does one step of training
        # note that the gradient of the value function wrt the params are the current features

        delta = reward + (self.gamma * self.value(features)) - self.value(pfeatures)
        self.e = self.gamma * self.ld * self.e + pfeatures
        self.params = self.params + self.alpha * delta * self.e

    def reset(self):
        self.e = np.zeros(self.nstates)


class TD(object):

    def __init__(self, nstates, alpha, gamma, ld):
        self.V = np.zeros(nstates)
        self.e = np.zeros(nstates)
        self.nstates = nstates
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount
        self.ld = ld # lambda

    def train(self, pstate, reward, state):
        # basically does one step of training

        delta = reward + (self.gamma * self.V[state]) - self.V[pstate]
        self.e[pstate] += 1

        for s in range(self.nstates):
            self.V[s] += self.alpha * delta * self.e[s]
            self.e[s] *= (self.gamma * self.ld)

    def sim(self, niters, env):
        # added for direct comparison with other implementations

        for i in range(niters):
            env.reset()
            next = env.current
            self.reset() # replacing traces?

            while not env.terminal(next):
                previous, action, reward, next = env.move(env.random_policy())

                self.train(previous, reward, next)

            print self.V

    def reset(self):
        self.e = np.zeros(self.nstates)

