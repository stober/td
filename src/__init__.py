#!/usr/bin/python
"""
Author: Jeremy M. Stober
Program: TD.PY
Date: Thursday, April 26 2011
Description: TD V function approximation using simple tabular state representation.
"""

import os, sys, getopt, pdb
import random as pr
import numpy as np
import numpy.random as npr
import pickle
from cmac import TraceCMAC

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

        self.e[pstate] += 1.0 

        #for s in range(self.nstates):
        self.V += self.alpha * delta * self.e
        self.e *= (self.gamma * self.ld)

        return delta

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

    def train(self, pstate, paction, reward, state, action, delta = None):
        """
        A single step of reinforcement learning.
        """

        if delta is None:
            delta = self.delta(pstate, paction, reward, state, action)

        self.e[paction,pstate] += 1.0 # replacing traces

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

class TDQCmac(TDQ):
    """
    Uses CMACs to approximate the action value function.
    """
    def __init__(self, nactions, alpha, gamma, ld):
        self.nactions = nactions
        self.alpha = alpha
        self.gamma = gamma
        self.ld = ld
        self.cmac = []
        for a in range(nactions):
            self.cmac.append(TraceCMAC(32, 0.1, alpha, ld * gamma, replace=True,inc=1.0))

    def value(self, action, vector):
        return self.cmac[action].eval(vector)

    def train(self, pvector, paction, reward, vector, action):
        delta = self.delta(pvector, paction, reward, vector, action)
        self.cmac[paction].train(pvector, delta)

    def reset(self):
        for cmac in self.cmac:
            cmac.reset()

def flip(p):
    """ Flip a biased coin. """
    if pr.random() < p:
        return True
    else:
        return False

def rargmax(vector):
    """ Argmax that chooses randomly among eligable maximum indices. """
    m = np.amax(vector)
    indices = np.argwhere(vector == m)
    if len(indices) == 0:
        pdb.set_trace()
        return 0
    return pr.choice(indices)

def softmax(w, t = 1.0):
    e = np.exp(w / t)
    dist = e / np.sum(e)
    return dist

class ActorCritic(object):
    """
    Actor-critic RL with softmax selection.
    """
    def __init__(self, nactions, nstates, alpha, beta, gamma, ld_alpha, ld_beta):
        self.critic = TD(nstates, alpha, gamma, ld_alpha)
        self.actor = TDQ(nactions, nstates, beta, gamma, ld_beta)
        self.epsilon = 0.01
        self.nactions = nactions
        self.nstates = nstates

    def softmax_policy(self, vector):
        vals = np.array([self.actor.value(a,vector) for a in range(self.nactions)])
        dist = softmax(vals, t = 1.0)
        res = npr.multinomial(1,dist)
        return np.argmax(res)

    def epsilon_policy(self, vector):
        if flip(self.epsilon):
            return pr.choice(range(self.nactions))
        else:
            return rargmax([self.actor.value(a,vector) for a in range(self.nactions)])

    def best(self, vector):
        return rargmax([self.actor.value(a,vector) for a in range(self.nactions)])

    def train(self, pstate, paction, reward, state, action):
        delta = self.critic.train(pstate,reward,state) # find the td error and use it to train the critic
        self.actor.train(pstate, paction, reward, state, action, delta)

    def reset(self):
        self.actor.reset()
        self.critic.reset()
        
    def learn(self, nepisodes, env):
        """
        Right now this is specifically for learning the cartpole task.
        """

        # learn for niters episodes with resets
        count = 0
        for i in range(nepisodes):
            self.reset()
            env.reset()
            next_action = self.softmax_policy(env.state())
            print "Episode %d, Prev count %d" % (i, count)
            count = 0
            while not env.failure():
                pstate, paction, reward, state = env.move(next_action,boxed = True)
                next_action = self.softmax_policy(env.state())
                self.train(pstate, paction, reward, state, next_action)
                count += 1
                if count % 1000 == 0:
                    print "Count: %d" % count
                if count > 10000:
                    break

    

class Sarsa(TDQ):
    """
    On policy RL.
    """

    def __init__(self, nactions, nstates, alpha, gamma, ld, epsilon):
        self.epsilon = epsilon
        TDQ.__init__(self, nactions, nstates, alpha, gamma, ld)

    def softmax_policy(self, vector):
        vals = np.array([self.value(a,vector) for a in range(self.nactions)])
        dist = softmax(vals, t = 10.0)
        res = npr.multinomial(1,dist)
        return np.argmax(res)

    def epsilon_policy(self, vector):
        if flip(self.epsilon):
            return pr.choice(range(self.nactions))
        else:
            return rargmax([self.value(a,vector) for a in range(self.nactions)])

    def best(self, vector):
        return rargmax([self.value(a,vector) for a in range(self.nactions)])

    def learn(self, nepisodes, env):
        """
        Right now this is specifically for learning the cartpole task.
        """

        # learn for niters episodes with resets
        count = 0
        for i in range(nepisodes):
            self.reset()
            env.reset()
            next_action = self.epsilon_policy(env.state())
            print "Episode %d, Prev count %d" % (i, count)
            count = 0
            while not env.failure():
                pstate, paction, reward, state = env.move(next_action,boxed = True)
                next_action = self.epsilon_policy(env.state())
                self.train(pstate, paction, reward, state, next_action)
                count += 1
                if count % 1000 == 0:
                    print "Count: %d" % count
                if count > 10000:
                    break

    def test(self, env):
        env.reset()
        count = 0
        while not env.failure():
            next_action = self.best(env.state())
            env.move(next_action)
            count += 1
            if count % 1000 == 0:
                print "Count: %d" % count
            if count > 10000:
                break
        print "Balanced for %d timesteps." % count


class SarsaCmac(TDQCmac):
    """
    On policy RL. Use CMACs to approximate on-policy value function.
    """

    def __init__(self, nactions, alpha, gamma, ld, epsilon):
        self.epsilon = epsilon
        TDQCmac.__init__(self, nactions, alpha, gamma, ld)

    def softmax_policy(self, *vector):
        vals = np.array([self.value(a,vector) for a in range(self.nactions)])
        dist = softmax(vals, t = 0.001)
        res = npr.multinomial(1,dist)
        return np.argmax(res)

    def epsilon_policy(self, *vector):
        if flip(self.epsilon):
            return pr.choice(range(self.nactions))
        else:
            return rargmax([self.value(a,vector) for a in range(self.nactions)])

    def best(self, *vector):
        return rargmax([self.value(a,vector) for a in range(self.nactions)])

    def learn(self, nepisodes, env):
        # learn for niters episodes with resets
        count = 0
        for i in range(nepisodes):
            self.reset()
            env.reset()
            next_action = self.softmax_policy(env.x,env.xdot,env.theta,env.thetadot)
            print "Episode %d, Prev count %d" % (i, count)
            count = 0
            while not env.failure():
                pstate, paction, reward, state = env.move(next_action)
                next_action = self.softmax_policy(env.x,env.xdot,env.theta,env.thetadot)
                self.train(pstate, paction, reward, state, next_action)
                count += 1
                if count % 1000 == 0:
                    print "Count: %d" % count
                if count > 10000:
                    break

    def test(self, env):
        env.reset()
        count = 0
        while not env.failure():
            next_action = self.best(env.x,env.xdot,env.theta,env.thetadot)
            env.move(next_action)
            count += 1
            if count % 1000 == 0:
                print "Count: %d" % count
            if count > 10000:
                break
        print "Balanced for %d timesteps." % count

class ActorCriticCmac(object):
    pass
