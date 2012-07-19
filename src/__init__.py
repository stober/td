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
from utils import incavg

class TD(object):
    """
    Discrete value function approximation via temporal difference learning.
    """

    def __init__(self, nstates, alpha, gamma, ld, init_val = 0.0):
        self.V = np.ones(nstates) * init_val
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

    def learn(self, nepisodes, env, policy, verbose = True):
        # learn for niters episodes with resets
        for i in range(nepisodes):
            self.reset()
            t = env.single_episode(policy) # includes env reset
            for (previous, action, reward, state, next_action) in t:
                self.train(previous, reward, state)
            if verbose:
                print i

    def reset(self):
        self.e = np.zeros(self.nstates)

class TDQ(object):
    """
    Discrete action-value function approximation via temporal difference learning.
    """
    def __init__(self, nactions, nstates, alpha, gamma, ld, init_val = 0.0):
        self.V = np.ones((nactions, nstates)) * init_val
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

    def learn(self, nepisodes, env, policy, episodic = True, verbose = True):
        # learn for nepisodes with resets
        if episodic:
            for i in range(nepisodes):
                self.reset()
                t = env.single_episode(policy) # includes env reset
                for (previous, paction, reward, state, action) in t:
                    self.train(previous, paction, reward, state, action)
                if verbose:
                    print i

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

class TDCmac(TD):
    def __init__(self, alpha, gamma, ld, nlevels = 10, resolution = 0.1):
        self.nlevels = nlevels
        self.resolution = resolution
        self.cmac = TraceCMAC(self.nlevels, self.resolution, alpha, ld * gamma, replace = True, inc = 1.0)
        self.gamma = gamma

    def __len__(self):
        return len(self.cmac)

    def value(self, vector):
        return self.cmac.eval(vector)

    def train(self, pvector, reward, vector):
        delta = self.delta(pvector, reward, vector)
        self.cmac.train(pvector, delta)
        return delta

    def reset(self):
        self.cmac.reset()

class TDQCmac(TDQ):
    """
    Uses CMACs to approximate the action value function.
    """
    def __init__(self, nactions, alpha, gamma, ld, nlevels = 10, resolution = 0.1):
        self.nactions = nactions
        self.alpha = alpha
        self.gamma = gamma
        self.ld = ld
        self.cmac = []
        self.nlevels = nlevels
        self.resolution = resolution
        for a in range(nactions):
            self.cmac.append(TraceCMAC(self.nlevels, self.resolution, alpha, ld * gamma, replace=True, inc=1.0))

    def __len__(self):
        s = 0
        for c in self.cmac:
            s += len(c)
        return s

    def value(self, action, vector):
        return self.cmac[action].eval(vector)

    def train(self, pvector, paction, reward, vector, action, delta = None):
        if delta is None:
            delta = self.delta(pvector, paction, reward, vector, action)
        self.cmac[paction].train(pvector, delta)
        return delta

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
    indices = np.nonzero(vector == m)[0]
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
    def __init__(self, nactions, nstates, alpha, beta, gamma, ld_alpha, ld_beta, actor_init = 0.0, critic_init = 0.0):
        self.critic = TD(nstates, alpha, gamma, ld_alpha, critic_init)
        self.actor = TDQ(nactions, nstates, beta, gamma, ld_beta, actor_init)
        self.epsilon = 0.01
        self.nactions = nactions
        self.nstates = nstates

    def softmax_policy(self, vector, t = 1.0):
        vals = np.array([self.actor.value(a,vector) for a in range(self.nactions)])
        dist = softmax(vals, t = 1.0)
        res = npr.multinomial(1,dist)
        return np.argmax(res)

    def __len__(self):
        return 0

    def value(self, vector):
        return self.critic.value(vector)

    def avalue(self, action, vector):
        return self.actor.value(action, vector)


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

class SampleModelValueIteration(object):
    def __init__(self, nactions, nstates):
        
        self.reward_model = {}
        self.nstates = nstates
        self.nactions = nactions
        self.transition_model = np.zeros((nstates, nactions, nstates))
        self.values = np.zeros(nstates)
        self.optimal = np.zeros(nstates) # store best policy here
        self.learned = False # set to true when finished learning.
        self.gamma = 0.9

    def random(self, state):
        return pr.choice(range(self.nactions))

    def best(self, state):
        return self.optimal[state]

    def train(self, pstate, paction, reward, state, next_action):

        # track average reward
        if self.reward_model.has_key((pstate,paction,state)):
            self.reward_model[pstate,paction,state].send(reward)
        else:
            self.reward_model[pstate,paction,state] = incavg(reward)

        # track transition counts (normalize later to get multinomial
        # distributions conditioned on pstate and paction).
        self.transition_model[pstate,paction,state] += 1


    def learn(self, nepisodes, env, verbose = True):
        # First collect samples to populate a model of the environment.
        count = 0
        for i in range(nepisodes):
            env.reset()
            next_action = self.random(env.state())
            if verbose: print "Episode %d, Prev count %d" % (i, count)
            count = 0
            while not env.failure():
                pstate, paction, reward, state = env.move(next_action)
                next_action = self.random(env.state())
                self.train(pstate, paction, reward, state, next_action)
                count += 1
                if count % 1000 == 0:
                    print "Count: %d" % count
                if count > 10000:
                    break

        # some additional processing prior to value iteration
        self.normalized_reward = {}
        for k,v in self.reward_model.items():
            self.normalized_reward[k] = v.next()

        self.normalized_transitions = np.zeros((self.nstates, self.nactions, self.nstates))
        for i in range(self.nstates):
            for a in range(self.nactions):
                if np.sum(self.transition_model[i,a,:]) == 0:
                    pass
                else:
                    self.normalized_transitions[i,a,:] = self.transition_model[i,a,:] / float(np.sum(self.transition_model[i,a,:]))

        # double check that we have a learned distribution

        print "Value Iteration"
        values = np.zeros(self.nstates)
        rtol = 1e-4

        condition = True
        i = 0
        while condition:
            delta = 0
            for s in range(self.nstates):
                v = values[s]
                sums = np.zeros(self.nactions)
                for a in range(self.nactions):
                    for t in range(self.nstates):
                        sums[a] += self.normalized_transitions[s,a,t] * (self.normalized_reward.get((s,a,t),0.0) + self.gamma * values[t])
                
                values[s] = np.max(sums)
                delta = max(delta, abs(v - values[s]))
            print i,delta
            i += 1
            if delta < rtol:
                break

        # compute the optimal policy
        policy = np.zeros(self.nstates, dtype=int) # record the best action for each state
        for s in range(self.nstates):
            sums = np.zeros(self.nactions)
            for a in range(self.nactions):
                for t in range(self.nstates):
                    sums[a] += self.normalized_transitions[s,a,t] * (self.normalized_reward.get((s,a,t),0.0) + self.gamma * values[t])
            policy[s] = np.argmax(sums)


        return values,policy
        
        # return self.normalized_reward, self.normalized_transitions
        # Next do value iteration using the learned.

    

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

    def learn(self, nepisodes, env, verbose = True):
        """
        Right now this is specifically for learning the cartpole task.
        """

        # learn for niters episodes with resets
        count = 0
        for i in range(nepisodes):
            self.reset()
            env.reset()
            next_action = self.epsilon_policy(env.state())
            if verbose: print "Episode %d, Prev count %d" % (i, count)
            count = 0
            while not env.failure():
                #,boxed = True
                pstate, paction, reward, state = env.move(next_action)
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

class ActorCriticCmac(ActorCritic):

    def __init__(self, nactions, alpha, beta, gamma, ld_alpha, ld_beta, tdlvls = 10, tdres = 0.1, tdqlvls = 10, tdqres=0.1):
        self.critic = TDCmac(alpha, gamma, ld_alpha, tdlvls, tdres)
        self.actor = TDQCmac(nactions, beta, gamma, ld_beta, tdqlvls, tdqres)
        self.epsilon = 0.01
        self.nactions = nactions

    def __len__(self):
        return len(self.critic) + len(self.actor)

    def learn(self, nepisodes, env):
        """
        Right now this is specifically for learning the cartpole task.
        """

        # learn for niters episodes with resets
        count = 0
        for i in range(nepisodes):
            self.reset()
            env.reset()
            next_action = self.softmax_policy([env.x,env.xdot,env.theta,env.thetadot])
            print "Episode %d, Prev count %d" % (i, count)
            count = 0
            while not env.failure():
                pstate, paction, reward, state = env.move(next_action,boxed = False)
                next_action = self.softmax_policy([env.x,env.xdot,env.theta,env.thetadot])
                self.train(pstate, paction, reward, state, next_action)
                count += 1
                if count % 1000 == 0:
                    print "Count: %d" % count
                if count > 10000:
                    break


