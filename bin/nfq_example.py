#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: NFQ_EXAMPLE.PY
Date: Thursday, March  1 2012
Description: Test NFQ on my cartpole simulation.
"""

from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners.valuebased import NFQ, ActionValueNetwork
from cartpole import CartPole
import numpy as np

module = ActionValueNetwork(4,2)
learner = NFQ()
learner.explorer.epsilon = 0.4
agent = LearningAgent(module, learner)

env = CartPole()
cnt = 0
for i in range(1000):
    
    env.reset()
    print "Episode: %d, Count: %d" % (i,cnt)
    cnt = 0
    while not env.failure():
        agent.integrateObservation(env.observation())
        action = agent.getAction()
        pstate, paction, reward, state = env.move(action)
        cnt += 1
        agent.giveReward(reward)
    agent.learn(1)

