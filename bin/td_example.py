#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: TD_EXAMPLE.PY
Date: Friday, February 24 2012
Description: Examples using TD algorithms to learn value functions.
"""


from gridworld.boyan import Boyan
from gridworld.chainwalk import Chainwalk
from cartpole import CartPole
from td import TD, TDQ, TDQCmac, SarsaCmac, Sarsa, ActorCritic, ActorCriticCmac

# a simple environment
env = Boyan()
learner = TD(13, 0.1, 1.0, 0.8)
learner.learn(1000,env,env.random_policy)
print learner.V

env = Chainwalk()
learnerq = TDQ(2,4, 0.1, 0.9, 0.8)

import pdb

env = CartPole()
#learnerq = SarsaCmac(2,0.01,0.95,0.9,0.01)
#learnerq = Sarsa(2,170,0.001,0.95,0.5,0.01)
#learnerq = ActorCritic(2, 162, 0.5, 0.5, 0.95, 0.8, 0.9) # From an old Sutton paper -- seems to work quite well.
learnerq = ActorCriticCmac(2, 0.5, 1.0, 0.95, 0.8, 0.9) # Clearly does some learning, but not nearly as well. Policy not as stable.
learnerq.learn(1000,env)


