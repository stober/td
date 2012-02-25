#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: TD_EXAMPLE.PY
Date: Friday, February 24 2012
Description: Examples using TD algorithms to learn value functions.
"""


from gridworld.boyan import Boyan
from gridworld.chainwalk import Chainwalk
from td import TD, TDQ

# a simple environment
env = Boyan()
learner = TD(13, 0.1, 1.0, 0.8)
learner.learn(1000,env,env.random_policy)
print learner.V

env = Chainwalk()
learnerq = TDQ(2,4, 0.1, 0.9, 0.8)

import pdb
pdb.set_trace()
learnerq.learn(1000,env,env.random_policy,episodic = False)
print learnerq.V

