#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: TD_EXAMPLE.PY
Date: Friday, February 24 2012
Description: Examples using TD algorithms to learn value functions.
"""


from gridworld.boyan import Boyan
from td import TD

# a simple environment
env = Boyan()
learner = TD(13, 0.1, 1.0, 0.8)
learner.learn(1000,env,env.random_policy)
print learner.V
