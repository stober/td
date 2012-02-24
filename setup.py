#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: SETUP.PY
Date: Friday, February 24, 2011
Description: Setup and install TD algorithms.
"""


from distutils.core import setup

setup(name='td',
      version='0.01',
      description='Temporal Difference Learning in Python',
      author="Jeremy Stober",
      author_email="stober@gmail.com",
      package_dir={"td" : "src"},
      packages=["td"]
      )

