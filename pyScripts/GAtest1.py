#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import srModel
from deap import creator, tools

inFolder = '../degradedImg/'
csv1 = 'paramsImage.csv'
csv2 = 'globalParams.csv'

# create Data object
D = srModel.Data(inFolder, csv1, csv2)

# use just a small window of the image to compute parameters
# reducing computational cost
windowshape = (5,5)
D.setWindowLR(windowshape)

# create parameter estimator object
E2 = srModel.ParameterEstimator(D)

gamma0 = 2
s0 = np.zeros((2,D.N))
theta0 = np.zeros(D.N)

# defining initial parameters
v0 = srModel.vectorizeParameters(theta0, s0)
