#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import srModel
import evolutionaryAlg

def gen_function(K):
	theta = np.random.rand(K)*8-4
	s = np.random.rand(2,K)*4-2
	return srModel.vectorizeParameters(theta,s)

def evalfunction(ind):
	return E2.vectorizedLikelihood(ind, gamma = 2)

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

pop = evolutionaryAlg.ini_pop(100, gen_function, args = D.N)
fitness = evolutionaryAlg.evaluate(pop, evalfunction)
