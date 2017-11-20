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

vtrue = srModel.vectorizeParameters(D.theta, D.s)

N_ga = 170
fit_mean = np.array([])
fit_max = np.array([])
fit_min = np.array([])
norms = np.array([])

pop = evolutionaryAlg.ini_pop(50, gen_function, args = D.N)
fitness, pop = evolutionaryAlg.evaluate(pop, evalfunction)
norms = np.hstack([norms, np.linalg.norm(pop[-1]-vtrue)])

for i in range(N_ga):
	fitnessi, popi = evolutionaryAlg.select_wheel(fitness, pop)

	fitness, pop = evolutionaryAlg.mate(fitnessi, popi, evalfunction)
	fitness, pop = evolutionaryAlg.mutate(fitness, pop, evalfunction, gen_function, args_gen = D.N, rate=0.2)

	fit_mean = np.hstack([fit_mean, np.mean(fitness)])
	fit_min = np.hstack([fit_min, np.min(fitness)])
	fit_max = np.hstack([fit_max, np.max(fitness)])
	norms = np.hstack([norms, np.linalg.norm(pop[-1]-vtrue)])
