#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script uses my own implementation of the Newton-Conjugate Gradients algorithm
# to maximize the lieklihood function of the parameters

import numpy as np
import srModel
import scipy.optimize
import vismodule

# Initial function definitions
# ---

L = list()
error_norms = list()
grad_norms = list()
v_min = None
iteration = 0

def fprime_L(v):
	# This function estimates the gradient of the likelihood function
	epsilon = 1.4901e-8
	return scipy.optimize.approx_fprime(v, E2.vectorizedLikelihood, epsilon, -1.0,
		gamma)

def f2prime_L(v):
	return np.eye(v.size)

def cb_func(v):
	global L, error_norms, grad_norms, v_min
	error_norms.append(np.linalg.norm(v-v_true))
	if error_norms[-1] == min(error_norms):
		v_min = v
	grad_norms.append(np.linalg.norm(fprime_L(v)))
	

# Initial setup
# ---
inFolder = '../degradedImg/'
csv1 = 'paramsImage.csv'
csv2 = 'globalParams.csv'

# create Data object
D = srModel.Data(inFolder, csv1, csv2)

# use just a small window of the image to compute parameters
# reducing computational cost
windowshape = (7,7)
D.setWindowLR(windowshape)
# D.f = 1

# create parameter estimator object
E2 = srModel.ParameterEstimator(D)

# Initial vector
gamma = D.gamma
s0 = np.zeros((2,D.N))
theta0 = np.zeros(D.N)
v0 = srModel.vectorizeParameters(theta0, s0)

# vector of true parameters for comparisson
v_true = srModel.vectorizeParameters(D.theta, D.s)

# setup for the Newton CG

v = srModel.fmin_cg(fprime_L, f2prime_L, v0, i_max = 40, j_max = 40, errCG = 1e-8,
	errNR = 1e-8, n = 5, callback = cb_func)
