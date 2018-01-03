#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import srModel
import scipy.optimize
import datetime
import vismodule

norms = np.array([])
P = list()
min_vectors = list()

def func_step(v):
	# function to be run after every iteration of the optimization algorithm
	global norms, P, min_vectors
	# save norm of the error (compared to true parameters) of the current solution
	n = np.linalg.norm(v-vtrue)
	norms = np.hstack([norms, n])

	# if the current error is minimum, save current solution
	if n == norms.min():
		min_vectors = srModel.unvectorizeParameters(v, D.N, params)
	
	# save current likelihood
	P.append(E2.vectorizedLikelihood(v, 1, gamma0, theta0))
	print 'iteration'
	print 'Current norm:', n

inFolder = '../degradedImg/'
csv1 = 'paramsImage.csv'
csv2 = 'globalParams.csv'

# create Data object
D = srModel.Data(inFolder, csv1, csv2)

# use just a small window of the image to compute parameters
# reducing computational cost
windowshape = (9,9)
D.setWindowLR(windowshape)

# create parameter estimator object
E2 = srModel.ParameterEstimator(D)

# defining initial parameters
gamma0 = 2 # tamanho da funcao de espalhamento de ponto
s0 = np.zeros((2,D.N)) #deslocamento da imagem
theta0 = np.zeros(D.N) #angulo de rotacao (com variancia de pi/100)

# FIRST STEP: Optimize shifts
# ===========================
# initial vector with shifts
v0 = srModel.vectorizeParameters(s0)
params = ('s0') # parameters included in vector for optimization

# vector with true shifts
vtrue = srModel.vectorizeParameters(D.s)

# norm of the error before algorithm
err_before = np.linalg.norm(v0-vtrue)
print 'Error before shifts optimization:', err_before

# Save the likelihood of the initial vector
P.append(E2.vectorizedLikelihood(v0, 1, gamma0, theta0))

# use cg to optimize shifts
v = scipy.optimize.fmin_cg(E2.vectorizedLikelihood, v0, args = (-1, gamma0, theta0), callback = func_step, epsilon = 1e-10, maxiter = 30)

# recover s from the vector
s_a = srModel.unvectorizeParameters(v, D.N, ('s'))
# norm of the error after algorithm
err_after = np.linalg.norm(v-vtrue)
print 'Error after shifts optimization:', err_after

# STEP TWO: Optimize shifts AND theta
# ===================================
# Build a new initial vector using the shifts from the previous step
v0 = srModel.vectorizeParameters(theta0, s_a)
params = ('theta','s0') # parameters included in vector for optimization

# vector with true shifts and angles
vtrue = srModel.vectorizeParameters(D.theta, D.s)

# norm of the error before algorithm
err_before = np.linalg.norm(v0-vtrue)
print 'Error before shifts AND theta optimization:', err_before

# Optimize shifts and rotations
v = scipy.optimize.fmin_cg(E2.vectorizedLikelihood, v0, args = (-1, gamma0), callback = func_step, epsilon = 1e-10, maxiter = 40)

# END OF CONJUGATE GRADIENTS ALGORITHM
# ====================================
# Time to wrap things up for visualization.

# norm of the error after algorithm
print 'Error after algorithm:', norms[-1]
P = np.array(P) # make array of list P

# Unpack parameters 
theta_a, s_a = srModel.unvectorizeParameters(v, D.N, ('theta', 's'))

if len(min_vectors) == 1:
	s_min = min_vectors
	theta_min = theta0
elif len(min_vectors) == 2:
	theta_min, s_min = min_vectors

err_theta = np.linalg.norm(D.theta - theta_a)
print 'Error theta:', err_theta

err_s = np.linalg.norm(D.s - s_a, axis=0)
print 'Mean of the error of s', err_s.mean()
print err_s[np.newaxis].T

vismodule.saveData(g0 = gamma0, s0 = theta0, t0 = theta0, sa = s_a, ta = theta_a, P = P, norms = norms, ws = np.array(windowshape))

fig1, ax1 = vismodule.compareParPlot(s_a, D.s, np.abs(D.theta-theta_a)*180/np.pi, titlenote = u'[Máxima verossimilhança]' )
fig2, ax2 = vismodule.compareParPlot(s_min, D.s, np.abs(D.theta-theta_min)*180/np.pi, titlenote = u'[Menor erro encontrado]')

fig3, ax3 = vismodule.progressionPlot(P, norms, E2.likelihood(D.gamma, D.theta, D.s))
plt.show()

vismodule.saveFigures(fig1, fig2, fig3)
