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
v_min = None

def func(v):
	global norms, P, v_min
	n = np.linalg.norm(v-vtrue)
	norms = np.hstack([norms, n])
	if n == norms.min():
		v_min = v
	P.append(E2.vectorizedLikelihood(v, 1, gamma = gamma0))
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
# D.f = 1

# create parameter estimator object
E2 = srModel.ParameterEstimator(D)

gamma0 = 2
s0 = np.random.rand(2,D.N)*4-2
theta0 = np.random.rand(D.N)*8-4

# defining initial parameters
# v0 = np.load('parvect3.npy')
v0 = srModel.vectorizeParameters(theta0, s0)


# Optimize shifts AND theta
# =========================
# Build a new initial vector using the shifts from the previous step

# Calculate the likelihood of the initial
P.append(E2.vectorizedLikelihood(v0, 1, gamma0))

# vector with true shifts and angles
vtrue = srModel.vectorizeParameters(D.theta, D.s)

# norm of the error before algorithm
norms = np.hstack([norms, np.linalg.norm(vtrue-v0)])
print 'Error before shifts AND theta optimization:', norms[-1]

# Optimize shifts and rotations
v = scipy.optimize.fmin_cg(E2.vectorizedLikelihood, v0, args = (-1.0, gamma0), callback = func, epsilon = 1e-10, maxiter = 60)

# END OF CONJUGATE GRADIENTS ALGORITHM
# ====================================
# Time to wrap things up for visualization.

# norm of the error after algorithm
err_after = norms[-1]
print 'Error after algorithm:', err_after

# Unpack parameters 
theta_a, s_a = srModel.unvectorizeParameters(v, D.N, ('theta', 's'))
theta_min, s_min = srModel.unvectorizeParameters(v_min, D.N, ('theta', 's'))

err_theta = np.linalg.norm(D.theta - theta_a)
print 'Error theta:', err_theta

err_s = np.linalg.norm(D.s - s_a, axis=0)
print 'Mean of the error of s', err_s.mean()
print err_s[np.newaxis].T

P = -np.abs(np.array(P))

fig1, ax1 = vismodule.compareParPlot(s_a, D.s, np.abs(D.theta-theta_a))
fig2, ax2 = vismodule.compareParPlot(s_min, D.s, np.abs(D.theta-theta_min))

fig3, ax3 = vismodule.progressionPlot(P, norms, E2.likelihood(D.gamma, D.theta, D.s))
plt.show()
fig1.savefig('./outfig/figure_1.png')
fig2.savefig('./outfig/figure_2.png')
fig3.savefig('./outfig/figure_3.png')
