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
	P.append(E2.vectorizedLikelihood(v, 1, gamma = gamma0, s = s0))
	print 'iteration'
	print 'Current norm:', n


inFolder = '../degradedImg/'
csv1 = 'paramsImage.csv'
csv2 = 'globalParams.csv'

# create Data object
D = srModel.Data(inFolder, csv1, csv2)

# use just a small window of the image to compute parameters
# reducing computational cost
windowshape = (5,5)
D.setWindowLR(windowshape)
# D.f = 1

# create parameter estimator object
E2 = srModel.ParameterEstimator(D)

gamma0 = 2
# s0 = np.random.rand(2,D.N)*4-2
# theta0 = (np.random.rand(D.N)*8-4)*np.pi/180
s0 = D.s
theta0 = np.zeros(D.N)

# defining initial parameters
# v0 = np.load('parvect3.npy')
v0 = srModel.vectorizeParameters(theta0)


# Optimize just theta
# ===================

# Calculate the likelihood of the initial
P.append(E2.vectorizedLikelihood(v0, 1, gamma0, None, s0))

# vector with true shifts and angles
vtrue = srModel.vectorizeParameters(D.theta)

# norm of the error before algorithm
norms = np.hstack([norms, np.linalg.norm(vtrue-v0)])
print 'Error before shifts AND theta optimization:', norms[-1]

# Optimize rotations
v = scipy.optimize.fmin_cg(E2.vectorizedLikelihood, v0, args = (-1.0, gamma0, None, s0), callback = func, epsilon = 1e-3, maxiter = 90, gtol = 1.5e4)


# END OF CONJUGATE GRADIENTS ALGORITHM
# ====================================
# Time to wrap things up for visualization.

# norm of the error after algorithm
err_after = norms[-1]
print 'Error after algorithm:', err_after

# Unpack parameters 
theta_a = srModel.unvectorizeParameters(v, D.N, ('theta',))
theta_min = srModel.unvectorizeParameters(v_min, D.N, ('theta',))
s_a = s0
s_min = s0

err_theta = np.linalg.norm(D.theta - theta_a)
print 'Error theta:', err_theta

err_s = np.linalg.norm(D.s - s_a, axis=0)
print 'Mean of the error of s', err_s.mean()
print err_s[np.newaxis].T

P = -np.abs(np.array(P))

vismodule.saveData(g0 = gamma0, s0 = theta0, t0 = theta0, sa = s_a, ta = theta_a, P = P, norms = norms)

fig1, ax1 = vismodule.compareParPlot(s_a, D.s, np.abs(D.theta-theta_a)*180/np.pi)
fig2, ax2 = vismodule.compareParPlot(s_min, D.s, np.abs(D.theta-theta_min)*180/np.pi)

fig3, ax3 = vismodule.progressionPlot(P, norms, E2.likelihood(D.gamma, D.theta, D.s))
plt.show()

vismodule.saveFigures(fig1, fig2, fig3)
