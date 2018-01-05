#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import srModel
import scipy.optimize
import vismodule

norms = []
P = []
min_vectors = ()
gradients = []

def func_step(v):
	# function to be run after every iteration of the optimization algorithm
	global norms, P, min_vectors, gradients
	# save norm of the error (compared to true parameters) of the current solution
	norms.append(np.linalg.norm(v-vtrue))

	# if the current error is minimum, save current solution
	if norms[-1] == min(norms):
		min_vectors = srModel.unvectorizeParameters(v, D.N, params)
	
	if len(params) == 1:
		args = (gamma0, theta0)
	elif len(params) == 2:
		args = (gamma0,)
	print 'args len:', len(args)
	gradients.append(np.linalg.norm(scipy.optimize.approx_fprime(v, E2.vectorizedLikelihood, 1.5e-8, 1.0, *args)))
	
	# save current likelihood
	P.append(E2.vectorizedLikelihood(v, 1, gamma0, theta0))
	print 'iteration'
	print 'Current norm:', norms[-1]

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

min_grad = 1.2e6 #CG algorithm should stop if gradient runs below this
epsilon = 1.49012e-8 # norm of the step used in gradient approximation

# FIRST STEP: Optimize shifts
# ===========================
# initial vector with shifts
v0 = srModel.vectorizeParameters(s0)
params = ('s') # parameters included in vector for optimization

vtrue = srModel.vectorizeParameters(D.s)

# run function on initial vector to save error norm, gradient and function evaluation
func_step(v0)

# norm of the error before algorithm
print 'Error before shifts optimization:', norms[0]

# use cg to optimize shifts
v = scipy.optimize.fmin_cg(E2.vectorizedLikelihood, v0, args = (-1, gamma0, theta0), callback = func_step, epsilon = epsilon, maxiter = 30, gtol = min_grad)

# recover s from the vector
s_a = srModel.unvectorizeParameters(v, D.N, ('s'))

print 'Error after shifts optimization:', norms[-1]

# STEP TWO: Optimize shifts AND theta
# ===================================
# Build a new initial vector using the shifts from the previous step
v0 = srModel.vectorizeParameters(theta0, s_a)
params = ('theta','s') # parameters included in vector for optimization

# vector with true shifts and angles
vtrue = srModel.vectorizeParameters(D.theta, D.s)

print 'Error before shifts AND theta optimization:', norms[-1]

# Optimize shifts and rotations
v = scipy.optimize.fmin_cg(E2.vectorizedLikelihood, v0, args = (-1, gamma0), callback = func_step, epsilon = epsilon, maxiter = 40, gtol = min_grad)

# END OF CONJUGATE GRADIENTS ALGORITHM
# ====================================
# Time to wrap things up for visualization.

# norm of the error after algorithm
print 'Error after algorithm:', norms[-1]
P = np.array(P) # make array of list P
norms = np.array(norms)
gradients = np.array(gradients)

# Unpack parameters 
theta_a, s_a = srModel.unvectorizeParameters(v, D.N, ('theta', 's'))

if type(min_vectors) is not tuple:
	s_min = min_vectors
	theta_min = theta0
else:
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
fig4, ax4 = vismodule.simplePlot((gradients,), title = u'Progressão da norma do gradiente', xlabel = u'Iteração')

vismodule.saveFigures(fig1, fig2, fig3, fig4)
