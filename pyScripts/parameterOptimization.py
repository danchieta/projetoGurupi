#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import time
# The modules below were written by me
import srModel
import vismodule

norms = []
P = []
min_vectors = ()
min_params = ()
# gradients = []

def unpack_min_vectors(min_vectors, params):
	par_dict = {'gamma':gamma0, 'theta':theta0, 's':s0}
	
	if type(min_vectors) is not tuple:
		min_vectors = (min_vectors,)
	for par, vec in zip(params, min_vectors):
		par_dict[par] = vec
	
	return par_dict['gamma'], par_dict['theta'], par_dict['s']
	

def func_step(v):
	# function to be run after every iteration of the optimization algorithm
	global norms, P, min_vectors, min_params #, gradients
	# save norm of the error (compared to true parameters) of the current solution
	norms.append(np.linalg.norm(v-vtrue))

	# if the current error is minimum, save current solution
	if norms[-1] == min(norms):
		min_vectors = srModel.unvectorizeParameters(v, D.N, params)
		min_params = params
	
	if len(params) == 1:
		args = (gamma0, theta0)
	elif len(params) == 2:
		args = (gamma0,)
	elif len(params) == 3:
		args = ()
	# gradients.append(np.linalg.norm(scipy.optimize.approx_fprime(v, E2.vectorizedLikelihood, epsilon, 1.0, *args)))
	
	# save current likelihood
	P.append(E2.vectorizedLikelihood(v, 1, *args))
	print 'current error [' + str(len(norms)-1) + '] =', norms[-1]
	# print 'current gradient [' + str(len(gradients)-1) + '] =', gradients[-1]

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
gamma0 = 4 # tamanho da funcao de espalhamento de ponto
s0 = np.zeros((2,D.N)) #deslocamento da imagem
theta0 = np.zeros(D.N) #angulo de rotacao (com variancia de pi/100)

epsilon = 1.49012e-8 # norm of the step used in gradient approximation
s_bounds = [(-2,2)]*s0.size
theta_bounds = [(-4*np.pi/180,4*np.pi/180)]*theta0.size
gamma_bounds = [(1,7)]

maxfeval = [70,70,30]

nfeval = [0,]*3
rc = [0,]*3
niterations = []

# STEP 1: Optimize shifts
# ===========================
# initial vector with shifts
v0 = srModel.vectorizeParameters(s0)
params = ('s',) # parameters included in vector for optimization

vtrue = srModel.vectorizeParameters(D.s)

# run function on initial vector to save error norm, gradient and function evaluation
func_step(v0)

# norm of the error before algorithm
print 'Error before shifts optimization:', norms[0]

tic = time.time() #start counting time
# use Truncated Newton Nethod to optimize shifts
v, nfeval[0], rc[0] = scipy.optimize.fmin_tnc(E2.vectorizedLikelihood, v0,
	args = (-1, gamma0, theta0), approx_grad = True, bounds = s_bounds,
	maxfun = maxfeval[0], callback = func_step)

# recover s from the vector
s_a = srModel.unvectorizeParameters(v, D.N, params)

print 'Error after shifts optimization:', norms[-1]
niterations.append(len(norms)-1)

# STEP 2: Optimize shifts AND theta
# ===================================
# Build a new initial vector using the shifts from the previous step
v0 = srModel.vectorizeParameters(theta0, s_a)
params = ('theta','s') # parameters included in vector for optimization

# vector with true shifts and angles
vtrue = srModel.vectorizeParameters(D.theta, D.s)

func_step(v0)
print 'Error before shifts AND theta optimization:', norms[-1]

# use Truncated Newton Nethod to optimize shifts and rotation angles
v, nfeval[1], rc[1] = scipy.optimize.fmin_tnc(E2.vectorizedLikelihood, v0,
	args = (-1, gamma0), approx_grad = True, bounds = theta_bounds + s_bounds,
	maxfun = maxfeval[1] , callback = func_step)

theta_a, s_a = srModel.unvectorizeParameters(v, D.N, params)

print 'Error after shifts AND theta optimization:', norms[-1]
niterations.append(len(norms)-1-sum(niterations))

# STEP 3: Optimize shifts AND theta
# ===================================
# Build a new initial vector using the shifts from the previous step
v0 = srModel.vectorizeParameters(gamma0, theta_a, s_a)
params = ('gamma','theta','s') # parameters included in vector for optimization

# vector with true shifts and angles
vtrue = srModel.vectorizeParameters(D.gamma, D.theta, D.s)

func_step(v0)
print 'Error before all parameters optimization:', norms[-1]

# use Truncated Newton Nethod to optimize shifts and rotation angles
v, nfeval[2], rc[2] = scipy.optimize.fmin_tnc(E2.vectorizedLikelihood, v0, args = (-1,),
	approx_grad = True, bounds = gamma_bounds + theta_bounds + s_bounds,
	maxfun = maxfeval[2] , callback = func_step)

toc = time.time() - tic # stop counting time
print 'Elapsed time:', toc
niterations.append(len(norms)-1-sum(niterations))

# END OF OPTIMIZATION PROCESS
# ====================================
# Time to wrap things up for visualization.

# norm of the error after algorithm
print 'Error after optimization:', norms[-1]
P = np.array(P) # make array of list P
norms = np.array(norms)
# gradients = np.array(gradients)

# Unpack parameters 
gamma_a, theta_a, s_a = srModel.unvectorizeParameters(v, D.N, params)

# Unpack the parameters that became coser to correct solution
gamma_min, theta_min, s_min = unpack_min_vectors(min_vectors, min_params)

err_theta = np.linalg.norm(D.theta - theta_a)
print 'Error theta:', err_theta

err_s = np.linalg.norm(D.s - s_a, axis=0)
print 'Mean of the error of s', err_s.mean()
print err_s[np.newaxis].T

true_likelihood = E2.likelihood(D.gamma, D.theta, D.s)

vismodule.saveData(g0 = gamma0, s0 = theta0, t0 = theta0, ga = gamma_a, sa = s_a,
	ta = theta_a, gtrue = D.gamma, ttrue = D.theta, strue = D.s, P = P, norms = norms,
	ws = np.array(windowshape), niter = np.array(niterations), tl = true_likelihood,
	nfe = np.array(nfeval))

fig1, ax1 = vismodule.compareParPlot(s_a, D.s, np.abs(D.theta-theta_a)*180/np.pi,
	titlenote = u'[Máxima verossimilhança]' )
fig2, ax2 = vismodule.compareParPlot(s_min, D.s, np.abs(D.theta-theta_min)*180/np.pi,
	titlenote = u'[Menor erro encontrado]')

fig3, ax3 = vismodule.progressionPlot(P, norms, true_likelihood)
plt.show()
# fig4, ax4 = vismodule.simplePlot((gradients,), title = u'Progressão da norma do gradiente', xlabel = u'Iteração')

note = 'Vetor inicial: zeros \nJanela: '+str(windowshape)+'\nMaxfeval: '+str(maxfeval) + '\nElapsed time:'+str(toc)+'\nnfeval: '+str(nfeval)+'\nniterations: '+str(niterations)
vismodule.saveFigures(fig1, fig2, fig3, note = note, filetype = '.pdf') #, fig4)
