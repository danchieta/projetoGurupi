#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import srModel
import scipy.optimize
import datetime

norms = np.array([])
P = list()

def func_step1(v):
	global norms
	global P
	n = np.linalg.norm(v-vtrue)
	norms = np.hstack([norms, n])
	P.append(E2.vectorizedLikelihood(v, 1, gamma0, theta0))
	print 'iteration'
	print 'Current norm:', n

def func_step2(v):
	global norms
	global P
	n = np.linalg.norm(v-vtrue)
	norms = np.hstack([norms, n])
	P.append(E2.vectorizedLikelihood(v, 1, gamma0))
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

# vector with true shifts
vtrue = srModel.vectorizeParameters(D.s)

# norm of the error before algorithm
err_before = np.linalg.norm(v0-vtrue)
print 'Error before shifts optimization:', err_before

# Save the likelihood of the initial vector
P.append(E2.vectorizedLikelihood(v0, 1, gamma0, theta0))

# use cg to optimize shifts
v = scipy.optimize.fmin_cg(E2.vectorizedLikelihood, v0, args = (-1, gamma0, theta0), callback = func_step1, epsilon = 1e-10, maxiter = 10)

# recover s from the vector
s_a = srModel.unvectorizeParameters(v, D.N, ('s'))
# norm of the error after algorithm
err_after = np.linalg.norm(v-vtrue)
print 'Error after shifts optimization:', err_after

# STEP TWO: Optimize shifts AND theta
# ===================================
# Build a new initial vector using the shifts from the previous step
v0 = srModel.vectorizeParameters(theta0, s_a)

# vector with true shifts and angles
vtrue = srModel.vectorizeParameters(D.theta, D.s)

# norm of the error before algorithm
err_before = np.linalg.norm(v0-vtrue)
print 'Error before shifts AND theta optimization:', err_before

# Optimize shifts and rotations
v = scipy.optimize.fmin_cg(E2.vectorizedLikelihood, v0, args = (-1, gamma0), callback = func_step2, epsilon = 1e-10, maxiter = 30)

# norm of the error after algorithm
print 'Error after algorithm:', norms[-1]

P = np.array(P)

# Unpack parameters 
theta_a, s_a = srModel.unvectorizeParameters(v, D.N, ('theta', 's'))
t_now = str(datetime.datetime.now())[0:-7].replace(':','')
np.savez('parameters '+t_now+'.npz', theta_a = theta_a, s_a = s_a,
	windowshape = np.array(windowshape), norms = norms, P = P)

err_theta = np.linalg.norm(D.theta - theta_a)
print 'Error theta:', err_theta

err_s = np.linalg.norm(D.s - s_a, axis=0)
print 'Mean of the error of s', err_s.mean()
print err_s[np.newaxis].T

fig1, ax1 = plt.subplots(1,2)
ax1[0].scatter(D.s[0,:], D.s[1,:], marker = 'o', label = u'Valores reais')
ax1[0].scatter(s_a[0,:], s_a[1,:], marker = '^', label = u'Valores estimados')
for k in range(D.N):
	ax1[0].plot([D.s[0,k],s_a[0,k]],[D.s[1,k],s_a[1,k]], 'k--')
ax1[0].legend(loc = 0)
ax1[0].set_title(u'Comparação entre deslocamentos estimados e deslocamentos reais')

ax1[1].bar(np.arange(D.N), D.theta, 0.4, label = u'Valores reais')
ax1[1].bar(np.arange(D.N)+0.4, theta_a, 0.4, label = u'Valores estimados')
ax1[1].legend(loc = 0)

fig2, ax2 = plt.subplots()
ax2.plot(np.ones(P.size)*E2.likelihood(D.gamma, D.theta, D.s), 'r-', label = 'Verossimilhança dos parâmetros reais'.decode('utf8'))
ax2.plot(P, label = 'Verossimilhança dos parâmetros estimados'.decode('utf8'))
ax2.set_title(u'Progressão do valor de verossimilhança durante\n a execução do algorítmo de gradientes conjugados', y = 1.0)
xticks2 = ax2.set_xticks(range(0,P.size, 10))
ax2.set_xlabel(u'Iteração')
# plt.ylabel('$p(\gamma, \theta_k, \mathbf{s}_k | y)$ at iteration')
ax2.legend(loc = 0)

fig3, ax = plt.subplots()
plt.plot(norms)
plt.title(u'Distância para a solução correta')
plt.xlabel(u'Iteração')
plt.ylabel('$|v_{atual} - v_{real}|$')

plt.show()
