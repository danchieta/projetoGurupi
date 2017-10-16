#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import srModel

inFolder = '../degradedImg/'
csv1 = 'paramsImage.csv'
csv2 = 'globalParams.csv'

# create Data object
D = srModel.Data(inFolder, csv1, csv2)

ss = np.array(range(2,27,2))
L = np.array([])
Lr = np.array([])

gamma0= 2 # tamanho da funcao de espalhamento de ponto
s0 = np.random.rand(2,D.N)*4-2 #deslocamento da imagem
theta0 = (np.random.rand(D.N)*8-4)*np.pi/180 #angulo de rotacao (com variancia de pi/100)

for s in ss:
	# use just a small window of the image to compute parameters
	# reducing computational cost
	windowshape = (s,s)
	D.setWindowLR(windowshape)
	D.f = 1

	# create parameter estimator object
	E2 = srModel.ParameterEstimator(D)

	L = np.hstack((L, E2.likelihood(D.gamma, D.theta, D.s)))
	Lr = np.hstack((Lr, E2.likelihood(gamma0, theta0, s0)))


windowshape = (s,s)
D.f = 1

# create parameter estimator object
E2 = srModel.ParameterEstimator(D)

print 'full image likelihood. True parameters:', E2.likelihood(D.gamma, D.theta, D.s)
print 'full image likelihood. Random parameters:', E2.likelihood(gamma0, theta0, s0)

norm = np.linalg.norm(srModel.vectorizeParameters(D.theta, D.s) - srModel.vectorizeParameters(theta0, s0))

fig1, ax1 = plt.subplots()
ax1.plot(ss, L, label = 'Real parameters')
ax1.plot(ss,Lr, label = 'Estimated parameters')
ax1.legend(loc = 0)
ax1.set_title('Norm: '+ str(norm))

plt.show()

