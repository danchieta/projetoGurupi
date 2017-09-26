import numpy as np
import matplotlib.pyplot as plt
import srModel
import scipy.optimize


def func(v):
	print 'iteration'
	print 'Current norm:', np.linalg.norm(v-vtrue)

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
s0 = np.random.rand(2,D.N)*8-4 #deslocamento da imagem
theta0 = (np.random.rand(D.N)*16-8)*np.pi/180 #angulo de rotacao (com variancia de pi/100)

# FIRST STEP: Optimize shifts
# ===========================
# initial vector with shifts
v0 = srModel.vectorizeParameters(s0)

# vector with true shifts
vtrue = srModel.vectorizeParameters(D.s)

# norm of the error before algorithm
err_before = np.linalg.norm(v0-vtrue)
print 'Error before shifts optimization:', err_before

# use cg to optimize shifts
v = scipy.optimize.fmin_cg(E2.vectorizedLikelihood, v0, args = (-1, gamma0, theta0), callback = func, maxiter = 10)

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
v = scipy.optimize.fmin_cg(E2.vectorizedLikelihood, v0, args = (-1, gamma0), callback = func, maxiter = 15)

# norm of the error after algorithm
err_after = np.linalg.norm(v-vtrue)
print 'Error after algorithm:', err_after

# Unpack parameters 
theta_a, s_a = srModel.unvectorizeParameters(v, D.N, ('theta', 's'))

err_theta = np.linalg.norm(D.theta - theta_a)
print 'Error theta:', err_theta

err_s = np.linalg.norm(D.s - s_a, axis=0)
print 'Mean of the error of s', err_s.mean()
print err_s[np.newaxis].T

plt.figure()
plt.scatter(D.s[0,:], D.s[1,:], marker = 'o', label = 'True shifts')
plt.scatter(s_a[0,:], s_a[1,:], marker = '^', label = 'Estimated shifts')
plt.show()
