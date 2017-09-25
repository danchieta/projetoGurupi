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
windowshape = (10,10)
D.setWindowLR(windowshape)

# create parameter estimator object
E2 = srModel.ParameterEstimator(D)

# defining initial parameters
gamma0 = 2 # tamanho da funcao de espalhamento de ponto
s0 = np.random.rand(2,D.N)*8-4 #deslocamento da imagem
theta0 = (np.random.rand(D.N)*16-8)*np.pi/180 #angulo de rotacao (com variancia de pi/100)

#initial vector with the parameters
v0 = srModel.vectorizeParameters(theta0, s0)

# vector with true parameters
vtrue = srModel.vectorizeParameters(D.theta, D.s)

# norm of the error before algorithm
err_before = np.linalg.norm(v0-vtrue)
print 'Error before algorithm:', err_before

# save likelihood of the initial vector
P = [E2.vectorizedLikelihood(v0, 1.0, gamma = 2)]

# use cg to maximize likelihood
v = scipy.optimize.fmin_cg(E2.vectorizedLikelihood, v0, args = (-1, gamma0), callback = func, maxiter = 32)

# compute likelihood of the result
P.append(E2.vectorizedLikelihood(v, 1, gamma0))

# norm of the error after algorithm
err_after = np.linalg.norm(v-vtrue)
print 'Error after algorithm:', err_after

theta_a, s_a = srModel.unvectorizeParameters(v, D.N, ('theta', 's'))

err_theta = np.linalg.norm(D.theta - theta_a)
print 'Error theta:', err_theta

err_s = np.linalg.norm(D.s - s_a, axis=0)
print 'Mean of the error of s', err_s.mean()
print err_s[np.newaxis].T

plt.figure(1)
plt.plot(P)
plt.show()
