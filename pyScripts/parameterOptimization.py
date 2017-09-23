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
gamma0 = 4 # tamanho da funcao de espalhamento de ponto
s0 = np.random.rand(2,D.N)*8-4 #deslocamento da imagem
theta0 = (np.random.rand(D.N)*16-8)*np.pi/180 #angulo de rotacao (com variancia de pi/100)

#initial vector with the parameters
v0 = srModel.vectorizeParameters(gamma0, theta0, s0)

# vector with true parameters
vtrue = srModel.vectorizeParameters(D.gamma, D.theta, D.s)

# norm of the error before algorithm
err_before = np.linalg.norm(v0-vtrue)
print 'Error before algorithm:', err_before

# save likelihood of the initial vector
P = [E2.vectorizedLikelihood(v0)]

# use cg to maximize likelihood
v = scipy.optimize.fmin_cg(E2.vectorizedLikelihood, v0, args = (-1,), callback = func, maxiter = 15)

# compute likelihood of the result
P.append(E2.vectorizedLikelihood(v))

# norm of the error after algorithm
err_after = np.linalg.norm(v-vtrue)
print 'Error after algorithm:', err_after

plt.figure(1)
plt.plot(P)
plt.show()
