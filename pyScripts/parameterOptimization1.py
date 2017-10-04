import numpy as np
import matplotlib.pyplot as plt
import srModel
import scipy.optimize
import datetime

norms = np.array([])
P = list()

def func(v):
	global norms
	global P
	n = np.linalg.norm(v-vtrue)
	norms = np.hstack([norms, n])
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

# create parameter estimator object
E2 = srModel.ParameterEstimator(D)


# defining initial parameters
gamma0 = 2 # tamanho da funcao de espalhamento de ponto
s0 = np.random.rand(2,D.N)*4-2 #deslocamento da imagem
theta0 = (np.random.rand(D.N)*8-4)*np.pi/180 #angulo de rotacao (com variancia de pi/100)

# Optimize shifts AND theta
# =========================
# Build a new initial vector using the shifts from the previous step
v0 = srModel.vectorizeParameters(theta0, s0)

# Calculate the likelihood of the initial
P.append(E2.vectorizedLikelihood(v0, 1, gamma0))

# vector with true shifts and angles
vtrue = srModel.vectorizeParameters(D.theta, D.s)

# norm of the error before algorithm
norms = np.hstack([norms, np.linalg.norm(vtrue-v0)])
print 'Error before shifts AND theta optimization:', norms[-1]

# Optimize shifts and rotations
v = scipy.optimize.fmin_cg(E2.vectorizedLikelihood, v0, args = (-1, gamma0), callback = func, maxiter = 20)

# END OF CONJUGATE GRADIENTS ALGORITHM
# ====================================
# Time to wrap things up for visualization.

# norm of the error after algorithm
err_after = norms[-1]
print 'Error after algorithm:', err_after

# Unpack parameters 
theta_a, s_a = srModel.unvectorizeParameters(v, D.N, ('theta', 's'))

err_theta = np.linalg.norm(D.theta - theta_a)
print 'Error theta:', err_theta

err_s = np.linalg.norm(D.s - s_a, axis=0)
print 'Mean of the error of s', err_s.mean()
print err_s[np.newaxis].T

P = -np.abs(np.array(P))

plt.figure(1)
plt.scatter(D.s[0,:], D.s[1,:], marker = 'o', label = 'True shifts')
plt.scatter(s_a[0,:], s_a[1,:], marker = '^', label = 'Estimated shifts')
for k in range(D.N):
	plt.plot([D.s[0,k],s_a[0,k]],[D.s[1,k],s_a[1,k]], 'k--')
plt.legend(loc = 0)
plt.title('True shifts versus estimated shifts')

plt.figure(2)
plt.plot(np.ones(P.size)*E2.likelihood(D.gamma, D.theta, D.s), 'r-', label = 'Likelihood of the true parameters')
plt.plot(P)
plt.title('Progression of the likelihood of current solution during CG algorithm', y = 1.05)
plt.xlabel('iteration')
plt.ylabel('$p(\gamma, \theta_k, \mathbf{s}_k | y)$ at iteration')
plt.legend(loc = 0)

plt.figure(3)
plt.plot(norms)
plt.title('Distance from correct solution')
plt.xlabel('iteration')
plt.ylabel('$|v_{current} - v_{true}|$')

plt.show()
