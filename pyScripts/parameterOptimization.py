import numpy as np
import matplotlib.pyplot as plt
import srModel
import scipy.optimize
import datetime

norms = list()

def func(v):
	global norms
	n = np.linalg.norm(v-vtrue)
	norms = np.hstack([norms, n])
	print 'iteration'
	print 'Current norm:', n


inFolder = '../degradedImg/'
csv1 = 'paramsImage.csv'
csv2 = 'globalParams.csv'

# create Data object
D = srModel.Data(inFolder, csv1, csv2)

# use just a small window of the image to compute parameters
# reducing computational cost
windowshape = (7,7)
D.setWindowLR(windowshape)

# create parameter estimator object
E2 = srModel.ParameterEstimator(D)

P = list()

# defining initial parameters
gamma0 = 2 # tamanho da funcao de espalhamento de ponto
s0 = np.random.rand(2,D.N)*4-2 #deslocamento da imagem
theta0 = (np.random.rand(D.N)*8-4)*np.pi/180 #angulo de rotacao (com variancia de pi/100)

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
v, L, _, _, _ = scipy.optimize.fmin_cg(E2.vectorizedLikelihood, v0, args = (-1, gamma0, theta0), callback = func, maxiter = 10, full_output = 1)

P.append(L)

norms_step1 = norms
norms = list()

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
v, L, _, _, _ = scipy.optimize.fmin_cg(E2.vectorizedLikelihood, v0, args = (-1, gamma0), callback = func, maxiter = 15, full_output = 1)

P.append(L)

norms_step2 = norms
norms = list()

# norm of the error after algorithm
err_after = np.linalg.norm(v-vtrue)
print 'Error after algorithm:', err_after

# Unpack parameters 
theta_a, s_a = srModel.unvectorizeParameters(v, D.N, ('theta', 's'))
t_now = str(datetime.datetime.now())[0:-7].replace(':','')
np.savez('parameters '+t_now+'.npz', theta_a = theta_a, s_a = s_a,
	windowshape = np.array(windowshape), norms_step1 = norms_step1, norms_step2 = norms_step2)

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
plt.plot(P)

plt.figure(3)
plt.plot(norms_step1)
plt.title('Distance from expected result during optimization of $\mathbf{s}_k$')
plt.xlabel('iteration')

plt.figure(4)
plt.plot(norms_step2)
plt.title('Distance from expected result during optimization of $\mathbf{s}_k$ and $\theta$')
plt.xlabel('iteration')

plt.show()
