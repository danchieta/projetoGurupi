import numpy as np
import matplotlib.pyplot as plt
import srModel

inFolder = '../degradedImg/'
csv1 = 'paramsImage.csv'
csv2 = 'globalParams.csv'


D = srModel.Data(inFolder, csv1, csv2)
D.setWindowLR((8,8))

E1 = srModel.ParameterEstimator(D)

# use ground true parameters
theta = D.theta
shift = D.s

# set the parameters for the preconditioned scaled gradients algorithm
gamma_current = 0.4 # set initial value for gamma

cgerror = 0.01 # conjugate gradients error tolerance

sigma0 = 0.1 # secant method step parameter
j_max = 10 # max number of secant iterations
scmerror = 0.3 # secant method error

i_max = 5 # max number of cg iterations
h = 0.2 # Step for numerical derivation

n = 10 # number of iterations to restart CG

i = 0
k = 0
# r = dL/dgamma
r = (E1.likelihood(gamma_current + h, theta, shift) - E1.likelihood(gamma_current, theta, shift))/h

# M is equal to the second order derivative of the likelihood function in gamma_current
M = -(E1.likelihood(gamma_current + h, theta, shift) - 2*E1.likelihood(gamma_current, theta, shift) + E1.likelihood(gamma_current - h, theta, shift))/(h**2)

s = r/M
d=s
delta_new = r*d
delta0 = delta_new

while i<i_max and delta_new > (cgerror**2)*delta0:
    print 'CG iteration', i + 1
    j = 0
    delta_d = d**2
    alpha = -sigma0
    # eta_prev is equal to the derivative of the likelihood function at gamma_current + sigma0*d
    eta_prev = -((E1.likelihood(gamma_current + sigma0*d + h, theta, shift) - E1.likelihood(gamma_current + sigma0*d, theta, shift))/h)*d
    
    while True:
        eta = -((E1.likelihood(gamma_current + h, theta, shift) - E1.likelihood(gamma_current, theta, shift))/h)*d
        alpha = alpha*eta/eta_prev-eta
        gamma_current = gamma_current + alpha*d
        eta_prev = eta
        j+=1
        if not (j < j_max and (alpha**2)*delta_d > scmerror**2):
            break
    r = (E1.likelihood(gamma_current + h, theta, shift) - E1.likelihood(gamma_current, theta, shift))/h
    delta_old = delta_new
    delta_mid = r*s
    M = -(E1.likelihood(gamma_current + h, theta, shift) - 2*E1.likelihood(gamma_current, theta, shift) + E1.likelihood(gamma_current - h, theta, shift))/(h**2)
    s = r/M
    delta_new = r*s
    beta = (delta_new - delta_mid)/delta_old
    k += 1
    if k == n or beta <= 0:
        d = s
        k = 0
    else:
        d = s + beta*d
    i = i+1
