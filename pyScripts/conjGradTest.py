import numpy as np
import matplotlib.pyplot as plt

inFolder = '../degradedImg/'
csv1 = 'paramsImage.csv'
csv2 = 'globalParams.csv'


D = srModel.Data(inFolder, csv1, csv2)
D.setWindowLR((8,8))

E1 = srModel.ParameterEstimator(D)

# use ground true parameters
theta = D.theta
s = D.s

# set the parameters for the preconditioned scaled gradients algorithm
gamma_current = 0.4 # set initial value for gamma

cgerror = 0.5 # conjugate gradients error tolerance
nrerro = 0.5 # Newton-Raphson error tolerance

k = 0
i_max = 5 # max number of cg iterations
h = 0.2 # Step for numerical derivation

# r = dL/dgamma
r = (E1.likelihood(gamma_current + h, theta, s) - E1.likelihood(gamma_current, theta, s))/h
