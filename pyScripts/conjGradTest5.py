import numpy as np
import matplotlib.pyplot as plt
import srModel

inFolder = '../degradedImg/'
csv1 = 'paramsImage.csv'
csv2 = 'globalParams.csv'

# create Data object
D = srModel.Data(inFolder, csv1, csv2)
# create imge estimator object
E2 = srModel.ImageEstimator(D, D.gamma, D.theta, D.s)

# initial value
x = np.ones((np.prod(D.getShapeHR()), 1))*128
# initial vector is set to be random 
# x = np.random.randint(0, high = 256, size=(np.prod(D.getShapeHR()),1))
xi = x #saving xi for further consults
# maximum number CG of iterations
i_max = 20
#maximum number of Newto-raphson iterations
j_max = 10
# CG error tolerance
errCG = 1e-3
# Newton-Raphson maximum number of iterations
errNR = 1e-3
# number of iterations to restart CG algorithm
n = 10

x = srModel.fmin_cg(E2.

img = x.reshape(D.getShapeHR(), order = 'f')
imgr = img.astype(np.uint8)

plt.figure(1)
plt.imshow(img)
plt.show()

plt.figure()
plt.plot(P)
plt.show(2)
