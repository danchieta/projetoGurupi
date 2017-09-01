import numpy as np
import matplotlib.pyplot as plt
import srModel
import scipy.optimize

inFolder = '../degradedImg/'
csv1 = 'paramsImage.csv'
csv2 = 'globalParams.csv'

# create Data object
D = srModel.Data(inFolder, csv1, csv2)
# create imge estimator object
E2 = srModel.ImageEstimator(D, D.gamma, D.theta, D.s)

xinit = 128*np.ones(np.prod(D.getShapeHR()))

x = scipy.optimize.fmin_cg(E2.getImageLikelihood, xinit, fprime = E2.getImgLdiff, maxiter = 5)

#plt.figure(1)
#plt.imshow(img)
#plt.show()

