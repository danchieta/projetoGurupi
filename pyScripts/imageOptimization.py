import numpy as np
import matplotlib.pyplot as plt
import srModel
import scipy.optimize
from PIL import Image

inFolder = '../degradedImg/'
csv1 = 'paramsImage.csv'
csv2 = 'globalParams.csv'

# load the file with the parameters
pfile = np.load('parameters 2017-10-12 220712.npz')

# create Data object
D = srModel.Data(inFolder, csv1, csv2)
# create imge estimator object

gamma = 2
theta = pfile['theta_a']
s = pfile['s_a']

E2 = srModel.ImageEstimator(D, gamma, theta, s)

# initial value
x0 = np.ones(np.prod(D.getShapeHR()))*128
# initial vector is set to be random 
# x = np.random.randint(0, high = 256, size=(np.prod(D.getShapeHR()),1))
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

#x = srModel.fmin_cg(E2.getImgLdiff, E2.getImgLdiff2, x0)
x = scipy.optimize.fmin_ncg(E2.getImageLikelihood, x0, fprime=E2.getImgLdiff, fhess=E2.getImgLdiff2, args = (-1.0,), epsilon = 1e-36, avextol = 1e-30)

img = (D.getImgVec(1).min()+(D.getImgVec(1).max()-D.getImgVec(1).min())*(x-x.min())/(x.max()-x.min())).reshape(D.getShapeHR(), order = 'f')
imgr = Image.fromarray(img.astype(np.uint8))
imgr.save('result.png')

plt.figure(1)
plt.imshow(imgr)
plt.show()
