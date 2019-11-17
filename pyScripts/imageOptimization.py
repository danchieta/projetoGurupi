import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from PIL import Image

import vismodule
import srModel

inFolder = '../degradedImg/'
csv1 = 'paramsImage.csv'
csv2 = 'globalParams.csv'


# create Data object
D = srModel.Data(inFolder, csv1, csv2)
# create imge estimator object

# load the file with the estimated parameters
params_file = np.load('../results/resultdata/parameters_2018-12-16_190047.npz')

gamma = 2
theta = params_file['ta']
s = params_file['sa']

E2 = srModel.ImageEstimator(D, gamma, theta, s)

# initial value
x0 = np.ones(np.prod(D.getShapeHR()))*128
# initial vector is set to be random 
# x = np.random.randint(0, high = 256, size=(np.prod(D.getShapeHR()),1))

x, nfeval, rc = scipy.optimize.fmin_tnc(E2.getImageLikelihood, x0, fprime = E2.getImgLdiff, args = (-1.0,))

img = srModel.equalize_histogram(x).reshape(D.getShapeHR(), order = 'f')
imgr = Image.fromarray(img.astype(np.uint8))
vismodule.save_image(imgr)

plt.figure(1)
plt.imshow(imgr.convert('RGB'))
plt.show()
