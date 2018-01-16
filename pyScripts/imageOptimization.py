import numpy as np
import matplotlib.pyplot as plt
import srModel
import scipy.optimize
from PIL import Image

inFolder = '../degradedImg/'
csv1 = 'paramsImage.csv'
csv2 = 'globalParams.csv'


# create Data object
D = srModel.Data(inFolder, csv1, csv2)
# create imge estimator object

gamma = 2
theta = D.theta
s = D.s

E2 = srModel.ImageEstimator(D, gamma, theta, s)

# initial value
x0 = np.ones(np.prod(D.getShapeHR()))*128
# initial vector is set to be random 
# x = np.random.randint(0, high = 256, size=(np.prod(D.getShapeHR()),1))

x, nfeval, rc = scipy.optimize.fmin_tnc(E2.getImageLikelihood, x0, fprime = E2.getImgLdiff, args = (-1.0,))


img = D.getImgVec(0).min() + ((x - x.min())*(D.getImgVec(0).max() - D.getImgVec(0).min())/(x.max() - x.min())).reshape(D.getShapeHR(), order = 'f')
imgr = Image.fromarray(img.astype(np.uint8))
imgr.save('result.png')

plt.figure(1)
plt.imshow(imgr)
plt.show()
