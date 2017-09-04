import numpy as np
import matplotlib.pyplot as plt
import srModel
import scipy.optimize, scipy.misc
from PIL import Image

# set where to look for degraded images and data
inFolder = '../degradedImg/'
csv1 = 'paramsImage.csv'
csv2 = 'globalParams.csv'

def funcIter(x):
	print 'iteration'

	P.append(E2.getImageLikelihood(x))

def histEq(x, nbins = 257):
	h, bins = np.histogram(x, range(nbins))
	T = ((nbins - 2.0)/x.size)*np.cumsum(h)
	xt = np.interp(x, range(256), h)

# load original HR image for comparisson
xtrue = np.array(Image.open('../testIMG/imtestes.png').convert('L'))
xtrue = np.hstack([xtrue.mean(axis = 1)[np.newaxis].T, xtrue, xtrue.mean(axis = 1)[np.newaxis].T])
xtrue = xtrue.reshape(xtrue.size, order = 'f')


P = [] # initiate list 

# create Data object
D = srModel.Data(inFolder, csv1, csv2)

# load and resample one of the LR images to use as staarting point 
x0 = np.array(D.getImg(0).resize(D.getShapeHR().dot([[0,1],[1,0]]), resample = 3)).reshape((np.prod(D.getShapeHR()),1), order = 'f')

xinit = 128*np.ones(np.prod(D.getShapeHR()))
rmse1 = np.sqrt(np.sum((xinit-xtrue)**2.0))
print 'RMSE of the initial:', rmse1

rmse2 = np.sqrt(np.sum((x0-xtrue)**2.0))
print 'RMSE of the resampled inmage:', rmse2

# create imge estimator object
E2 = srModel.ImageEstimator(D, D.gamma, D.theta, D.s)

P.append(E2.getImageLikelihood(xinit))

x = scipy.optimize.fmin_cg(E2.getImageLikelihood, x0, fprime = E2.getImgLdiff, callback = funcIter)

rmse3 = np.sqrt(np.sum((xtrue - x)**2.0))
print 'RMSE of final image:', rmse3

plt.imshow(x.reshape(D.getShapeHR(), order = 'f'))
plt.show()
