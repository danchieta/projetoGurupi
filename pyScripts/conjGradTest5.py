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

# The algorithm is meant to find the minimum of a function. Since we're trying to maximize a function, we adjusted the method to find the minimum of -L(x) 

# definition of CG gradioent variables
i = 0
k = 0
r = -E2.getImgLdiff(x)
d = r
delta_new = r.T.dot(r)
delta0 = delta_new

# start tracking the likelihood of x
P = [E2.getImageLikelihood(x)]

while i<i_max and delta_new > (errCG**2.0)*delta0:
	print 'i =', i
	j = 0
	delta_d = d.T.dot(d)
	
	while True:
		print '    j =', j
		alpha = -(E2.getImgLdiff(x).T.dot(d))/(d.T.dot(E2.getImgLdiff2(saveToDisk = True).dot(d)))
		x = x+alpha[0,0]*d
		j = j + 1
		if not(j<j_max and (alpha**2.0)*delta_d>errNR**2.0):
			break
	P.append(E2.getImageLikelihood(x))
	r = -E2.getImgLdiff(x)
	delta_old = delta_new
	delta_new = r.T.dot(r)
	beta = delta_new/delta_old
	d = r + beta*d
	k = k + 1
	if k == n or r.T.dot(d) <= 0:
		d = r
		k = 0
	i = i+1

img = x.reshape(D.getShapeHR(), order = 'f')
imgr = img.astype(np.uint8)

plt.figure(1)
plt.imshow(img)
plt.show()

plt.figure()
plt.plot(P)
plt.show(2)
