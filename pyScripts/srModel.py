import numpy as np
from PIL import Image
import genModel

def fmin_cg(fdiff, fdiff2, x0, i_max = 20, j_max = 10, errCG = 1e-3, errNR = 1e-3, n = 10, mode = 'min'):
	# find a value which minimizes a function f.
	# i_max - maximum number CG of iterations
	# j_max - maximum number of Newto-raphson iterations
	# errCG - CG error tolerance
	# errNR - Newton-Raphson maximum number of iterations
	# n - number of iterations to restart CG algorithm

	if mode == 'min':
		o = 1
	elif mode == 'max':
		o = -1
	else:
		raise

	# initial value
	x = x0
	# definition of CG gradioent variables
	i = 0
	k = 0
	r = -o*fdiff(x)
	d = r
	delta_new = r.T.dot(r)
	delta0 = delta_new

	while i<i_max and delta_new > (errCG**2.0)*delta0:
		print 'i =', i
		j = 0
		delta_d = d.T.dot(d)
		
		while True:
			print '    j =', j
			alpha = -(o*fidff(x).T.dot(d))/(d.T.dot(o*fdiff2(saveToDisk = True).dot(d)))
			x = x+alpha[0,0]*d
			j = j + 1
			if not(j<j_max and (alpha**2.0)*delta_d>errNR**2.0):
				break
		r = -o*fdiff(x)
		delta_old = delta_new
		delta_new = r.T.dot(r)
		beta = delta_new/delta_old
		d = r + beta*d
		k = k + 1
		if k == n or r.T.dot(d) <= 0:
			d = r
			k = 0
		i = i+1
	return x


def getWList(imageData, gamma, theta, s):
	shapei = imageData.getShapeHR()
	shapeo = np.round(shapei*imageData.f).astype('int')
	v = (shapei/2.0) #centro da imagem 
	W = []
	for k in range(imageData.N):
		W.append(genModel.psf(gamma, theta[k], s[:,k], shapei, shapeo, v))
	return W

def priorCovMat(shapeHR, A = 0.04, r=1, dtype='float64', savetoDisk = False):
	# gera matriz de covariancia para funcao de probabilidade a priori da imagem HR
	# a ser estimada.

	try:
		if not savetoDisk:
			raise Exception()
		covFile = np.load('priorCov.npz')
		if covFile['A'] == A and covFile['r'] == r and covFile['invZ'].shape[0]==np.prod(shapeHR):
			print 'Loading inverse covarianve matrix and determinant from disk.'
			detZ = covFile['detZ']
			invZ = covFile['invZ']
		else:
			raise Exception()

	except:
		print 'Computing covariance matrix of the prior distribution'
		vec_i = genModel.vecOfSub(shapeHR).astype(dtype)
		Z = np.array([vec_i[0][np.newaxis].T - vec_i[0],
			vec_i[1][np.newaxis].T - vec_i[1]])
		Z = np.linalg.norm(Z,axis=0)
		Z = A*np.exp(-Z**2/r**2)

		print '   Computing log determinant'
		sign, detZ = np.linalg.slogdet(Z)

		print '   Computing inverse matrix'
		invZ = np.linalg.inv(Z)

		if savetoDisk:
			print 'Saving covariance matrix to disk.'
			np.savez('priorCov.npz', invZ=invZ, detZ=detZ, A=A, r=r)
	return invZ, detZ

def getSigma(W, invZ, beta, N):
	print 'Computing Sigma/covariance matrix of the posterior distribution'	
	Sigma = invZ

	for k in range(N):
		# print '    iteration: ' + str(k+1) + '/' + str(N)
		Sigma = Sigma + beta*np.dot(W[k].T,W[k])

	print '    Computing log determinant'
	sign, detSigma = np.linalg.slogdet(Sigma)

	return Sigma, detSigma

def getMu(W, imageData, Sigma):
	print 'Computing mu/mean vector of the posterior distribution'
	mu = np.zeros((imageData.getShapeHR().prod(),1))

	for k in range(imageData.N):
		# print '    iteration: ' + str(k+1) + '/' + str(imageData.N)
		y = imageData.getImgVec(k)
		mu = mu + np.dot(W[k].T,y)

	return imageData.beta*(np.dot(Sigma,mu))

def getloglikelihood(imageData, logDetSigma, W, invZ_x, logDetZ, mu):
	print 'Computing L/log likelihood function'
	
	beta = imageData.beta
	M = np.round(imageData.getShapeHR()*imageData.f).prod()
	L = np.dot(np.dot(mu.T,invZ_x),mu)
	L = L + logDetZ
	L = L - logDetSigma
	L = L - imageData.N*M*np.log(imageData.beta)

	for k in range(imageData.N):
		# print '    iteration: ' + str(k+1) + '/' + str(imageData.N)
		y = imageData.getImgVec(k)
		L = L + beta*np.linalg.norm(y - np.dot(W[k],mu))**2
	return -L[0,0]/2

def imageLikelihood(imageData, x, W, logDetZ_x, invZ_x):
	# funcao calcula log(p(y|x,s,theta,gamma))
	beta = imageData.beta

	M = np.prod(imageData.getShapeLR())

	P = 0
	for k in range(imageData.N):
		P = P + np.linalg.norm(imageData.getImgVec(k)-W[k].dot(x))**2
	P = imageData.beta*P - imageData.N*M*np.log(beta/(2*np.pi))

	# adding the prior
	P = P + imageData.N*np.log(2*np.pi) + logDetZ_x + np.dot(x.T, invZ_x).dot(x)
	return (-P/2.0)[0,0]

def gradImageLikelihood(imageData, x, W, invZ_x):
	# calcula gradiente da funcao de verossimilhanca da imagem
	L = 0
	for k in range(imageData.N):
		L = L + np.dot(W[k].T, imageData.getImgVec(k) - np.dot(W[k],x))
	L = L - (invZ_x + invZ_x.T).dot(x)/2.0
	return L
class ParameterEstimator:
	def __init__(self, imageData, A = 0.04, r=1):
		self.L = []
		self.imageData = imageData
		self.invZ_x, self.logDetZ_x = priorCovMat(self.imageData.getShapeHR(), A, r)

	def likelihood(self, gamma, theta, s):
		W = getWList(self.imageData, gamma, theta, s)
		Sigma, logDetSigma = getSigma(W, self.invZ_x, self.imageData.beta, self.imageData.N)
		mu = getMu(W, self.imageData, Sigma)
		del Sigma
		L = getloglikelihood(self.imageData, logDetSigma, W, self.invZ_x, self.logDetZ_x, mu)
		self.L.append(L)

		return L

class ImageEstimator:
	def __init__(self, imageData, gamma, theta, s):
		self.imageData = imageData
		self.W = getWList(self.imageData, gamma, theta, s)

		self.invZ_x, self.logDetZ_x = priorCovMat(self.imageData.getShapeHR(), dtype = 'float32', savetoDisk=True)

	def getImageLikelihood(self, x):
		if (x.shape[0] == 1):
			# if x is a row vector
			return -imageLikelihood(self.imageData, x[np.newaxis].T, self.W, self.logDetZ_x, self.invZ_x)
		elif (x.shape[1] == 1):
			# if x is a column vector
			return -imageLikelihood(self.imageData, x, self.W, self.logDetZ_x, self.invZ_x)
		else:
			raise(Exception)

	def getImgLdiff(self,x):
		if (x.shape[0] == 1):
			# if x is a row vector
			return -gradImageLikelihood(self.imageData, x[np.newaxis].T, self.W, self.invZ_x).T.squeeze()

		elif (x.shape[1] == 1):
			# if x is a column vector
			return -gradImageLikelihood(self.imageData, x, self.W, self.invZ_x)
		else:
			raise(Exception)
	def getImgLdiff2(self, saveToDisk = False): 
		def calcImgdiff2(self):
			print 'Calculating second order differential'
			imgDiff2 = -(self.invZ_x.T + self.invZ_x)/2.0
			for k in range(self.imageData.N):
				imgDiff2 = imgDiff2 - self.W[k].T.dot(self.W[k])
			return imgDiff2

		try:
			return -self.imgDiff2
		except:
			if saveToDisk:
				try:
					# load matrix from file and return
					diff2File= np.load('diff2.npz')
					self.imgDiff2 = diff2File['imgDiff2']			
					print 'Second order differential loaded from disk'
					return -self.imgDiff2
				except:
					# calculate and save matrix to disk
					self.imgDiff2 = calcImgdiff2(self)# calculate diff2
					print 'Saving second order differential to disk.'
					np.savez('diff2.npz', imgDiff2=self.imgDiff2)
					return -self.imgDiff2
			else:
				self.imgDiff2 = calcImgdiff2(self)# calculate diff2
				return self.imgDiff2# return
class Data:
	def __init__(self,inFolder,csvfile1, csvfile2):
		self.inFolder = inFolder
		filename1 = inFolder + csvfile1
		filename2 = inFolder + csvfile2

		self.windowed = False

		self.filename = np.genfromtxt(filename1, dtype=str, skip_header = 1, usecols = 0, delimiter = ';' ).tolist()
		self.s = np.genfromtxt(filename1, skip_header = 1, usecols = [1,2], delimiter = ';' ).T
		self.theta = np.genfromtxt(filename1, skip_header = 1, usecols = 3, delimiter = ';' )

		self.beta = np.genfromtxt(filename2, skip_header = 1, usecols = 2, delimiter = ';' )
		self.f = np.genfromtxt(filename2, skip_header = 1, usecols = 3, delimiter = ';' )
		self.gamma = np.genfromtxt(filename2, skip_header = 1, usecols = 4, delimiter = ';' )
		self.N = np.asscalar(np.genfromtxt(filename2, skip_header = 1, usecols = 5, delimiter = ';' ).astype(int))
		self.shapeLR = np.array(Image.open(self.inFolder + self.filename[0]).convert('L')).shape

	def getImgVec(self, index):
		img = np.array(Image.open(self.inFolder + self.filename[index]).convert('L'))
		if not self.windowed:
			return img.reshape((img.size,1), order = 'f')
		else:
			upperCorner = (np.zeros(2) - np.array(self.windowShapeLR)/2.0 + np.array(img.shape)/2.0).astype(int)
			lowerCorner = upperCorner + np.array(self.windowShapeLR)
			window = img[upperCorner[0]:lowerCorner[0],upperCorner[1]:lowerCorner[1]]
			return window.reshape((window.size,1), order = 'f') #.reshape(window.size,1)
	def getImg(self, index):
		return Image.open(self.inFolder + self.filename[index]).convert('L')


	def setWindowLR(self, shape):
		self.windowShapeLR = shape
		self.windowed = True

	def getShapeLR(self):
		if  not self.windowed:
			return self.shapeLR
		else:
			return self.windowShapeLR

	def getShapeHR(self):
		if not self.windowed:
			return (self.shapeLR/self.f).astype(int)
		else:
			return (self.windowShapeLR/self.f).astype(int)
