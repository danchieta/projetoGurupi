import numpy as np
from PIL import Image
import genModel

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
		if covFile['A'] == A and covFile['r'] == r:
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
		print '    iteration: ' + str(k+1) + '/' + str(N)
		Sigma = Sigma + beta*np.dot(W[k].T,W[k])

	print '    Computing log determinant'
	sign, detSigma = np.linalg.slogdet(Sigma)

	return Sigma, detSigma

def getMu(W, imageData, Sigma):
	print 'Computing mu/mean vector of the posterior distribution'
	mu = np.zeros((imageData.getShapeHR().prod(),1))

	for k in range(imageData.N):
		print '    iteration: ' + str(k+1) + '/' + str(imageData.N)
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
		print '    iteration: ' + str(k+1) + '/' + str(imageData.N)
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
	return (-P/2.0)

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
		return imageLikelihood(self.imageData, x, self.W, self.logDetZ_x, self.invZ_x)


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
			return img.reshape(img.size,1)
		else:
			upperCorner = (np.zeros(2) - np.array(self.windowShapeLR)/2.0 + np.array(img.shape)/2.0).astype(int)
			lowerCorner = upperCorner + np.array(self.windowShapeLR)
			window = img[upperCorner[0]:lowerCorner[0],upperCorner[1]:lowerCorner[1]]
			return window.reshape(window.size,1) #.reshape(window.size,1)


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