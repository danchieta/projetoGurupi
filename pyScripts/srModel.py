import numpy as np
from PIL import Image
from scipy import sparse
import scipy.sparse.linalg 
import genModel

def sub2ind(shape, vecOS):
	#converte vetor de subscritos para vetor de indices
	ind = []
	for i in range(vecOS.shape[1]):
		ind.append(vecOS[0,i]*shape[0]+vecOS[1,i])
	return np.array(ind)

def getVecL(shapeHR, shapeL):
	V = genModel.vecOfSub(shapeL) + np.array(shapeHR)[np.newaxis].T/2 - np.array(shapeL)[np.newaxis].T/2
	return V

def getWList(imageData, gamma, theta, s):
	shapei = imageData.shapeHR
	shapeo = np.round(shapei*imageData.f).astype('int')
	v = (shapei/2.0) #centro da imagem 
	W = []
	for k in range(imageData.N):
		W.append(genModel.psf(gamma, theta[k], s[:,k], shapei, shapeo, v))
	return W

def priorDist(shapeHR, windowSizeL = None, A = 0.04, r=1):
	# gera matriz de covariancia para funcao de probabilidade a priori da imagem HR
	# a ser estimada.
	if windowSizeL == None:
		vec_i = np.float16(genModel.vecOfSub(shapeHR))
	else:
		vec_i = getVecL(shapeHR, windowSizeL)

	print 'Computing covariance matrix of the prior distribution'
	Z = np.array([vec_i[0][np.newaxis].T - vec_i[0],
		vec_i[1][np.newaxis].T - vec_i[1]])
	Z = np.linalg.norm(Z,axis=0)
	Z = A*np.exp(-Z**2/r**2)

	print '   Computing log determinant'
	sign, detZ = np.linalg.slogdet(Z.astype(np.float32))

	print '   Computing inverse matrix'
	invZ = sparse.csc_matrix(np.linalg.inv(Z.astype(np.float)))
	return invZ, detZ/np.log(10.0)

def getSigma(W, invZ, beta, N):
	print 'Computing Sigma/covariance matrix of the posterior distribution'	
	Sigma = invZ

	for k in range(N):
		print '    iteration: ' + str(k+1) + '/' + str(N)
		Sigma = Sigma + beta*np.dot(W[k].T.toarray(),W[k].toarray())

	print '    Computing log determinant'
	sign, detSigma = np.linalg.slogdet(Sigma.astype(np.float32))

	return sparse.csc_matrix(Sigma), detSigma/np.log(10.0)

def getMu(W, imageData, Sigma):
	print 'Computing mu/mean vector of the posterior distribution'
	mu = sparse.csc_matrix(np.zeros((imageData.shapeHR.prod(),1)))

	for k in range(imageData.N):
		print '    iteration: ' + str(k+1) + '/' + str(imageData.N)
		y = sparse.csc_matrix(imageData.getImgVec(k))
		mu = mu + W[k].T*y

	return imageData.beta*(Sigma*mu)

def getloglikelihood(imageData, logDetSigma, W, invZ_x, logDetZ, mu):
	print 'Computing L/log likelihood function'
	
	beta = imageData.beta
	M = np.round(imageData.shapeHR*imageData.f).prod()
	L = np.dot(np.dot(mu.T.toarray(),invZ_x.toarray()),mu.toarray())
	L = L + logDetZ
	L = L - logDetSigma
	L = L - imageData.N*M*np.log10(imageData.beta)

	for k in range(imageData.N):
		print '    iteration: ' + str(k+1) + '/' + str(imageData.N)
		y = imageData.getImgVec(k)
		L = L + beta*np.linalg.norm(y - W[k]*mu)**2
	return -L[0,0]/2

class Estimator:
	def __init__(self, imageData, windowSizeL = None, A = 0.04, r=1):
		self.L = []
		self.imageData = imageData

		if windowSizeL == None:
			# if no window is provided the prior will use the whole image size
			self.windowSizeL = imageData.shapeLR
			self.windowSizeH = imageData.shapeHR
		else:
			# if a windows is provided, the prior will use just pixels on the center of the image
			self.windowSizeL = windowSizeL
			self.windowSizeH = np.divide(windowSizeL,imageData.f)

		self.invZ_x, self.logDetZ_x = priorDist(self.imageData.shapeHR, self.windowSizeH, A, r)

	def likelihood(self, gamma, theta, s):
		W = getWList(self.imageData, gamma, theta, s)
		Sigma, logDetSigma = getSigma(W, self.invZ_x, self.imageData.beta, self.imageData.N)
		mu = getMu(W, self.imageData, Sigma)
		del Sigma
		L = getloglikelihood(self.imageData, logDetSigma, W, self.invZ_x, self.logDetZ_x, mu)
		self.L.append(L)

		return L


class Data:
	def __init__(self,inFolder,csvfile1, csvfile2):
		self.inFolder = inFolder
		filename1 = inFolder + csvfile1
		filename2 = inFolder + csvfile2
		self.filename = np.genfromtxt(filename1, dtype=str, skip_header = 1, usecols = 0, delimiter = ';' ).tolist()
		self.s = np.genfromtxt(filename1, skip_header = 1, usecols = [1,2], delimiter = ';' ).T
		self.theta = np.genfromtxt(filename1, skip_header = 1, usecols = 3, delimiter = ';' )

		self.shapeHR = np.genfromtxt(filename2, skip_header = 1, usecols = [0,1], delimiter = ';' ).astype(int)
		self.beta = np.genfromtxt(filename2, skip_header = 1, usecols = 2, delimiter = ';' )
		self.f = np.genfromtxt(filename2, skip_header = 1, usecols = 3, delimiter = ';' )
		self.gamma = np.genfromtxt(filename2, skip_header = 1, usecols = 4, delimiter = ';' )
		self.N = np.asscalar(np.genfromtxt(filename2, skip_header = 1, usecols = 5, delimiter = ';' ).astype(int))
		self.shapeLR = Image.open(self.inFolder + self.filename[0]).size
	def getImgVec(self, index):
		img = np.array(Image.open(self.inFolder + self.filename[index]).convert('L'))
		d = np.array(img.shape)
		return img.reshape(d.prod(),1)