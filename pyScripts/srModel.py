import numpy as np
from PIL import Image
from scipy import sparse
import scipy.sparse.linalg 
import genModel

def getImgVec(filename):
	img = np.array(Image.open(inFolder + filename).convert('L'))
	d = np.array(img.shape)
	return img.reshape(d.prod(),1)

def readCSV(filename1, filename2):
	filename = np.genfromtxt(filename1, dtype=str, skip_header = 1, usecols = 0, delimiter = ';' ).tolist()
	s = np.genfromtxt(filename1, skip_header = 1, usecols = [1,2], delimiter = ';' ).T
	theta = np.genfromtxt(filename1, skip_header = 1, usecols = 3, delimiter = ';' )

	shapeHR = np.genfromtxt(filename2, skip_header = 1, usecols = [0,1], delimiter = ';' ).astype(int)
	beta = np.genfromtxt(filename2, skip_header = 1, usecols = 2, delimiter = ';' )
	f = np.genfromtxt(filename2, skip_header = 1, usecols = 3, delimiter = ';' )
	gamma = np.genfromtxt(filename2, skip_header = 1, usecols = 4, delimiter = ';' )
	N = np.asscalar(np.genfromtxt(filename2, skip_header = 1, usecols = 5, delimiter = ';' ).astype(int))

	return (filename,s,theta,shapeHR,beta,f,gamma,N)

def priorDist(shapeHR, A = 0.04, r=1):
	# gera matriz de covariancia para funcao de probabilidade a priori da imagem HR
	# a ser estimada.
	print 'Computing covariance matrix of the prior distribution'
	vec_i = np.float16(genModel.vecOfSub(shapeHR))
	Z = np.array([vec_i[0][np.newaxis].T - vec_i[0],
		vec_i[1][np.newaxis].T - vec_i[1]])
	Z = np.linalg.norm(Z,axis=0)
	Z = A*np.exp(-Z**2/r**2)

	print '   Computing log determinant'
	sign, detZ = np.linalg.slogdet(Z.astype(np.float32))

	print '   Computing inverse matrix'
	invZ = sparse.csc_matrix(np.linalg.inv(Z.astype(np.float)))
	return sparse.csc_matrix(Z), invZ, detZ/np.log(10.0)

def getSigma(W, invZ, beta):
	print 'Computing Sigma/covariance matrix of the posterior distribution'	
	Sigma = invZ

	print 'N: ' + str(N)
	for k in range(N):
		print '    iteration: ', str(k+1), '/', str(N)
		Sigma = Sigma + beta*np.dot(W[k].T.toarray(),W[k].toarray())

	print '    Computing log determinant'
	sign, detSigma = np.linalg.slogdet(Sigma.astype(np.float32))

	return sparse.csc_matrix(Sigma), detSigma/np.log(10.0)

def getMu(W, filename, Sigma, beta, shapeHR):
	print 'Computing mu/mean vector of the posterior distribution'
	mu = sparse.csc_matrix(np.zeros((shapeHR.prod(),1)))

	for k in range(N):
		print '    iteration: ' + str(k)
		y = sparse.csc_matrix(getImgVec(filename[k]))
		mu = mu + W[k].T*y

	return beta*(Sigma*mu)

def getloglikelihood(filename, beta, shapeHR, f, gamma, s, theta):
	print 'Computing L/log likelihood function'
	
	M = np.round(shapeHR*f).prod()
	L = np.dot(np.dot(mu.T.toarray(),invZ_x.toarray()),mu.toarray())
	L = L + logDetZ
	L = L - logDetSigma
	L = L - N*M*np.log10(beta)

	for k in range(N):
		print '    iteration: ' + str(k+1) + '/' + str(N)
		y = getImgVec(filename[k])
		L = L + beta*np.linalg.norm(y - W[k]*mu)**2
	return -L[0,0]/2

class Estimator:
	def setWList(self,gamma, theta, s, shapeHR, f):
		# gera matriz do sistema (funao espalhamento de ponto) para cara imagem
		# para os parametros fornecidos
		print 'Computing W matrices'
		shapeLR = np.round(shapeHR*f).astype('int')
		v = (shapeHR/2.0) #centro da imagem 
		self.W = []

		for k in range(N):
			print '    iteration: ', str(k+1), '/', str(N)
			self.W.append(genModel.psf(gamma, theta[k], s[:,k], shapeHR, shapeLR, v))

class LikelihoodFunction:

	def __init__(self, W, beta, shapeHR, A = 0.04, r = 1):
		self.Z_x = priorDist(shapeHR, A, r)

