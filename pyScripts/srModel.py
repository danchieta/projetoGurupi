import numpy as np
import matplotlib.pyplot as plt
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
	s = np.genfromtxt(filename1, skip_header = 1, usecols = [1,2], delimiter = ';' )
	theta = np.genfromtxt(filename1, skip_header = 1, usecols = 3, delimiter = ';' )

	shapei = np.genfromtxt(filename2, skip_header = 1, usecols = [0,1], delimiter = ';' ).astype(int)
	beta = np.genfromtxt(filename2, skip_header = 1, usecols = 2, delimiter = ';' )
	f = np.genfromtxt(filename2, skip_header = 1, usecols = 3, delimiter = ';' )
	gamma = np.genfromtxt(filename2, skip_header = 1, usecols = 4, delimiter = ';' )
	N = np.genfromtxt(filename2, skip_header = 1, usecols = 5, delimiter = ';' )

	return (filename,s,theta,shapei,beta,f,gamma,N)

def priorDist(shapei, A = 0.04, r=1):
	# gera matriz de covariancia para funcao de probabilidade a priori da imagem HR
	# a ser estimada.
	vec_i = np.float16(genModel.vecOfSub(shapei))
	Z = np.array([vec_i[0][np.newaxis].T - vec_i[0],
		vec_i[1][np.newaxis].T - vec_i[1]])
	Z = np.linalg.norm(Z,axis=0)
	Z = A*np.exp(-Z**2/r**2)

	return sparse.csc_matrix(Z)

def getSigma(Z, beta, gamma, theta, s, shapei, f):
	shapeo = np.round(shapei*f).astype('int')
	v = (shapei/2.0) #centro da imagem 
	Sigma = Z

	print 'N: ' + str(N)
	for k in range(N):
		print 'iteration: ' + str(k)
		W = genModel.psf(gamma, theta[k], s[:,k], shapei, shapeo, v)
		Sigma = Sigma + beta*(W.T*W)

	return Sigma

def getMu(filename, Sigma, beta, gamma, theta, s, shapei,f):
	shapeo = np.round(shapei*f).astype('int')
	v = (shapei/2.0) #centro da imagem 
	mu = sparse.csc_matrix(np.zeros((shapei.prod(),1)))

	for k in range(N):
		print 'iteration: ' + str(k)
		y = sparse.csc_matrix(getImgVec(filename[k]))
		W = genModel.psf(gamma, theta[k], s[:,k], shapei, shapeo, v)
		mu = mu + W.T*y
	return beta*(Sigma*mu)

def getloglikelihood(filename, Sigma, mu, beta, gamma, theta, s, shapei,f):
	shapeo = np.round(shapei*f).astype('int')
	v = (shapei/2.0) #centro da imagem 
	M = shapeo.prod()

	L = mu.T*np.linalg.inv(Z_x.toarray())*mu
	return L


inFolder = '../degradedImg/'
csv1 = 'paramsImage.csv'
csv2 = 'globalParams.csv'

filename,s,theta,shapei,beta,f,gamma,N = readCSV(inFolder+csv1, inFolder+csv2)

print 'calculando covariancia a priori'
Z_x = priorDist(shapei)

print 'calculando Sigma'
Sigma = getSigma(Z_x, beta, gamma, theta, s, shapei, f)

print 'calculando mu'
mu = getMu(filename, Sigma, beta, gamma, theta, s, shapei,f)

print 'calculando L'
L = getloglikelihood(filename, Sigma, mu, beta, gamma, theta, s, shapei,f)

