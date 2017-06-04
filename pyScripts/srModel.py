import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import sparse
import scipy.sparse.linalg 
import genModel

def getWList(gamma, theta, s, shapei, f):
	shapeo = np.round(shapei*f).astype('int')
	v = (shapei/2.0) #centro da imagem 

	W = []

	for k in range(N):
		W.append(genModel.psf(gamma, theta[k], s[:,k], shapei, shapeo, v))

	return W


def getImgVec(filename):
	img = np.array(Image.open(inFolder + filename).convert('L'))
	d = np.array(img.shape)
	return img.reshape(d.prod(),1)

def readCSV(filename1, filename2):
	filename = np.genfromtxt(filename1, dtype=str, skip_header = 1, usecols = 0, delimiter = ';' ).tolist()
	s = np.genfromtxt(filename1, skip_header = 1, usecols = [1,2], delimiter = ';' ).T
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

def getSigma(W, Z, beta):
	Sigma = Z

	print 'N: ' + str(N)
	for k in range(N):
		print 'iteration: ' + str(k)
		Sigma = Sigma + beta*(W[k].T*W[k])

	return Sigma

def getMu(W, filename, Sigma, beta, shapei):
	mu = sparse.csc_matrix(np.zeros((shapei.prod(),1)))

	for k in range(N):
		print 'iteration: ' + str(k)
		y = sparse.csc_matrix(getImgVec(filename[k]))
		mu = mu + W[k].T*y
	return beta*(Sigma*mu)

def getloglikelihood(filename, Sigma, mu, beta, shapei,f):
	M = np.round(shapei*f).prod()
	L = mu.T*np.linalg.inv(Z_x.toarray())*mu
	#L = L + np.log10(np.linalg.det(Z_x.toarray()))
	L = L - np.log10(np.linalg.det(Sigma.toarray()))
	L = L - N*M*np.log10(beta)

	for k in range(N):
		y = getImgVec(filename[k])
		L = L + beta*np.linalg.norm(y - (W[k]*mu).toarray())**2
	return L


inFolder = '../degradedImg/'
csv1 = 'paramsImage.csv'
csv2 = 'globalParams.csv'

filename,s_true,theta_true,shapei,beta,f,gamma_true,N = readCSV(inFolder+csv1, inFolder+csv2)

gamma = 1
s = np.array([[3,4],[4,3]])
theta = np.array([4,4])*np.pi/180

print 'calculando psfs'
W = getWList(gamma, theta, s, shapei, f)

print 'calculando covariancia a priori'
Z_x = priorDist(shapei)

print 'calculando Sigma'
Sigma = getSigma(W, Z_x, beta)

# print 'calculando mu'
# mu = getMu(W, filename, Sigma, beta, shapei)

# print 'calculando L'
# L = getloglikelihood(filename, Sigma, mu, beta, shapei,f)

