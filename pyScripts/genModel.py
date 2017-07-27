#este modulo contem as funcoes referentes ao modelo de observacao
import numpy as np

# function return vector of subscripts
def decimate(shapei, shapeo):
	y,x = np.meshgrid(np.linspace(0,shapei[1]-1, shapeo[1]),
		np.linspace(0,shapei[0]-1, shapeo[0]))
	x = np.reshape(x,(1,x.size),order='f').squeeze()
	y = np.reshape(y,(1,y.size),order='f').squeeze()
	return np.array([x,y])

def vecOfSub(shp):
	y,x = np.meshgrid(range(shp[1]),range(shp[0]))
	x = np.reshape(x,(1,x.size),order='f').squeeze()
	y = np.reshape(y,(1,y.size),order='f').squeeze()
	return np.array([x,y])

def psf(gamma, theta, s, shapei, shapeo, v):
	# numero de pixels em cada imagem
	N = np.prod(shapei)
	M = np.prod(shapeo)

	# Matriz de rotacao
	R = np.array([[np.cos(theta) , np.sin(theta)],
		[-np.sin(theta), np.cos(theta)]]) 

	#Vetores de subscritos das imagens de saida e entrada
	vec_j = decimate(shapei,shapeo)
	vec_i = vecOfSub(shapei).astype(np.float16)
	
	v = v.reshape(2,1) #centro da imagem como vetor coluna
	s = np.array(s).reshape(2,1) #deslocamento como vetor coluna

	# Rotacao e deslocamento de cada pixel da imagem HR
	vec_u = np.dot(R, (vec_j-v))+v+s

	# gerando a matriz da funcao de espalhamento de ponto
	vec_W = np.array([vec_i[0] - vec_u[0][np.newaxis].T,
		vec_i[1] - vec_u[1][np.newaxis].T])
	vec_W = np.exp(-np.linalg.norm(vec_W, axis = 0)**2./gamma**2.)

	# retorna linha da PSF normalizada
	return vec_W/vec_W.sum(axis = 1)[np.newaxis].T

def degradaImagem(img, gamma, theta, s, f, sigma = 4, beta = 400):
	d = np.array(img.shape) #dimensoes da imagem de entrada
	img = img.reshape((d.prod(),1), order = 'f')
	dd = np.round(d*f).astype('int') #dimensoes da imagem de saida
	v = (d/2.0) #centro da imagem 
	y = np.zeros(dd.prod()) + np.random.randn(dd.prod(),1)*sigma #Vetor imagem resultante

	W = psf(gamma, theta, s, d, dd, v) #gera uma linha imgrfuncao de espalhamento de ponto
	y = np.dot(W,img) + np.sqrt(1/beta) #aplicacao da fucao de espalhamento de ponto
		#print 100.0*i/dd.prod()

	return y.reshape(dd, order = 'f')