import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image
from IPython import embed


# function return vector of subscripts
def vecOfSub(shp):
	x,y = np.meshgrid(range(shp[0]),range(shp[1]))
	x = np.reshape(x,(1,x.size),order='f').squeeze()
	y = np.reshape(y,(1,y.size),order='f').squeeze()
	return np.array([x,y])

def psf(j, gamma, theta, s, shapei, shapeo, v):
	# numero de pixels em cada imagem
	N = shapei.prod()
	M = shapeo.prod()

	# Matriz de rotacao
	R = np.array([[np.cos(theta) , np.sin(theta)],[-np.sin(theta), np.cos(theta)]]) 

	#Vetores de subscritos das imagens de saida e entrada
	vec_j = vecOfSub(shapeo)
	vec_i = vecOfSub(shapei)
	
	v = v.reshape(2,1) #centro da imagem como vetor coluna
	s = np.array(s).reshape(2,1) #deslocamento como vetor coluna

	# Rotação e deslocamento de cada pixel da imagem HR
	vec_u = np.dot(R, (vec_j-v))+v+s

	# gerando uma linha da função de espalhamento de ponto
	vec_W = np.exp(-np.linalg.norm(vec_i - vec_u[:,j].reshape(2,1), axis=0)**2/gamma**2)

	# retorna linha da PSF normalizada
	return vec_W/vec_W.sum()


img = np.array(Image.open('../testIMG/imteste.png').convert('L'))

d = np.array(img.shape) #dimensoes da imagem de entrada

img = img.reshape(d.prod(),1)

f = 1 # fator de subamostragem
gamma = 5 # tamanho da funcao de espalhamento de ponto

dd = np.round(d*f).astype('int') #dimensoes da imagem de saida

s = (0,0) #deslocamento da imagem
v = (dd/2.0).round() #centro da imagem 

theta = 0 #angulo de rotacao

y = np.zeros(dd.prod())
#W = psf(0,gamma, theta, s, d, dd, v) #funcao de espalhamento de ponto


for i in range(dd.prod()):
	W = psf(i,gamma, theta, s, d, dd, v) #funcao de espalhamento de ponto
	y[i] = np.dot(W,img)
	print 100.0*i/dd.prod()
	
imgr = Image.fromarray(y.reshape(dd)).convert('RGB')
imgr.save('res2.bmp')