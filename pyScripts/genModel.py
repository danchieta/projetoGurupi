import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image

from IPython.core.debugger import Tracer


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
	
	
	v = np.repeat(v,M).reshape(2,M)
	s = np.repeat(s,M).reshape(2,M)

	vec_u = np.dot(R, (vec_j-v))+v+s

	vec_W = np.array([])
	
	for k in range(N):
		vec_W = np.append(vec_W, -np.linalg.norm(vec_i[:,k] - vec_u[:,j])**2/gamma**2)

	return vec_W/vec_W.sum()


img = np.array(Image.open('../testIMG/imteste.png').convert('L'))

d = np.array(img.shape) #dimensoes da imagem de entrada

img = img.reshape(d.prod(),1)

f = 0.5 # fator de subamostragem
gamma = 2.0 # tamanho da funcao de espalhamento de ponto

dd = np.round(d*f).astype('int') #dimensoes da imagem de saida

s = (0,0) #deslocamento da imagem
v = (dd/2.0).round() #centro da imagem 

theta = np.pi/6 #angulo de rotacao

y = np.zeros(dd.prod())

for i in range(dd.prod()):
	W = psf(i,gamma, theta, s, d, dd, v) #funcao de espalhamento de ponto
	y[i] = np.dot(W,img)



#imgr = Image.fromarray(img).convert('RGB')
#imgr.save('res2.bmp')