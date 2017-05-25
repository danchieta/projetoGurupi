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

def degradaImagem(img, gamma, theta, s, f):
	d = np.array(img.shape) #dimensoes da imagem de entrada
	img = img.reshape(d.prod(),1)
	dd = np.round(d*f).astype('int') #dimensoes da imagem de saida
	v = (dd/2.0).round() #centro da imagem 
	y = np.zeros(dd.prod()) #Vetor imagem resultante

	for i in range(dd.prod()):
		W = psf(i,gamma, theta, s, d, dd, v) #gera uma linha imgrfuncao de espalhamento de ponto
		y[i] = np.dot(W,img) #aplicacao da fucao de espalhamento de ponto
		print 100.0*i/dd.prod()

	return y.reshape(dd)


N = 30 #numero de imagens a serem geradas
img = np.array(Image.open('../testIMG/imtestes.png').convert('L'))
f = 0.9 # fator de subamostragem
gamma = 4 # tamanho da funcao de espalhamento de ponto
s = np.random.randn(2,N) #deslocamento da imagem
theta = np.random.randn(N)*2*np.pi/100 #angulo de rotacao (com variancia de pi/100)

for k in range(N):	
	print k
	y = degradaImagem(img,gamma,theta[k],s[:,k],f)
	imgr = Image.fromarray(y).convert('RGB')
	imgr.save('../resultIMG/result-'+str(k)+'.png')

#imgr.save('res2.bmp')

