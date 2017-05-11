import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image

# function return vector of subscripts
def vecOfSub(shp):
	x,y = np.meshgrid(range(shp[0]),range(shp[1]))
	x = np.reshape(x,(1,x.size),order='f').squeeze()
	y = np.reshape(y,(1,y.size),order='f').squeeze()
	return np.array([x,y])

def psf(i, j, gamma, theta, s, shapei, shapeo, v):
	N = shapei.prod()
	M = shapeo.prod()

	R = np.array([[np.cos(theta) , np.sin(theta)],[-np.sin(theta), np.cos(theta)]]) 

	vec_j = vecOfSub(shapeo)
	vec_i = vecOfSub(shapei)

	v = np.repeat(v,M).reshape(2,M)
	s = np.repeat(s,M).reshape(2,M)

	vec_u = np.dot(R, (vec_j-v))+v+s

	W = -np.linalg.norm(vec_i[:,i] - vec_u[:,j])**2/gamma**2

	return W


img = np.array(Image.open('../testIMG/imteste.png').convert('L'))

f = 0.5 # fator de subamostragem
gamma = 2.0


d = np.array(img.shape)
dd = np.round(d*f).astype('int')

s = (0,0)
v = (dd/2.0).round()

theta = np.pi/6 #angulo de rotacao

W = psf(0,0,gamma, theta, s, d, dd, v)

imgr = Image.fromarray(img).convert('RGB')
#imgr.save('res2.bmp')