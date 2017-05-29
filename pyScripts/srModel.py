import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy

def vecOfSub(shp):
	x,y = np.meshgrid(range(shp[0]),range(shp[1]))
	x = np.reshape(x,(1,x.size),order='f').squeeze()
	y = np.reshape(y,(1,y.size),order='f').squeeze()
	return np.array([x,y])

def priorDist(shapei, A = 1, r = 1):
	#retorna matriz de covariancia de distribuicao a priori para imagem HR

	vec_i = np.float16(vecOfSub(shapei))

	Z = np.array([
		np.meshgrid(vec_i[0],vec_i[0])[0] - np.meshgrid(vec_i[0],vec_i[0])[1],
		np.meshgrid(vec_i[1],vec_i[1])[0] - np.meshgrid(vec_i[1],vec_i[1])[1],
		]).astype(np.float16)

	return A*np.exp(-(np.linalg.norm(Z, axis=0)**2)/r**2.)

inFolder = '../degradedImg/' #diretorio de saida

filename = np.genfromtxt(inFolder + 'paramsImage.csv', dtype=str, skip_header = 1, usecols = 0, delimiter = ';' )
s = np.genfromtxt(inFolder + 'paramsImage.csv', skip_header = 1, usecols = [1,2], delimiter = ';' )
theta = np.genfromtxt(inFolder + 'paramsImage.csv', skip_header = 1, usecols = 3, delimiter = ';' )
gamma = np.genfromtxt(inFolder + 'paramsImage.csv', skip_header = 1, usecols = 4, delimiter = ';' )[0]
f = np.genfromtxt(inFolder + 'paramsImage.csv', skip_header = 1, usecols = 5, delimiter = ';' )[0]

img = np.array(Image.open(inFolder + filename[1]).convert('L'))
d_in = np.round(img.shape/f).astype(int)

Z = priorDist(d_in)

plt.spy(Z)
plt.show()

