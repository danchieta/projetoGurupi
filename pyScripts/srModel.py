import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import sparse

def vecOfSub(shp):
	x,y = np.meshgrid(range(shp[0]),range(shp[1]))
	x = np.reshape(x,(1,x.size),order='f').squeeze()
	y = np.reshape(y,(1,y.size),order='f').squeeze()
	return np.array([x,y])

def priorDist(shapei, A=0.04, r=1):

	vec_i = vecOfSub(shapei)
	vec_i = np.complex64(vec_i[0] + vec_i[1]*1j)

	Z = np.meshgrid(vec_i, vec_i)
	Z = np.abs(Z[0] - Z[1]).astype(np.float16)

	return sparse.csc_matrix(A*np.exp(-(Z**2.)/r**2.))

def priorDist()



d_in  = (100,150)

Z = priorDist(d_in, A = 0.04)


#plt.spy(Z)
#plt.show()

# inFolder = '../degradedImg/' #diretorio de saida

# filename = np.genfromtxt(inFolder + 'paramsImage.csv', dtype=str, skip_header = 1, usecols = 0, delimiter = ';' )
# s = np.genfromtxt(inFolder + 'paramsImage.csv', skip_header = 1, usecols = [1,2], delimiter = ';' )
# theta = np.genfromtxt(inFolder + 'paramsImage.csv', skip_header = 1, usecols = 3, delimiter = ';' )
# gamma = np.genfromtxt(inFolder + 'paramsImage.csv', skip_header = 1, usecols = 4, delimiter = ';' )[0]
# f = np.genfromtxt(inFolder + 'paramsImage.csv', skip_header = 1, usecols = 5, delimiter = ';' )[0]

# img = np.array(Image.open(inFolder + filename[1]).convert('L'))
# d_in = np.round(img.shape/f).astype(int)

# Z = priorDist(d_in)

# plt.spy(Z)
# plt.show()