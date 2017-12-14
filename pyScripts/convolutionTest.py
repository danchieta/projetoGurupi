import numpy as np
from PIL import Image
import csv
import genModel
import os
import scipy.signal

def impulse(shape):
	center = tuple(np.round(np.array(shape)/2).astype(np.int))
	out = np.zeros(shape)
	out[center] = 1
	return out.reshape((np.prod(shape),1), order = 'f')

outFolder = '../degradedImg/' #diretorio de saida
outFormat = '.bmp' #formato de saida

imghr = np.array(Image.open('../testIMG/imtestes.png').convert('L'))

f = 0.25
gamma = 2
s = np.array([[2],[3]])
theta = 10*np.pi/180
beta = 400

shapei = imghr.shape
shapeo = tuple(np.array(shapei))
v = np.round(np.array(shapei)/2)

impulse_shape = (101,101)
impulse_shapeo = tuple(np.ceil(np.array(impulse_shape)*f).astype(np.int))

W = genModel.psf(gamma, theta, s, impulse_shape, impulse_shapeo, v)
unitimp = impulse(impulse_shape)

H = W.dot(unitimp)

imgi = Image.fromarray(unitimp.reshape(impulse_shape, order = 'f')*255)
imgo = Image.fromarray(H.reshape(impulse_shapeo, order = 'f')*255/H.sum())

resimg = scipy.signal.convolve2d(imghr, H/H.sum(), mode = 'same')
res = Image.fromarray(resimg)
