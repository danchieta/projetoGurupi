import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image

def vecOfSub(shape):
	V = np.zeros(shape)
	for x in range(shape[0]):
		for y in range(shape[1]):
			V[x,y] = (x,y)
	return V


f = 0.5

r = vecOfSub((3,3))

img = np.array(Image.open('../testIMG/imteste.png').convert('L'))

d = img.size
dd = np.round(d*f).astype('int')


imgr = Image.fromarray(img).convert('RGB')
imgr.save('res2.bmp')