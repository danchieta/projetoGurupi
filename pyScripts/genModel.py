import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image

def ind2sub(ind,shp):
	n = np.prod(shp)
	return divmod(ind, shp[0])
	

def vecOfSub(shp):
	n = prod(shp)
	V = np.zeros((2,n))
	for x in range(n):
		V[:,x]


f = 0.5

#r = vecOfSub((3,3))

img = np.array(Image.open('../testIMG/imteste.png').convert('L'))

d = img.size
dd = np.round(d*f).astype('int')


imgr = Image.fromarray(img).convert('RGB')
imgr.save('res2.bmp')