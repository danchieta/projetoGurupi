import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image

def vecOfSub(shp):
	x,y = np.meshgrid(range(shp[0]),range(shp[1]))
	x = np.transpose(x,(1,x.size),order='f')
	y = np.transpose(y,(1,y.size),order='f')
	return np.array([x,y])


f = 0.5

#r = vecOfSub((3,3))

img = np.array(Image.open('../testIMG/imteste.png').convert('L'))

d = img.size
dd = np.round(d*f).astype('int')


imgr = Image.fromarray(img).convert('RGB')
imgr.save('res2.bmp')