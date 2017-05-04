import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image

def vecOfSub(shp):
	x,y = np.meshgrid(range(shp[0]),range(shp[1]))
	x = np.reshape(x,(1,x.size),order='f')
	y = np.reshape(y,(1,y.size),order='f')
	return np.array(x,y)


f = 0.5 # fator de subamostragem
theta = np.pi/6 #angulo de rotacao

R = np.array([[np.cos(theta) , np.sin(theta)],[-np.sin(theta) , np.cos(theta)] ])

#r = vecOfSub((3,3))

img = np.array(Image.open('../testIMG/imteste.png').convert('L'))

d = img.size
dd = np.round(d*f).astype('int')


imgr = Image.fromarray(img).convert('RGB')
#imgr.save('res2.bmp')