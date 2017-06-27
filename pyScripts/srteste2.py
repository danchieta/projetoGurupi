import srModel
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

inFolder = '../degradedImg/'
csv1 = 'paramsImage.csv'
csv2 = 'globalParams.csv'

x = np.array(Image.open('../testIMG/imtestes.png').convert('L'))
x = np.hstack([np.zeros((x.shape[0],1), dtype='uint8'), x, np.zeros((x.shape[0],1), dtype='uint8')])
x = x.reshape(x.size, 1)

D = srModel.Data(inFolder, csv1, csv2)

E2 = srModel.ImageEstimator(D, D.gamma, D.theta, D.s)
P = [E2.getImageLikelihood(x)]

N = 20

span = 10
xtest = x + np.random.randint(0,high=span + 1, size=(x.size,N)) - span/2

for k in range(N):
	print 'iteration:', k+1
	P.append(E2.getImageLikelihood(xtest[:,k][np.newaxis].T))

print 'argmax P:', np.argmax(P)

if not np.argmax(P) == 0:
	MSE = np.sqrt(np.power(x-xtest[:,np.argmax(P) - 1], 2).mean())
	print 'RMSE:', MSE

	plt.imshow(xtest[:,np.argmax(P)- 1].atype(np.uint8).reshape(D.getShapeHR()))
	plt.show()

	imgr = Image.fromarray(xtest[:,np.argmax(P)- 1].astype(np.uint8).reshape(D.getShapeHR()))
	imgr.save('teste.bmp')