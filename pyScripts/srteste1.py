import srModel
import numpy as np

inFolder = '../degradedImg/'
csv1 = 'paramsImage.csv'
csv2 = 'globalParams.csv'


D = srModel.Data(inFolder, csv1, csv2)
D.setWindowLR((20,20))

E1 = srModel.Estimator(D)
L = []

L = [E1.likelihood(D.gamma, D.theta, D.s)]

num = 3

expNum = 15
acertos = 0

for i in range(expNum):
	gamma = np.random.rand(num)*2 + 1
	theta = (np.random.rand(D.N,num)*16-8)*np.pi/180
	s = np.random.rand(2,D.N,num)*4-2

	for k in range(num):
		L.append(E1.likelihood(gamma[k], theta[:,k], s[:,:,k]))

	print L
	print np.argmax(L)
	if np.argmax(L) == 0:
		acertos += 1
	L = [L[0]]
	del gamma, theta, s

print 'acertos: ', acertos