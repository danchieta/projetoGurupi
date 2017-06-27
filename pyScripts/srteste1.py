import srModel
import numpy as np

inFolder = '../degradedImg/'
csv1 = 'paramsImage.csv'
csv2 = 'globalParams.csv'


D = srModel.Data(inFolder, csv1, csv2)
D.setWindowLR((8,8))

E1 = srModel.ParameterEstimator(D)

L = [E1.likelihood(D.gamma, D.theta, D.s)]

num = 5
expNum = 15

acertos = 0

for i in range(expNum):
	gamma = D.gamma
	theta = (np.random.rand(D.N,num)*16-8)*np.pi/180
	s = np.random.rand(2,D.N,num)*8-4

	for k in range(num):
		L.append(E1.likelihood(gamma, theta[:,k], s[:,:,k]))

	if np.argmax(L) == 0:
		acertos += 1	
	print L
	print np.argmax(L)

	L = [L[0]]
	del theta,s

print acertos