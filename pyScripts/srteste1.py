import srModel
import numpy as np

inFolder = '../degradedImg/'
csv1 = 'paramsImage.csv'
csv2 = 'globalParams.csv'


D = srModel.Data(inFolder, csv1, csv2)
E1 = srModel.Estimator(D)
L = []

L.append(E1.likelihood(D.gamma, D.theta, D.s))

num = 3

gamma = np.random.rand(num)*num
theta = (np.random.rand(D.N,num)*16-8)*pi/180
s = np.random.rand(2,D.N,num)*4-2

for k in range(num):
	L.append(E1.likelihood(gamma[k], theta[:,k], s[:,:,k]))