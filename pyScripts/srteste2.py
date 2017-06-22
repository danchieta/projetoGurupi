import srModel
import numpy as np
import matplotlib.pyplot as plt

inFolder = '../degradedImg/'
csv1 = 'paramsImage.csv'
csv2 = 'globalParams.csv'


D = srModel.Data(inFolder, csv1, csv2)
D.setWindowLR((8,8))

E1 = srModel.Estimator(D)

L = []

gamma = np.linspace(0.5,4,8)

for k in range(gamma.size):
	L.append(E1.likelihood(gamma[k], D.theta, D.s))

plt.plot(gamma,L)
plo.show()