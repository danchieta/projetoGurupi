import numpy as np
import matplotlib.pyplot as plt
import srModel
import scipy.optimize

# por que voce me esquece e some?
# e se eu me interessar por alguem?
# e se ela de repente me ganha?

inFolder = '../degradedImg/'
csv1 = 'paramsImage.csv'
csv2 = 'globalParams.csv'

#load parameters from file
parFile = np.load('parameters 2017-10-01 18:28:06.552050.npz')

theta_a = parFile['theta_a']
s_a = parFile['s_a']

# create Data object
D = srModel.Data(inFolder, csv1, csv2)

err_theta = np.linalg.norm(D.theta - theta_a)
print 'Error theta:', err_theta

err_s = np.linalg.norm(D.s - s_a, axis=0)
print 'Mean of the error of s', err_s.mean()
print err_s[np.newaxis].T


plt.figure(1)
plt.scatter(D.s[0,:], D.s[1,:], marker = 'o', label = 'True shifts')
plt.scatter(s_a[0,:], s_a[1,:], marker = '^', label = 'Estimated shifts')
for k in range(D.N):
	plt.plot([D.s[0,k],s_a[0,k]],[D.s[1,k],s_a[1,k]], 'k--')
plt.legend(loc = 0)
plt.title('True shifts versus estimated shifts')
plt.show()

