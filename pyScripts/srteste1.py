import srModel
import numpy as np

inFolder = '../degradedImg/'
csv1 = 'paramsImage.csv'
csv2 = 'globalParams.csv'


D = srModel.Data(inFolder, csv1, csv2)
D.setWindowLR((5,5))

E1 = srModel.Estimator(D)
L = []

L = E1.likelihood(D.gamma, D.theta, D.s)