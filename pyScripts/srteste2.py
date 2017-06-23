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
# P = E2.getImageLikelihood(x)

