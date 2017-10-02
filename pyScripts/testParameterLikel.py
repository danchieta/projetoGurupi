import numpy as np
import srModel


inFolder = '../degradedImg/'
csv1 = 'paramsImage.csv'
csv2 = 'globalParams.csv'

# create Data object
D = srModel.Data(inFolder, csv1, csv2)

# use just a small window of the image to compute parameters
# reducing computational cost
windowshape = (9,9)
D.setWindowLR(windowshape)

# create parameter estimator object
E2 = srModel.ParameterEstimator(D)

nsamples = 10
ntrials = 5


