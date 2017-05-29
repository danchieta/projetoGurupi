import numpy as np
from PIL import Image
import csv

inFolder = '../degradedImg/' #diretorio de saida

filename = np.genfromtxt(inFolder + 'paramsImage.csv', dtype=str, skip_header = 1, usecols = 0, delimiter = ';' )
s = np.genfromtxt(inFolder + 'paramsImage.csv', skip_header = 1, usecols = [1,2], delimiter = ';' )
theta = np.genfromtxt(inFolder + 'paramsImage.csv', skip_header = 1, usecols = 3, delimiter = ';' )
gamma = np.genfromtxt(inFolder + 'paramsImage.csv', skip_header = 1, usecols = 4, delimiter = ';' )[0]
f = np.genfromtxt(inFolder + 'paramsImage.csv', skip_header = 1, usecols = 5, delimiter = ';' )[0]

