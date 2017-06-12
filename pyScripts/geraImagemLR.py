import numpy as np
from PIL import Image
import csv
from genModel import *

outFolder = '../degradedImg/' #diretorio de saida
outFormat = '.bmp' #formato de saida

N = 10 #numero de imagens a serem 
img = np.array(Image.open('../testIMG/imtestes.png').convert('L')) #abre imagem a ser degradada
f = 0.25 # fator de subamostragem
gamma = 2 # tamanho da funcao de espalhamento de ponto
s = np.random.rand(2,N)*4-2 #deslocamento da imagem
theta = (np.random.rand(N)*8-4)*np.pi/180 #angulo de rotacao (com variancia de pi/100)
beta = 0.05

sigma = np.sqrt(1/beta) #desvio padrao do ruido
filename = [] #inicia lista com nomes de arquivo

for k in range(N):
	print 'gerando imgagem' + str(k)
	y = degradaImagem(img,gamma,theta[k],s[:,k],f)
	imgr = Image.fromarray(y).convert('RGB')
	filename.append('result-'+str(k)+outFormat)
	imgr.save(outFolder+filename[k])

#salva parametros em arquivo .csv
with open(outFolder + 'paramsImage.csv', 'wb') as csvfile:
	fields = ['filename','sx','sy','theta']
	
	plan = csv.DictWriter(csvfile, fieldnames=fields, delimiter=';')
	plan.writeheader()
	
	for k in range(N):
		plan.writerow({'filename':filename[k],
			'sx':s[0,k],
			'sy':s[1,k],
			'theta':theta[k]})

with open(outFolder + 'globalParams.csv', 'wb') as csvfile:
	fields = ['shapei0', 'shapei1', 'beta', 'f','gamma', 'N']
	
	plan = csv.DictWriter(csvfile, fieldnames=fields, delimiter=';')
	plan.writeheader()
	
	plan.writerow({'shapei0':img.shape[0],
		'shapei1': img.shape[1],
		'beta' : beta,
		'f':f,
		'gamma':2,
		'N':N})