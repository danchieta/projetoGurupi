import numpy as np
from PIL import Image
import csv
import genModel
import os

outFolder = '../ece584Degraded/' #diretorio de saida
outFormat = '.png' #formato de saida

# cria pasta de saida caso ela nao exista
if not os.path.exists(outFolder):
    os.makedirs(outFolder)

N = 10 #numero de imagens a serem geradas
img = np.array(Image.open('../testIMG/letter.png').convert('L')) #abre imagem a ser degradada
f = 0.25 # fator de subamostragem
gamma = 2 # tamanho da funcao de espalhamento de ponto
s = np.random.rand(2,N)*4-2 #deslocamento da imagem
theta = (np.random.rand(N)*8-4)*np.pi/180 #angulo de rotacao (com variancia de pi/100)
beta = 400 # precisao = 1/variancia do ruido

filename = [] #inicia lista com nomes de arquivo

for k in range(N):
	print('gerando imgagem' + str(k))
	y = genModel.degradaImagem(img,gamma,theta[k],s[:,k],f,beta)
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
		'gamma': gamma,
		'N':N})
