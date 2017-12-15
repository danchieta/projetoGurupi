#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os

def compareParPlot(s, strue, thetaerror, titlenote = None):
	if titlenote is not None:
		titlenote = '\n'+titlenote
	else:
		titlenote = str()

	fig1, ax1 = plt.subplots(1,2, figsize = (9.5,4.75))
	fig1.subplots_adjust(right=.97, left = .08)
	ax1[0].scatter(strue[0,:], strue[1,:], marker = 'o', label = u'Valores reais')
	ax1[0].scatter(s[0,:], s[1,:], marker = '^', label = u'Valores estimados')
	for k in range(thetaerror.size):
		ax1[0].plot([strue[0,k],s[0,k]],[strue[1,k],s[1,k]], 'k--')
	ax1[0].legend(loc = 0)
	ax1[0].set_title(u'Comparação dos parâmetros de deslocamento'+titlenote)
	ax1[0].set_xlabel(u'Deslocamento horizontal')
	ax1[0].set_ylabel(u'Deslocamento vertical')

	cwidth = 0.75
	ax1[1].bar(np.arange(thetaerror.size), thetaerror, cwidth)
	xticks11 = ax1[1].set_xticks(range(thetaerror.size))
	ax1[1].set_title(u'Erro dos ângulos de rotação estimados'+titlenote)
	ax1[1].set_xlabel(u'Imagem')
	ax1[1].set_ylabel(u'Erro do ângulo estimado (graus)')

	return fig1, ax1

def progressionPlot(P, norms, Ptrue = None ):
	fig2, ax2 = plt.subplots(2,1, figsize = (6,7))
	fig2.subplots_adjust(hspace=.3, top = .92)
	if Ptrue is not None:
		ax2[0].plot(np.ones(P.size)*Ptrue, 'r-', label = 'Verossimilhança dos parâmetros reais'.decode('utf8'))
	ax2[0].plot(P, label = 'Verossimilhança dos parâmetros estimados'.decode('utf8'))
	ax2[0].set_title(u'Progressão do valor de verossimilhança durante\n a execução do algorítmo de gradientes conjugados')
	xticks2 = ax2[0].set_xticks(range(0,P.size,10))
	ax2[0].set_xlabel(u'Iteração')
	# plt.ylabel('$p(\gamma, \theta_k, \mathbf{s}_k | y)$ at iteration')
	ax2[0].legend(loc = 0)

	ax2[1].plot(norms)
	ax2[1].set_title(u'Distância para a solução correta')
	xticks3 = ax2[1].set_xticks(range(0,P.size,10))
	ax2[1].set_xlabel(u'Iteração')
	ax2[1].set_ylabel('$\|c_{atual} - c_{real}\|$')
	
	return fig2, ax2

def saveData(**kwargs):
	outFolder = 'resultVectors/'
	
	if not os.path.exists(outFolder):
		os.makedirs(outFolder)

	t_now = str(datetime.datetime.now())[0:-7].replace(':', str()).replace(' ', '_')
	filename = 'parameters_'+t_now+'.npz'
	np.savez(outFolder + filename, **kwargs)

def saveFigures(*args, **kwargs):
	outFolder = 'resultfigures/'

	if not os.path.exists(outFolder):
		os.makedirs(outFolder)
	
	t_now = str(datetime.datetime.now())[0:-7].replace(':', str()).replace(' ', '_')
	folderName = t_now + '/'

	os.makedirs(outFolder + folderName)

	try:
		extension = '.' + kwargs['filetype']
	except(KeyError):
		extension = str()
	
	i = 0
	for fig in args:
		fig.savefig(outFolder + folderName + 'figure_' + str(i) + extension)
		i+=1
