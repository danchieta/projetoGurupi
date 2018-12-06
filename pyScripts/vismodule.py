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
	ax1[0].scatter(strue[0,:], strue[1,:], marker = 'o', label = 'Valores reais')
	ax1[0].scatter(s[0,:], s[1,:], marker = '^', label = 'Valores estimados')
	for k in range(thetaerror.size):
		ax1[0].plot([strue[0,k],s[0,k]],[strue[1,k],s[1,k]], 'k--')
	ax1[0].legend(loc = 0)
	ax1[0].set_title('Comparação dos parâmetros de deslocamento'+titlenote)
	ax1[0].set_xlabel('Deslocamento horizontal')
	ax1[0].set_ylabel('Deslocamento vertical')

	cwidth = 0.75
	ax1[1].bar(np.arange(thetaerror.size), thetaerror, cwidth)
	xticks11 = ax1[1].set_xticks(list(range(thetaerror.size)))
	ax1[1].set_title('Erro dos ângulos de rotação estimados'+titlenote)
	ax1[1].set_xlabel('Imagem')
	ax1[1].set_ylabel('Erro do ângulo estimado (graus)')

	return fig1, ax1

def progressionPlot(P, norms, Ptrue = None ):
	fig2, ax2 = plt.subplots(1,2, figsize = (12,4))
	fig2.subplots_adjust(hspace=.3, top = .92, left=.05, right=.96)
	if Ptrue is not None:
		ax2[0].plot(np.ones(P.size)*Ptrue, 'r-', label = 'Verossimilhança dos parâmetros reais'.decode('utf8'))
	ax2[0].plot(P, label = 'Verossimilhança dos parâmetros estimados'.decode('utf8'))
	ax2[0].set_title('Progressão do valor de verossimilhança')
	if P.size <= 20:
		ticks = list(range(0,P.size))
	else:
		ticks = list(range(0,P.size, 10))

	xticks2 = ax2[0].set_xticks(ticks)
	ax2[0].set_xlabel('Iteração')
	# plt.ylabel('$p(\gamma, \theta_k, \mathbf{s}_k | y)$ at iteration')
	ax2[0].legend(loc = 0)

	ax2[1].plot(norms)
	ax2[1].set_title('Distância para a solução correta')
	xticks3 = ax2[1].set_xticks(ticks)
	ax2[1].set_xlabel('Iteração')
	ax2[1].set_ylabel('$\|c_{atual} - c_{real}\|$')
	
	return fig2, ax2

def saveData(**kwargs):
	if 'outFolder' in kwargs:
		outFolder = kwargs.pop('outFolder')
	else:
		outFolder = '../results/resultdata/'
	
	if not os.path.exists(outFolder):
		os.makedirs(outFolder)

	t_now = str(datetime.datetime.now())[0:-7].replace(':', str()).replace(' ', '_')
	filename = 'parameters_'+t_now+'.npz'
	np.savez(outFolder + filename, **kwargs)

def saveFigures(*args, **kwargs):
	if 'outFolder' in kwargs:
		outFolder = kwargs.pop('outFolder')
	else:
		outFolder = '../results/resultplots/'

	if not os.path.exists(outFolder):
		os.makedirs(outFolder)
	
	t_now = str(datetime.datetime.now())[0:-7].replace(':', str()).replace(' ', '_')
	folderName = t_now + '/'

	os.makedirs(outFolder + folderName)

	if 'note' in kwargs:
		note = file(outFolder+folderName+'note.txt', 'w')
		note.write(kwargs.pop('note'))
		note.close()

	if 'filetype' in kwargs:
		extension = '.' + kwargs['filetype']
	else:
		extension = str()
	
	i = 0
	for fig in args:
		fig.savefig(outFolder + folderName + 'figure_' + str(i) + extension)
		i+=1


def simplePlot(args, title=None, xlabel=None, ylabel=None):
	fig2, ax2 = plt.subplots(figsize = (6,7))
	if len(args) <= 2:
		ax2.plot(*args)
	else:
		raise Exception
	
	if title is not None:
		ax2.set_title(title)
	if xlabel is not None:
		ax2.set_xlabel(xlabel)
	if ylabel is not None:
		ax2.set_ylabel(ylabel)

	return fig2, ax2


def save_image(figure, outfolder = '../results/result_images/', extension = '.png'):
	if not os.path.exists(outfolder):
		os.makedirs(outfolder)
	t_now = str(datetime.datetime.now())[0:-7].replace(':', str()).replace(' ', '_')
	figure.save(outfolder+'result_'+t_now+extension)
