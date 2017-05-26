import csv
import numpy as np

N = 50
gamma = np.random.randint(0, high=10, size=N)
s = np.random.randn(2,N)
theta = np.random.randn(N)*np.pi/180

with open('praramsImage.csv', 'wb') as csvfile:
	fields = ['filename','gamma','s','theta']
	
	plan = csv.DictWriter(csvfile, fieldnames=fields, delimiter=';')
	plan.writeheader()
	
	for k in range(N):
		plan.writerow({'filename':'image-' + str(k) + '.png',
			'gamma':gamma[k],
			's':s[:,k],
			'theta':theta[k]})
	#for k in range(N):
	#	filename = 'image-' + str(k) + '.png'
		
	
	