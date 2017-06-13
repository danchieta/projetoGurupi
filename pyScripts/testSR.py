inFolder = '../degradedImg/'
csv1 = 'paramsImage.csv'
csv2 = 'globalParams.csv'

filename,s_true,theta_true,shapei,beta,f,gamma_true,N = readCSV(inFolder+csv1, inFolder+csv2)

Z_x, invZ_x, logDetZ = priorDist(shapei)

gamma = gamma_true
s = [np.random.rand(2,N)*10-5, s_true, np.random.rand(2,N)*10-5, np.random.rand(2,N)*10-5]
theta = [(np.random.rand(N)*16-8)*np.pi/180, theta_true, (np.random.rand(N)*16-8)*np.pi/180,(np.random.rand(N)*16-8)*np.pi/180]

likelihood = []

for k in range(4):
	print 'start iteration: ', k+1
	W = getWList(gamma, theta[k], s[k], shapei, f)

	Sigma, logDetSigma = getSigma(W, invZ_x, beta)

	mu = getMu(W, filename, Sigma, beta, shapei)

	L = getloglikelihood(filename, Sigma, mu, beta, shapei, f, logDetZ, logDetSigma)

	del W,Sigma,mu
	likelihood.append(L)