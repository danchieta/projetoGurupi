import numpy as np

def sort_fit(fitness, pop):
	fitness, pop = (list(t) for t in zip(*sorted(zip(fitness, pop))) )
	return fitness, pop

def normalize(vec):
	vec = np.array(vec)
	vec = vec/sum(vec)
	return list(vec)

def issorted(vec)
	for i in range(1,len(vec)):
		if vec[i] < vec[i-1]:
			return false
	return true


def ini_pop(N, gen_function, args = None):
	pop = [gen_function(args) for i in range(N)]
	return pop

def evaluate(pop, evfunction):
	fitness = [evfunction(individual) for individual in pop]
	fitness, pop = sort_fit(fitness,pop)
	return fitness, pop

def select_wheel(fitness, pop):
	if not issorted(fitness):
		fitness, pop = sort_fit(fitness, pop)
	
	N = len(fitness)
	P = np.cumsum(normalize(fitness))
