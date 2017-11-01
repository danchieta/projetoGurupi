import numpy as np
from scipy.interpolate import interp1d
from random import shuffle
# sweet dreams are made of this
# who am I to disagree?
def sort_fit(fitness, pop):
	fitness, pop = (list(t) for t in zip(*sorted(zip(fitness, pop))) )
	return fitness, pop

def normalize(vec):
	vec = np.array(vec)
	vec = vec/sum(vec)
	return list(vec)

def issorted(vec):
	for i in range(1,len(vec)):
		if vec[i] < vec[i-1]:
			return False
	return True


def ini_pop(N, gen_function, args = None):
	pop = [gen_function(args) for i in range(N)]
	return pop

def crossover_mask(vec1, vec2):
	if (len(vec1) != len(vec2)):
		raise Exception("Size of the sequences must match!")

	if type(vec1) is not np.ndarray: 	
		vec1 = np.array(vec1)
	if type(vec2) is not np.ndarray:
		vec2 = np.array(vec2)

	m = np.random.randint(0,2,len(vec1))
	n = np.abs(m-1)
	
	child1 = vec1*m + vec2*n
	child2 = vec1*n + vec2*m

	return child1, child2

def evaluate(pop, evfunction):
	fitness = [evfunction(individual) for individual in pop]
	fitness, pop = sort_fit(fitness,pop)
	return fitness, pop

def select_wheel(fitness, pop):
	if not issorted(fitness):
		fitness, pop = sort_fit(fitness, pop)

	popi = list()
	fitnessi = list()
	# number of individuals in the intermediate population
	N = len(fitness)/2 - 1

	# select the best individual to the intermediate population
	# and remove her from the wheel
	popi.append(pop[-1])
	fitnessi.append(fitness[-1])
	del pop[-1]
	del fitness[-1]

	P = np.cumsum(normalize(fitness))

	for i in range(N):
		Pf = interp1d(P, range(len(fitness)))
		# select a random index based o the fitness of each individual
		index = int(np.round(Pf( np.random.rand()*(P[-1]-P[0])+P[0] )))

		popi.append(pop[index])
		fitnessi.append(fitness[index])

		del pop[index]
		del fitness[index]

		P = np.cumsum(normalize(fitness))

	return sort_fit(fitnessi, popi)

def mate(fitnessi, popi, evfunction):
	indexes = range(len(fitnessi))
	shuffle(indexes)

	if len(indexes)%2==1:
		indexes.append(indexes[0])
	
	indexes = zip(indexes[0:len(indexes)/2],indexes[len(indexes)/2::])
	childs = list()

	for i,j in indexes:
		child1, child2 = crossover_mask(popi[i], popi[j])
		childs = childs + [child1, child2]

	fitness_childs, childs = evaluate(childs, evfunction)

	return sort_fit(fitnessi + fitness_childs, popi + childs)
	
def mutate(fitness, pop, evfunction, gen_function, args_gen = None, rate = 0.01):
	if not issorted(fitness):
		fitness, pop = sort_fit(fitness,pop)
	# this vector helps to select the cromossomes that will suffer mutation
	vec_bool = np.random.rand(len(fitness)-1) <= rate

	for i in range(0,len(fitness)-1):
		if vec_bool[i]:
			j = np.random.randint(0,pop[i].size)
			pop[i][j] = gen_function(args_gen)[j]
			fitness[i] = evfunction(pop[i])

	return sort_fit(fitness, pop)

	
