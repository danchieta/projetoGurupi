import numpy as np

def ini_pop(N, gen_function, args = None):
	pop = [gen_function(args) for i in range(N)]
	return pop

def evaluate(pop, evfunction):
	fitness = [evfunction(individual) for individual in pop]
	return fitness
