from sre_constants import _NamedIntConstant
import numpy as np
np.set_printoptions(suppress=True)

class EvolutionaryProgramming:
    def __init__(self, func, bounds, iterations=50, args = (), popsize = 30):
        self.func = func
        self.bounds = np.array(bounds)
        self.args = args
        self.popsize = popsize
        self.population = None
        self.fitness = None
        self.nfev = 0
        self.iter = iterations

    class Format(object):
        def __init__(self,P,nit,fun,nfev):
            self.P = P
            self.nit = nit
            self.fun = fun
            self.nfev = nfev
        
        def __str__(self):
            return  str(self.__class__) + '\n'+ '\n'.join(('{} = {}'.format(item, self.__dict__[item]) for item in self.__dict__))

    def init_population(self):
        nvar = len(self.bounds)
        self.population = np.zeros((self.popsize,nvar*2))
        self.fitness = np.zeros((self.popsize))

        for v in range(nvar):
            vmin, vmax = self.bounds[v,0], self.bounds[v,1]
            vmut = np.abs(vmax-vmin)/10
            self.population[:,v] = np.random.uniform(vmin,vmax,(self.popsize))
            self.population[:,v+nvar] = np.abs(np.random.normal(loc=0,scale=vmut,size=(self.popsize))) + 0.001

        for i in range(self.popsize):
            P = self.population[i,:nvar]
            self.fitness[i] = self.func(P,*self.args)
            self.nfev+=1

            #print(self.population[i,:nvar], self.fitness[i])

    def mutation(self, alpha = 0.2):
        nvar = len(self.bounds)
        mutants = np.copy(self.population)

        for v in range(nvar):
            mutants[:,v] += np.random.normal(0,mutants[:,v+nvar],(self.popsize))
            mutants[:,v+nvar] *= (1 + np.random.normal(0,alpha))

        return mutants

    def survivor_selection(self, offspring):
        nvar = len(self.bounds)
        fitness_off = np.zeros(len(offspring))

        for i in range(len(offspring)):
            P = offspring[i,:nvar]
            fitness_off[i] = self.func(P,*self.args)
            self.nfev+=1

        self.population = offspring[np.argsort(fitness_off)[:self.popsize]]
        self.fitness = np.sort(fitness_off)[:self.popsize]

    def solve(self):
        self.init_population()
        nit =0 
        for i in range(self.iter):
            nit+=1
            mutants = self.mutation()
            self.survivor_selection(np.concatenate((self.population,mutants)))
            if self.fitness[0] == 0:break

        # P: el arreglo con la solución
        # nit: número de generaciones
        # fun: fitness del mejor individuo al terminar la ejecución
        # nfev: número de veces que se manda llamar la función func
        P = self.population[0] 
        fun = self.fitness[0]
        nfev = self.nfev
        return self.Format(P,nit,fun,nfev)

def evolutionary_programming(func, bounds, iterations=50, args = (), popsize = 20):
    ep = EvolutionaryProgramming(func,bounds,iterations, args,popsize)
    return ep.solve()