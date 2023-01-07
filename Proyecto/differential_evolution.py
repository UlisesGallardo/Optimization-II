from sre_constants import _NamedIntConstant
import numpy as np
np.set_printoptions(suppress=True)

class DifferentialEvolution:
    def __init__(self, func, callback, iterations, bounds, args, popsize, Cr, rep):
        self.func = func
        self.bounds = np.array(bounds)
        self.callback = callback
        self.args = args
        self.popsize = popsize
        self.population = None
        self.fitness = None
        self.rep = rep

        self.iter = iterations
        self.cr = Cr
        self.nfev = 0

    def init_population(self):
        nvar = len(self.bounds)
        self.population = np.zeros((self.popsize,nvar))
        self.fitness = np.zeros((self.popsize))

        for v in range(nvar):
            vmin, vmax = self.bounds[v,0], self.bounds[v,1]
            self.population[:,v] = np.random.uniform(vmin,vmax,(self.popsize))

        for i in range(self.popsize):
            P = self.population[i]
            self.fitness[i] = self.func(P,*self.args)
            self.nfev+=1

    def mutation(self):
        r1, r2, r3 = int(np.random.uniform(0,self.popsize)), int(np.random.uniform(0,self.popsize)), int(np.random.uniform(0,self.popsize))
        while r1 == r2 or r2 == r3 or r1 == r3:
            r1, r2, r3 = int(np.random.uniform(0,self.popsize)), int(np.random.uniform(0,self.popsize)), int(np.random.uniform(0,self.popsize))
        F = np.random.uniform(0,2.1)
        return self.population[r1] +  F*(self.population[r2] - self.population[r3])

    def crossover(self, x, v):
        u = np.copy(x)
        for i in range(len(x)):
            P = np.random.uniform(0,1)
            if P < self.cr:
                u[i] = v[i]
        return u
    
    def selection(self, fitness_x, fitness_u):
        if fitness_u < fitness_x:
            return True
        return False

    def solve(self):
        r = 0
        while r<self.rep:
            self.init_population()
            gen = 0
            while gen < self.iter:
                for i in range(self.popsize):
                    v = self.mutation()
                    u = self.crossover(self.population[i],v)
                    
                    while(True):
                        v = self.mutation()
                        u = self.crossover(self.population[i],v)  
                        count = 0
                        for j in range(len(self.bounds)):
                            if((u[j] < self.bounds[j][0] or u[j] > self.bounds[j][1])): count+=1
                        if count == 0:break

                    fitness_u = self.func(u,*self.args)
                    self.nfev += 1

                    if fitness_u < self.fitness[i]:
                        self.population[i],self.fitness[i] = u,fitness_u
                gen += 1
            
                if self.callback !=None:
                    best_idx= np.argmin(self.fitness)
                    best = self.population[best_idx]
                    self.callback(best)
            r+=1
            print("rep: ",r, "/",self.rep)

        best_idx= np.argmin(self.fitness)
        best = self.population[best_idx]

        return best, gen, self.fitness[best_idx], self.nfev

def differential_evolution(func, iterations, bounds, callback = None, args = (), popsize = 2000, Cr = .9, rep=1): # Cr = Posibilidad de Cruza
    ep = DifferentialEvolution(func, callback, iterations, bounds, args, popsize, Cr, rep)
    return ep.solve()