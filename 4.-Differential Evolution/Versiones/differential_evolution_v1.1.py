from sre_constants import _NamedIntConstant
import numpy as np
np.set_printoptions(suppress=True)

class DifferentialEvolution:
    def __init__(self, func, callback, iterations, bounds, args, popsize, Cr):
        self.func = func
        self.callback = callback
        self.bounds = np.array(bounds)
        self.args = args
        self.popsize = popsize
        self.population = None
        self.fitness = None

        self.iter = iterations
        self.cr = Cr
        self.nfev = 0

    # class Format(object):
    #     def __init__(self,P,nit,fun,nfev):
    #         self.P = P
    #         self.nit = nit
    #         self.fun = fun
    #         self.nfev = nfev
        
    #     def __str__(self):
    #         return  str(self.__class__) + '\n'+ '\n'.join(('{} = {}'.format(item, self.__dict__[item]) for item in self.__dict__))

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

            # print(self.population[i,], self.fitness[i])

    def mutation(self):
        r1, r2, r3 = int(np.random.uniform(0,self.popsize)), int(np.random.uniform(0,self.popsize)), int(np.random.uniform(0,self.popsize))
        while r1 == r2 or r2 == r3 or r1 == r3:
            r1, r2, r3 = int(np.random.uniform(0,self.popsize)), int(np.random.uniform(0,self.popsize)), int(np.random.uniform(0,self.popsize))
        F = np.random.uniform(0,2)
        return self.population[r1] +  F*(self.population[r2] - self.population[r3])

    def crossover(self, x, v):
        u = np.copy(x)
        for i in range(len(x)):
            P = np.random.uniform(0,1)
            if P < self.cr:
                u[i] = v[i]
        return u
    
    def selection(self, fitness_x, fitness_u):
        if fitness_u > fitness_x:
            return True
        return False

    def solve(self):
        self.init_population()
        gen = 0

        while gen < self.iter:
            # print(gen)
            for i in range(self.popsize):
                v = self.mutation()
                u = self.crossover(self.population[i],v)

                for j in range(len(self.bounds)):
                    if(u[j] < self.bounds[j][0]): u[j] = self.bounds[j][0]
                    if(u[j] > self.bounds[j][1]): u[j] = self.bounds[j][1]


                fitness_u = self.func(u,*self.args)
                self.nfev += 1
                # print(self.population[0],self.fitness[0],"\n v = ",v,'\n u =',u,fitness_u)
                if fitness_u < self.fitness[i]:
                    self.population[i],self.fitness[i] = u,fitness_u
                # print(self.population[i],self.fitness[i])

            best_idx= np.argmin(self.fitness)
            best = self.population[best_idx]
            self.callback(best)
            print("Generation",gen)
            gen += 1

        best_idx= np.argmin(self.fitness)
        best = self.population[best_idx]

        return best, gen, self.fitness[best_idx], self.nfev

        # nit =0 
        # for i in range(self.iter):
        #     nit+=1
        #     mutants = self.mutation()
        #     self.survivor_selection(np.concatenate((self.population,mutants)))
        #     if self.fitness[0] == 0:break

        # # P: el arreglo con la solución
        # # nit: número de generaciones
        # # fun: fitness del mejor individuo al terminar la ejecución
        # # nfev: número de veces que se manda llamar la función func
        # P = self.population[0] 
        # fun = self.fitness[0]
        # nfev = self.nfev
        # return self.Format(P,nit,fun,nfev)

def differential_evolution(func, callback, iterations, bounds, args = (), popsize = 30, Cr = .5): # Cr = Posibilidad de Cruza
    ep = DifferentialEvolution(func, callback,  iterations, bounds, args, popsize, Cr)
    return ep.solve()