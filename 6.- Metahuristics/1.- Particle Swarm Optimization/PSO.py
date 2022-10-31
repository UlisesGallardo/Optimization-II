from ast import arg
from sqlite3 import Time
from sre_constants import _NamedIntConstant
import numpy as np
np.set_printoptions(suppress=True)

class PSO:
    def __init__(self, func, iterations, bounds, args, N, rep):
        self.func = func
        self.bounds = np.array(bounds)
        self.args = args
        self.N = N
        self.X = None #Position of particles
        self.fitness = None
        self.V = None #Velocities
        self.Xbest = None
        self.best_fitness = None
        self.best_global  = None
        self.best_global_fitness = None

        self.hyper_parameters = None #w, c1,c2

        self.iter = iterations
        self.nfev = 0
        self.rep = rep


        self.A = [[1,2],
            [3,2]]
        self.b = [ 80,
            120]
        self.bounds_1 = [[0,None],
                [0,None]]
        self.c = [-20000,-15000]


    def init_particles(self):
        nvar = len(self.bounds)
        self.X,self.V,self.Xbest = [np.zeros((self.N,nvar)),np.zeros((self.N,nvar)),np.zeros((self.N,nvar))]
        self.fitness,self.best_fitness = [np.zeros((self.N)),np.zeros((self.N))]
        self.best_global =np.zeros((1,nvar))

        w = np.random.uniform(0.7,1)
        self.hyper_parameters = np.zeros(3)
        self.hyper_parameters[:] = w

        for v in range(nvar):
            vmin, vmax = self.bounds[v,0], self.bounds[v,1]
            
            vmax = 100 if vmax == None else vmax
            self.X[:,v] = np.random.uniform(vmin,vmax,(self.N))
            self.V[:,v] = np.random.uniform(0.1,vmax*0.1,(self.N))
            self.Xbest = self.X

        for i in range(self.N):
            self.fitness[i] = self.func(self.X[i], *self.args)
            self.best_fitness = self.fitness
            self.nfev+=1

        best_idx= np.argmin(self.best_fitness)
        self.best_global = self.Xbest[best_idx]
        self.best_global_fitness = self.best_fitness[best_idx]

    def solve(self):
        r = 0
        best_value = 99999999999
        best_individual = 99999999999
        while r<self.rep:
            self.init_particles()

            t = 0 #Time
            while t < self.iter:
                r1 = np.random.uniform(0,1,(self.N))
                r2 = np.random.uniform(0,1,(self.N))

                for i in range(self.N):
                    self.V[i] = self.hyper_parameters[0]*self.V[i] + r1[i]*self.hyper_parameters[1]*(self.Xbest[i]-self.X[i]) + r2[i]*self.hyper_parameters[2]*(self.best_global-self.X[i])
                    self.X[i] = self.X[i] + self.V[i]
                    self.fitness[i] = self.func(self.X[i], *self.args)
                    self.nfev+=1

                    if self.fitness[i] < self.best_fitness[i]:
                        self.best_fitness[i] = self.fitness[i]
                        self.Xbest[i] = self.X[i]

                best_idx= np.argmin(self.best_fitness)
                if self.best_fitness[best_idx] < self.best_global_fitness:
                    #print("el mejor->", best_idx, self.best_fitness, end="\n\n")
                    #print(self.Xbest[best_idx],end="\n\n")
                    self.best_global = self.Xbest[best_idx].copy()
                    self.best_global_fitness = self.best_fitness[best_idx].copy()
                t += 1
            r+=1
            print("rep: ",r, "/",self.rep)
        return self.best_global, self.iter, self.best_global_fitness, self.nfev

def particle_swarm_optimization(func, iterations, bounds, args = (), N = 2000, rep=1):
    ep  = PSO(func, iterations, bounds, args, N, rep)
    return ep.solve()