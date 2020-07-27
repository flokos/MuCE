#------------------------------------------------------------------------------+
#
#   Nathan A. Rooy
#   Simple Particle Swarm Optimization (PSO) with Python
#   July, 2016
#   Updated by:
#   Theodore Flokos
#   January, 2018
#------------------------------------------------------------------------------+

#--- IMPORT DEPENDENCIES ------------------------------------------------------+

from __future__ import division
import random
import math
import numpy as np
import time
import pandas as pd
class Particle:
    def __init__(self,x0,w,c1,c2):
        self.position_i=[]          # particle position
        self.velocity_i=[]          # particle velocity
        self.pos_best_i=[]          # best position individual
        self.err_best_i=-1          # best error individual
        self.err_i=-1               # error individual
        self.w=w       # constant inertia weight (how much to weigh the previous velocity)
        self.c1=c1     # cognitive constant
        self.c2=c2     # social constant
        for i in range(0,num_dimensions):
            self.velocity_i.append(np.random.uniform(-1,1))
            self.position_i.append(x0[i])

    # evaluate current fitness
    def evaluate(self,costFunc):
        self.err_i=costFunc(self.position_i)
        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i==-1:
            self.pos_best_i=self.position_i
            self.err_best_i=self.err_i

    # update new particle velocity
    def update_velocity(self,pos_best_g):
        for i in range(0,num_dimensions):
            r1= np.random.random()
            r2= np.random.random()

            vel_cognitive=self.c1*r1*(self.pos_best_i[i]-self.position_i[i])
            vel_social=self.c2*r2*(pos_best_g[i]-self.position_i[i])
            self.velocity_i[i]=self.w*self.velocity_i[i]+vel_cognitive+vel_social

    # update the particle position based off new velocity updates
    def update_position(self,bounds):
        for i in range(0,num_dimensions):
            self.position_i[i]=self.position_i[i]+self.velocity_i[i]

            # adjust maximum position if necessary
            if self.position_i[i]>bounds[i][1]:
                self.position_i[i]=bounds[i][1]

            # adjust minimum position if neseccary
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i]=bounds[i][0]
                
class PSO():
    def __init__(self,costFunc=None,bounds=None,num_particles=None,w=None,c1=None,c2=None,maxiter=None,threshold=None):
        global num_dimensions
        
        if costFunc == None:
            raise Exception('Cost Function not given.')
        if bounds == None:
            raise Exception('Variable bounds not given.')
        if w == None:
            raise Exception('Inertia weight parameter w missing.')
        if c1 == None:
            raise Exception('Cognitive parameter c1 missing.')
        if c2 == None:
            raise Exception('Social parameter c2 missing.')
        if maxiter == None and threshold == None:
            raise Exception('You have to define stopping criteria.')
        
        num_dimensions=len(bounds)
        err_best_g=1000                   # best error for group
        pos_best_g=[]                   # best position for group
        # establish the swarm
        swarm=[]
        #Initialize num_particles particles with uniformely random starting positions
        initial_positions = []
        for var_bounds in bounds:
            initial_positions.append(np.random.uniform(var_bounds[0],var_bounds[1],num_particles))    
        for i in range(0,num_particles):
            rand_pos = []
            swarm.append(Particle([initial_positions[0][i],initial_positions[1][i]],w,c1,c2))

        # begin optimization loop
        i=0
        while i < maxiter and err_best_g > threshold:
            print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            particle_positions = [prtcl.position_i for prtcl in swarm]
            print particle_positions
            for j in range(0,num_particles):
                start = time.time()
                swarm[j].evaluate(costFunc)

                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g=list(swarm[j].position_i)
                    err_best_g=float(swarm[j].err_i)
                finnish = time.time()
               
            # cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
            a = {
                'Acc':[err_best_g],
                'C':[pos_best_g[0]],
                'Gamma':[pos_best_g[1]],
            }
            df = pd.DataFrame.from_dict(a)
            df.to_csv(path_or_buf='./pso.csv')
            i = i + 1

        # print final results
        print 'FINAL:'
        print pos_best_g
        print err_best_g

if __name__ == "__PSO__":
    main()

