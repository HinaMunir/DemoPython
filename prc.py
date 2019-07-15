# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 20:38:59 2019

@author: Hina
"""

"""a='hello! world, hi'
print(a.split(','))
print(list(range(10)))

#Graph
class Graph:
    def __init__(self):
        self.gp={}
    def addNode(self,key):
        self.gp[key]=[]
    def addEdge(self,key,edge):
        self.gp[key].append(edge)
    def initializing(self):
        self.addNode('A')
        self.addNode('B')
        self.addNode('S')
        self.addNode('C')
        self.addNode('G')
        self.addNode('D')
        self.addNode('F')
        self.addNode('E')
        self.addNode('H')
        self.addEdge('A','B')
        self.addEdge('B','A')
        self.addEdge('A','S')
        self.addEdge('S','A')
        self.addEdge('S','C')
        self.addEdge('C','S')
        self.addEdge('S','G')
        self.addEdge('G','S')
        self.addEdge('C','D')
        self.addEdge('D','C')
        self.addEdge('C','E')
        self.addEdge('E','C')
        self.addEdge('C','F')
        self.addEdge('F','C')
        self.addEdge('G','F')
        self.addEdge('F','G')
        self.addEdge('G','H')
        self.addEdge('H','G')
        self.addEdge('E','H')
        self.addEdge('H','E')
    def print_gp(self):
        print(self.gp)
    def BFS(self,start,goal):
        frontier=[]
        action=[]
        exp=[]
        frontier.append(start)
        while((len(frontier))!=0):
            temp=frontier.pop()
            if temp not in exp:
                exp.append(temp)
            action=self.gp[temp]
            for i in range(0,len(action)):
                temp_=action[i]
                if temp_==goal:
                    exp.append(temp_)
                    print(exp)
                    return 'goal reached'
                if temp_ not in exp:
                    frontier.append(temp_)
                    exp.append(temp_)
        return 'failure'      
    def DFS(self,start,goal):
        frontier=[]
        action=[]
        exp=[]
        frontier.append(start)
        while((len(frontier))!=0):
            temp=frontier.pop()
            if temp not in exp:
                exp.append(temp)
            action=self.gp[temp]
            for i in range(0,len(action)):
                temp_=action[i]
                if temp_==goal:
                    exp.append(temp_)
                    print(exp)
                    return 'goal reached'
                if temp_ not in exp:
                    frontier.append(temp_)
                   
        return 'failure'           
        
bfs=Graph()
bfs.initializing()
bfs.print_gp()        
print(bfs.BFS('A','F'))
print(bfs.DFS('A','F')

#A*

class Graph:
    def __init__(self):
        self.gp={}
        self.h={'A':10,'B':8,'C':5,'D':7,'E':3,'F':6,'G':5,'H':3,'I':1,'J':0}   
    def addVertex(self,node):
        self.gp[node]=[]
    def addEdge(self,node,edge):
        self.gp[node].append(edge)
    def initializing(self):
        self.addVertex('A')
        self.addVertex('B')
        self.addVertex('C')
        self.addVertex('D')
        self.addVertex('E')
        self.addVertex('F')
        self.addVertex('G')
        self.addVertex('H')
        self.addVertex('I')
        self.addVertex('J')
        self.addEdge('A',[6,'B'])
        self.addEdge('A',[3,'F'])
        self.addEdge('B',[6,'A'])
        self.addEdge('B',[3,'C'])
        self.addEdge('B',[2,'D'])
        self.addEdge('C',[3,'B'])
        self.addEdge('C',[1,'D'])
        self.addEdge('C',[5,'E'])
        self.addEdge('D',[2,'B'])
        self.addEdge('D',[1,'C'])
        self.addEdge('D',[8,'E'])
        self.addEdge('E',[5,'C'])
        self.addEdge('E',[8,'D'])
        self.addEdge('E',[5,'I'])
        self.addEdge('E',[5,'J'])
        self.addEdge('F',[3,'A'])
        self.addEdge('F',[1,'G'])
        self.addEdge('F',[7,'H'])
        self.addEdge('G',[1,'F'])
        self.addEdge('G',[3,'I'])
        self.addEdge('H',[7,'F'])
        self.addEdge('H',[2,'I'])
        self.addEdge('I',[3,'G'])
        self.addEdge('I',[2,'H'])
        self.addEdge('I',[5,'E'])
        self.addEdge('I',[3,'J'])
        self.addEdge('J',[5,'E'])
        self.addEdge('J',[3,'I'])       
    def aSt(self,start,goal):
        front=[]
        exp=[]
        act=[]
        fn=[]
        front.append([start,self.h[start]])
        while len(front)>0:
            temp=front.pop()
            act=self.gp[temp[0]]
            for i in range(0,len(act)):
                v=act[i]
                h=self.h[v[1]]
                f=h+v[0]
                fn.append([f,v[1]])
            node=min(fn)
            exp.append(node)
            if node[1]!=goal:
                front.append([node[1],self.h[node[1]]])
        print(exp)        
            
        
ast=Graph()
ast.initializing()
ast.aSt('A','J')       """
"""
import random
pop=50
global t,g
class chrom:
    def __init__(self):
        self.chromo=[]
        self.f=0
        self.g=g
    def cg(self):
        return random.choice(self.g)
    def cc(self):
        l=len(t)
        for i in range(0,l):
            self.chromo.append(self.cg())
        return self.chromo
    def cf(self):
       
        for i in range(0,len(t)):
            if self.chromo[i]==t[i]:
               self.f+=1
        return self.f
        
t=[]
g=[]
t.extend('Hina Munir')
g.extend('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ')

popu=[]
for i in range(0,pop):
    c=chrom()
    popu.append([c.cc(),c.cf()])
print(popu)    
while popu[0][1]!=len(t):
    popu.sort(key=lambda ind:ind[1] , reverse=True)
    if popu[0][1]==len(t):
        break
    else:
        t1=int(pop*(0.1))
        new=[]
        t2=int(pop*(0.9))
        new.extend(popu[0:t1])
        for i in range(0,t2):
            parent1=random.choice(popu[t1:t2])
            print(parent1)
            parent2=random.choice(popu[t1:t2])
            print(parent2)
            child=[]
            ch=chrom()
            for p1,p2 in zip(parent1[0],parent2[0]):
                prob=random.random()
                if prob< 0.5:
                    child.append(p1)
                elif prob< 0.9:
                    child.append(p2)
                else:
                    child.append(ch.cg())
                  
            ch.chromo=child
            new.append([ch,ch.cf()])
    popu.extend(new)        
            
print(popu[0][0])            """
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 08:05:06 2019

@author: Hina
"""

import random
POPULATION_SIZE=50
global TARGET,GENES
class Chromosome:
    def __init__(self):
        self.chromosome=[]      
        self.GENES=GENES
        self.fitness=0
    def cal_fitness(self):    
        length=len(TARGET)
        for i in range(0,length):
            if self.chromosome[i]==TARGET[i]:                
                self.fitness=self.fitness+1            
        return self.fitness                
    def get_gene(self):
        return random.choice(self.GENES)
    def create_chromosome(self):
        length=len(TARGET)
        i=0
        for i in range(length):
            self.chromosome.append(self.get_gene())
        return self.chromosome    
   
        
    
GENES=[]
GENES.extend('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ')
TARGET=[]
TARGET.extend('I Love Artificial Intelligence')
population=[]
for i in range(POPULATION_SIZE):
    chromosome=Chromosome()
    population.append([chromosome.create_chromosome(),chromosome.cal_fitness()])
print('GENES:')    
print(GENES)
print('TARGET:')
print(TARGET)
print('Population:')
for index in range(0,len(population)):
    print(population[index]) 
while population[0][1]!= len(TARGET):    
    population.sort(key=lambda ind: ind[1], reverse=True )
    if population[0][1]== len(TARGET):
       break
    else:
        elite_ten_percent=int(POPULATION_SIZE*(0.1))
        new_generation=[]
        new_generation.extend(population[0:elite_ten_percent])
        elite_ninety_percent=int(POPULATION_SIZE*(0.9))
        
        for i in range(0,elite_ninety_percent):
            
            parent1=random.choice(population[elite_ten_percent:elite_ninety_percent])
            parent2=random.choice(population[elite_ten_percent:elite_ninety_percent])
            
            child=[]
            chromo=Chromosome()
            
            for p1,p2 in zip(parent1[0],parent2[0]):
                probability=random.random()
                if probability < 0.5:
                    child.append(p1)
                elif probability < 0.9:
                    child.append(p2)              
                else:
                    child.append(chromo.get_gene())
            
            chromo.chromosome=child   
            new_generation.append([chromo.chromosome,chromo.cal_fitness()])
                     
        population.extend(new_generation) 
        
print('Target Reached:\n',population[0][0])        