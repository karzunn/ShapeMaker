from matplotlib import pyplot as plt
import numpy as np
import random
from PIL import Image

def positioner(x,y,Item,Space): #Function to position a shape at the given coordinates in a space.
    c, r = x, y
    Blank=Space.copy()
    Blank[r:r+Item.shape[0], c:c+Item.shape[1]]=Item
    return Blank

def stacker(positions,shapes,space): #Function to intake shapes and their positions, and stack them in a space.
    Blank=space.copy()
    for i in range(len(shapes)):
        Blank=Blank+positioner(positions[1][i],positions[0][i],shapes[i],space)
    return Blank

def fitness(stack): #Fitness evaluation method
    return np.sum(goal*(stack > 0).astype(int))/np.sum(stack*(goal == 0).astype(int))

def crossover(A,B):
    crossoverpt=round(random.uniform(0,len(A)-1))
    Child1=np.concatenate((A[:,0:crossoverpt],B[:,crossoverpt:]),axis=1)
    Child2=np.concatenate((B[:,0:crossoverpt],A[:,crossoverpt:]),axis=1)
    return Child1,Child2

def mutate(S):
    Selection=S.copy()
    x=random.randint(0,len(Selection[0])-1)
    y=random.randint(0,1)
    Selection[y][x]=random.randint(0,Space.shape[y]-Shapes[x].shape[y])
    return Selection

def breed(positions,shapes,space,Population,cp,pp,mp):
    
    Fitness=[None]*Population
    for y in range(Population):
        Fitness[y]=fitness(stacker(positions[y],shapes,space))
    
    rel_fit=np.array(Fitness)/np.sum(Fitness)
    
    pp=1/(1-pp)
    
    parents=[]
    while len(parents)<=round(Population/(4/cp))*2:
        r = random.uniform(min(rel_fit)+(max(rel_fit)-min(rel_fit))/pp,max(rel_fit))
        for i in range(Population):
            if r<=rel_fit[i]:
                parents.append(i)
                break
    nextgen=[]
    while len(nextgen)<round(Population-len(parents)*2):
        r = random.uniform(min(rel_fit)+(max(rel_fit)-min(rel_fit))/pp,max(rel_fit))
        for i in range(Population):
            if r<=rel_fit[i]:
                nextgen.append(i)
                break
    
    gen2=[positions[nextgen[i]] for i in range(len(nextgen))]

    pickedparents=[]
    children=[]
    for i in range(int(len(parents))):
        randnum1=round(random.uniform(0,(Population/(2/cp))-1))
        randnum2=round(random.uniform(0,(Population/(2/cp))-1))
        while randnum1==randnum2 and [randnum1,randnum2] in pickedparents:
            randnum2=round(random.uniform(0,(Population/(2/cp))-1))
        pickedparents.append([randnum1,randnum2])
        c1,c2=crossover(positions[parents[randnum1]],positions[parents[randnum2]])
        children.append(c1)
        children.append(c2)
        
    Finalists=children+gen2
    
    mutations=[]
    for x in range(round(Population*mp)):
        mutation=round(random.uniform(0,Population-1))
        while mutation in mutations:
            mutation=round(random.uniform(0,Population-1))
        mutations.append(mutation)
        Finalists[mutation]=mutate(Finalists[mutation])

    return Finalists





Population=500 #Population size is adjusted here

#Imports an image drawn using paint of a smiley face. The idea is to make a matrix that has 1's where there is a shape and 0's where there isn't.
goal = 1-np.sum(np.array(Image.open(r"C:\Users\cgo_1\Desktop\Python\Shapes\SmileyFace.PNG")),axis=2)/765
Space = np.zeros((goal.shape[0],goal.shape[1]))

#List of random shapes that the algorithm will use to best construct the goal shape
Shapes = np.array([np.ones((5,5)),np.ones((5,5)),np.ones((5,5)),np.ones((5,5)),np.ones((5,5)),np.ones((5,5)),np.ones((5,5)),np.ones((5,5)),np.ones((5,5)),np.ones((5,5)),np.ones((5,5)),np.ones((5,5))
                  ,np.ones((5,25)),np.ones((5,25)),np.ones((5,25)),np.ones((5,25)),np.ones((25,5)),np.ones((25,5)),np.ones((25,5)),np.ones((25,5)),
                  np.ones((5,5)),np.ones((5,5)),np.ones((5,5)),np.ones((5,5)),np.ones((5,5)),np.ones((5,5)),np.ones((5,5)),np.ones((5,5)),np.ones((5,5)),np.ones((5,5)),np.ones((5,5)),np.ones((5,5))])

positions=np.array([[[random.randint(0,(Space.shape[0])-(Shapes[y].shape[0])) for y in range(len(Shapes))],[random.randint(0,(Space.shape[1])-(Shapes[x].shape[1])) for x in range(len(Shapes))]] for p in range(Population)])

currentmax=0
improved=True
miniter=1
while improved==True:
    Children=breed(positions,Shapes,Space,Population,1,0,0.75) #A 100% crossover percentand a 0% mutation rate seem to work best.
    positions=Children.copy()
    FitnessFinal=[None]*Population
    for y in range(Population):
        FitnessFinal[y]=(fitness(stacker(positions[y],Shapes,Space)))
    if currentmax==max(FitnessFinal) and miniter>25:
        improved=False
        best=FitnessFinal.index(max(FitnessFinal))
    miniter=miniter+1
    currentmax=max(FitnessFinal)

plt.imshow(goal, interpolation='nearest')
plt.title('goal shape')
plt.show()
plt.imshow(stacker(positions[best],Shapes,Space), interpolation='nearest')
plt.title('algorithm recreation')
plt.show()
