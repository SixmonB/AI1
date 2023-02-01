import math
import statistics
from matplotlib import pyplot as plt

def euclid(dataset, tests):
    # calculo de distancias euclidianas
    distance = 0
    for i in range(len(dataset)):
        distance += (tests[i]- dataset[i])**2
    distance = math.sqrt(distance)
    return distance

def get_distances(dataset, tests):
    # recopilacion de distancias euclidianas
    distances = []
    for j in range(len(dataset[0])):
        distances.append(euclid(dataset[:,j], tests))
    return distances

def knn(distances, fruits, k):
    # ordenamiento de vectores y obtencion del resultado correspondiente
    distances, fruits = (list(t) for t in zip(*sorted(zip(distances, fruits))))
    neighbours = distances[0:k]
    fruit_neighbours = fruits[0:k]

    result = statistics.mode(fruit_neighbours)
    print(fruit_neighbours)
    return neighbours, fruit_neighbours, result
   

def normalize(x, xmin, xmax):
    # proceso de eliminacion de unidades y dimension
    x = (x-xmin)/(xmax-xmin)
    return x

def plothist(dataset,tests):
    # ploteo del scatter 
    c=["yellow","green","orange","red"]
    cp=["yellow","green","orange","red"]
    plt.figure(5)
    j=-1
    for k in range(4):
        for i in range(7):
            j+=1
            plt.scatter(dataset[0,j],dataset[1,j],color=c[k])
    j=-1
    for k in range(4):
        for i in range(6):
            j+=1
            plt.scatter(tests[0,j],tests[1,j],color=cp[k],marker='^')
    plt.show()