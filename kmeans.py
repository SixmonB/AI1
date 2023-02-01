import math
import numpy as np

def centroides(dataset):
    centroids = np.zeros((2,4))
    # asignacion de centroides original
    centroids[0] = [1, 0.2, 0, 0.3]   # los componentes de centroids estan guardados
    centroids[1] = [0, 0.2, 0.8, 1]   # como variables float. Hay que convertirlos a int despues.

    # coordenadas de los centroides segun las columnas -> banana, limon, naranja, tomate 

    k = 4   # 4 clusters
    max_iteraciones = 300
    it = 0
    
    # comienzan las iteraciones 
    
    while it <= max_iteraciones:
        cluster0 = np.zeros((2,1))
        cluster1 = np.zeros((2,1))
        cluster2 = np.zeros((2,1))
        cluster3 = np.zeros((2,1))
        for i in range(len(dataset[0])):
            indice = []
            distance = []
            for j in range(k):
                distance.append(math.sqrt( (dataset[0,i] - centroids[0,j]) **2 + (dataset[1,i] - centroids[1,j]) **2 ) )
                indice.append(j)
                
            # se ordena el vector de distancias para obtener el indice correspondiente y saber que cluster corresponde
            distance, indice = (list(t) for t in zip(*sorted(zip(distance, indice))))
            
            
            # se realiza una reasignacion para mantener la constancia de los tipos de datos
            aux = np.zeros((2,1))
            aux[0,0] = dataset[0,i]
            aux[1,0] = dataset[1,i]
            
            if(indice[0]==0):
                cluster0 = np.hstack((cluster0, aux))
            elif(indice[0]==1):
                cluster1 = np.hstack((cluster1, aux))
            elif(indice[0]==2):
                cluster2 = np.hstack((cluster2, aux))
            elif(indice[0]==3):
                cluster3 = np.hstack((cluster3, aux))
        
        
        # se calcula promedio dentro de los clusters para encontrar el nuevo centroide
        
        suma = np.zeros((2,4))
        for i in range(len(cluster0[0])):
            suma[0,0] += cluster0[0,i]
            suma[1,0] += cluster0[1,i]
        for i in range(len(cluster1[0])):
            suma[0,1] += cluster1[0,i]
            suma[1,1] += cluster1[1,i]
        for i in range(len(cluster2[0])):
            suma[0,2] += cluster2[0,i]
            suma[1,2] += cluster2[1,i]
        for i in range(len(cluster3[0])):
            suma[0,3] += cluster3[0,i]
            suma[1,3] += cluster3[1,i]
               
            
        centroids[:,0] = suma[:,0]/(len(cluster0[0]))
        centroids[:,1] = suma[:,1]/(len(cluster1[0]))
        centroids[:,2] = suma[:,2]/(len(cluster2[0]))
        centroids[:,3] = suma[:,3]/(len(cluster3[0]))
        it += 1
    return centroids
    
def kmeans(tests, centroids, fruits):
    # con los centroides calculados, se compara la imagen de entrada y se obtiene el mas cercano
    k = 4
    distance = []
    aux = np.zeros((2,1))
    aux[0,0] = tests[0]
    aux[1,0] = tests[1]
    for j in range(k):
        distance.append(math.sqrt( (aux[0,0] - centroids[0,j])**2 + (aux[1,0] - centroids[1,j])**2 ) )
    distance, fruits = (list(t) for t in zip(*sorted(zip(distance, fruits))))
    print(fruits)
    result = fruits[0]
    return result, distance    