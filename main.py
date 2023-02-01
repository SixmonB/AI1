from matplotlib import pyplot as plt
import preprocesing
import knn
import os
import statistics
import numpy as np
import kmeans

if __name__ == '__main__':
    
    # se importa imagenes de entrenamiento y obtenemos matrices de parametros
    training_images = preprocesing.load_images_from_folder("D:\sbsim\Documents\Facultad\Inteligencia Artificial 1\Proyecto FInal\Training images")
    dataset, pics, seg_pics = preprocesing.database(training_images)
    promhist = preprocesing.get_histograma(pics)
    dataset = np.vstack((dataset , promhist))
    
    # se plotea imagenes obtenidas
    fig, axes = plt.subplots(int(len(dataset[0])/2),2, figsize =(16,9))
    ax = axes.ravel()
    for i in range(len(dataset[0])):
        ax[i].imshow(pics[i], cmap=plt.cm.binary)
    
    fig, axes = plt.subplots(int(len(dataset[0])/2),2, figsize =(16,9))
    ax = axes.ravel()
    for i in range(len(dataset[0])):
        ax[i].imshow(seg_pics[i], cmap = plt.cm.binary)
        
    # se importa imagenes de prueba y obtenemos matrices de parametros
    test_images = preprocesing.load_images_from_folder("D:\sbsim\Documents\Facultad\Inteligencia Artificial 1\Proyecto FInal\Test images")
    tests, pruebas, seg_pruebas = preprocesing.database(test_images)
    promhist2 = preprocesing.get_histograma(pruebas)
    tests = np.vstack((tests, promhist2))
    fig, axes = plt.subplots(int(len(tests[0])/2),2, figsize =(16,9))
    ax = axes.ravel()
    
    # se plotea imagenes obtenidas
    for i in range(len(tests[0])):
        ax[i].imshow(pruebas[i], cmap=plt.cm.binary)
        
    fig, axes = plt.subplots(int(len(tests[0])/2),2, figsize =(16,9))
    ax = axes.ravel()
    for i in range(len(tests[0])):
        ax[i].imshow(seg_pruebas[i], cmap = plt.cm.binary)
    
    # se crea vectores de frutas para las imagenes de entrenamiento y prueba respectivamente
    fruits = ['banana']*7
    fruits.extend(['limon']*7)
    fruits.extend(['naranja']*7)
    fruits.extend(['tomate']*7)
    
    tested_fruits = ['banana']*6
    tested_fruits.extend(['limon']*6)
    tested_fruits.extend(['naranja']*6)
    tested_fruits.extend(['tomate']*6)
    
    # se adimensionaliza cada matriz
    for i in range(len(tests)):
        max_test = max(tests[i])
        min_test = min(tests[i])
        max_dataset = max(dataset[i])
        min_dataset = min(dataset[i])
        for j in range(len(tests[0])):
            tests[i,j] = knn.normalize(tests[i,j], min_test, max_test)
        for j in range(len(dataset[0])):
            dataset[i,j] = knn.normalize(dataset[i,j], min_dataset, max_dataset)
            
    
    # se plotea el scatter de datos
    knn.plothist(dataset, tests)
    
    precision_knn = 0
    precision_kmeans = 0
    
    fruits_kmeans = ['banana', 'limon', 'naranja', 'tomate']
    
    # se recorre la matriz de prueba para obtener el resultado de cada imagen
    for i in range(len(tests[0])):
        distances = knn.get_distances(dataset,tests[:,i])
        neighbours, fruit_neighbours, result_knn = knn.knn(distances, fruits,3)
        print('La fruta segun el knn es ' + result_knn)
        print('La fruta analizada es ' + tested_fruits[i])
        print('---------------------------------------')
        if(result_knn == tested_fruits[i]):
            precision_knn +=1
            
        centroids = kmeans.centroides(dataset)
        result_kmeans, distance_kmeans = kmeans.kmeans(tests[:,i], centroids, fruits_kmeans)
        print('La fruta segun el kmeans es ' + result_kmeans ) 
        print('La fruta analizada es ' + tested_fruits[i])
        print('---------------------------------------')
        if(result_kmeans == tested_fruits[i]):
            precision_kmeans +=1
        
    # print final de efectividad
    print('La precision del knn es de: ' + str(precision_knn/len(tested_fruits)*100))
    print('La precision del kmeans es de: ' + str(precision_kmeans/len(tested_fruits)*100))
    #print(distances)