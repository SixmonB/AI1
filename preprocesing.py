import cv2 as cv
import numpy as np
import math
import os

def prep(image):
    # se cambia el tamaño para que todas las imagenes sean iguales
    width = 700
    height = 500
    dim = (width, height)
    image = cv.resize(image, dim, interpolation = cv.INTER_AREA)
    	
    # se aplica estilizacion para difuminar colores y acentuar bordes
    image = cv.stylization(image, sigma_s=60, sigma_r=0.1)
    
    # se pasa la imagen a escala de grises
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    return gray_image, image

def segmentation(gray_image):
    # se aplica difuminacion gausiana
    blur = cv.GaussianBlur(gray_image,(15,15),0)
    
    # se aplica threshold binario
    ret, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    
    #dilatacion y erosion -> closing (se elimina huecos dentro de la imagen de la fruta)
    kernel = np.ones((7,7),np.uint8)
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    binary = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

    return thresh, binary

def database(training_images):
    dataset = np.zeros((1,len(training_images))) # 1 fila para la suma de los momentos hu y las columnas segun la cantidad de imagenes ingresada
    pics = []
    seg_pics = []
    for i in range(len(training_images)):
        # se procesa cada imagen
        gray_image, image = prep(training_images[i])         
        seg_image, binary = segmentation(gray_image)
        seg_pics.append(seg_image)
        
        # se busca los hu moments
        moments = cv.HuMoments(cv.moments(seg_image)).flatten()
        
        # se obtiene los contornos
        _, contours, _ = cv.findContours(binary,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        c = sorted(contours, key = cv.contourArea)[-1]
        c_mask=cv.drawContours(binary, c,-1, (0,255,0), 10)
        
        # se saca el fondo de la imagen
        new_img = cv.bitwise_and(image, image, mask=c_mask)
        x, y, w, h = cv.boundingRect(c)
        dst = new_img[y: y + h, x: x + w]

        dst_gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
        _, alpha = cv.threshold(dst_gray, 0, 255, cv.THRESH_BINARY)
        b, g, r = cv.split(dst)

        rgba = [r, g, b, alpha]
        dst = cv.merge(rgba, 4)
        pics.append(dst)
    
        # se calcula el area
        area = cv.contourArea(c)
        
        # se calcula el perimetro
        perimeter = cv.arcLength(c,True)
        
        # se calcula la redondez 
        redondez = 4*math.pi*(area/perimeter**2)
        
        # se junta en un vector todos los descriptores
        # descriptores = [area, perimetro, redondez, momentos hu] 

        moments = moments[0] + moments[2]
        
        dataset[:,i] = moments
        # se retorna base de datos y vectores de imagenes a color y segmentadas.
    return dataset, pics, seg_pics

def get_histograma(img):
    promhist=[]
    # se calcula el promedio de la suma del histograma de cada imagen
    for j in range(len(img)):
        promhistj=0
        color = ('b','g','r')
        for i, col in enumerate(color):
            hist=cv.calcHist([img[j]],[i], None, [256], [10, 250])
            promhistj+=(np.mean(hist))
            
        promhist.append(promhistj/3)

    return promhist

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images