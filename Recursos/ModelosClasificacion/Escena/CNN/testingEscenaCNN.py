import sys, glob, argparse
import numpy as np
import math, cv2
from scipy.stats import multivariate_normal
import time
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
import matplotlib.patches as patches

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-tde' , "--tdes", help="Directory of txt with description" , default='E:/DocumentosOSX/TrabajoGrado/TecnicasBarrido/Exhaustivo/BarridoVentana/DescripcionBarridoExhaustivo.txt')
    parser.add_argument('-dis' , "--dirscores", help="Directory of txt with scores" , default='E:/DocumentosOSX/TrabajoGrado/TecnicasBarrido/Exhaustivo/BarridoVentana/DescripcionBarridoExhaustivo.txt')
    parser.add_argument('-dip' , "--dirpred", help="Directory of txt with predictions" , default='E:/DocumentosOSX/TrabajoGrado/TecnicasBarrido/Exhaustivo/BarridoVentana/DescripcionBarridoExhaustivo.txt')
    parser.add_argument('-gra' , "--visualize"  , help="Mostrar resultados graficos 0(No) - 1(Si)"          , default=0, type=int)
    parser.add_argument('-ther' , "--thereshold"  , help="Thereshold para NMS"          , default=0.35, type=float)
    parser.add_argument('-fr' , "--frame"  , help="Directorio del frame"          , default='E:/DocumentosOSX/TrabajoGrado/TecnicasBarrido/Exhaustivo/BarridoVentana/Factor1.JPG')
    parser.add_argument('-gr' , "--ground"  , help="Directorio del txt con ground truth"          , default='E:\DocumentosOSX\TrabajoGrado/coord.txt')
    parser.add_argument('-dif' , "--diferencia"  , help="Thereshold para diferencia entre BBs"          , default=0.2, type=float)
    parser.add_argument('-an' , "--ancho"  , help="Ancho imagen"          , default=0, type=int)
    parser.add_argument('-al' , "--alto"  , help="Alto imagen"          , default=0, type=int)
    parser.add_argument('-cP' , "--critPredic"  , help="Criterio para maximo score de predicción"          , default=0.65, type=float)
    parser.add_argument('-cC' , "--critConteo"  , help="Conteo minimo entre vecinos"          , default=10, type=int)
    parser.add_argument('-eS' , "--escala"  , help="Criterio para maximo score de prediccion"          , default=1.7, type=float)

    args = parser.parse_args()
    return args
def leerDescripcion(directorio):
    dimensiones = []
    archivo  = open(directorio, 'r')
    caracter = archivo.readlines();
    caracter = [x.strip() for x in caracter]
    ancho = caracter[0]
    alto = caracter[1]
    caracter = caracter[3:]

    escalas = []
    coordenadas = []
    coordenadas2 = []
    inicio = 0
    for i in range(0,len(caracter)):
        if caracter[i] == '*':
            escalas.append(caracter[inicio:i])
            inicio = i + 1

    # Construccion de objeto coordenadas
    for i in range(0,len(escalas)):
        esc = np.asfarray(escalas[i])
        esc = np.reshape(esc,(-1,2))
        coordenadas.append(esc)

    # Construccion de objeto coordenadas2
    for esc in escalas:
        for elemento in esc:
            coordenadas2.append(elemento)
    coordenadas2 = np.asfarray(coordenadas2)
    coordenadas2 = np.reshape(coordenadas2,(-1,2))

    return ancho,alto,coordenadas,coordenadas2
def NMS(ancho,alto,coordenadas,coordenadas2,scores,preds,th,visualize,frame):

    print("Ejecutando algoritmo de NMS...")
    ancho = float(ancho)
    alto = float(alto)
    boxes = []
    boxesReales = []
    pick = []
    overlapThresh = th

    # Algoritmo para dibujar las personas encontradas sin NMS
    # Aqui se incluye el cambio de escala segun el barrido

    for i in range (0,len(coordenadas)):

        factor = math.pow(args.escala,i)
        #print ("Calculando boxes de escala",i,"con factor",factor)
        coresc = coordenadas[i]

        for j in range(0,len(coresc)):
            y1 = float(coresc[j,0]) * factor
            x1 = float(coresc[j,1]) * factor
            #print x1,y1,ancho*factor,alto*factor,i
            boxes.append(x1)
            boxes.append(y1)
            boxes.append(ancho * factor)
            boxes.append(alto * factor)
            boxes.append(i)

    boxes = np.asfarray(boxes)
    boxes = np.reshape(boxes,(-1,5))
    #print boxes
    # Boxes: X,Y,ANCHO,ALTO,ESCALA A LA QUE PERTENECE (PARA CAMBIAR COLOR)


    img = cv2.imread(frame,0)
    fig,ax = plt.subplots(1)
    plt.title("Resultados de las predicciones sin NMS")
    plt.axis('off')
    ax.imshow(img, cmap = 'gray')
    for i in range(0,len(boxes)):
        scor = scores[i]
        personaScore = scor

        if preds[i] == 1 and personaScore > args.critPredic:

            if boxes[i,4] ==0:
                color = 'r'
            else:
                color = 'b'

            rect = patches.Rectangle((boxes[i,0],boxes[i,1]),boxes[i,2],boxes[i,3],linewidth=1,edgecolor=color,facecolor='none')
            ax.add_patch(rect)


            personaX1 = boxes[i,0]
            personaY1 = boxes[i,1]
            personaX2 = personaX1 + boxes[i,2]
            personaY2 = personaY1 + boxes[i,3]

            scor = scores[i]
            personaScore = scor

            # Construccion de BoxesReales
            boxesReales.append(personaX1)
            boxesReales.append(personaY1)
            boxesReales.append(personaX2)
            boxesReales.append(personaY2)
            boxesReales.append(personaScore)
            boxesReales.append(boxes[i,2])
            boxesReales.append(boxes[i,3])

            # boxesReales = X1,Y1,X2,Y2,SCORE(MAX),ANCHO,ALTO

    boxesReales = np.asfarray(boxesReales)
    boxesReales = np.reshape(boxesReales,(-1,7))
    boxesIniciales = len(boxesReales)

    # Inicio de algoritmo de NMS
    x1 = boxesReales[:,0]
    y1 = boxesReales[:,1]
    x2 = boxesReales[:,2]
    y2 = boxesReales[:,3]
    sc = boxesReales[:,4]

    # Algoritmo de limpieza 1
    critconteo = args.critConteo
    # Ordeno ventanas de menor a mayor
    idxs = np.argsort(boxesReales[:,4])
    # Invierto las ventanas para que qude de mayor a menor
    idxs = idxs[::-1]


    pick0 = []
    suppress = []
    for p in range(0,len(idxs)):
        i = idxs[p]
        conta = 0

        for pos in range(0, len(idxs)):

            # Itero sobre todos los BB que hay
            j = idxs[pos]
            #print ("Analizando traslape de imagen",i,"con",j)
            # Encuentro coordenadas para el traslape
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
			# Encuentro area de traslape
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            # Computo la relacion de area de traslape con area real
            overlap = float(w * h)
            #print ("Overlap:",overlap)
			# Si se sobrepasa el area de traslape, borro esa caja
            if overlap > 0:
                #print("Se suprimio la imagen",i)
                conta = conta + 1
        #print "La caja",p,"de coordenadas",x1[i],y1[i],x2[i],y2[i],"se traslapa",conta
        if conta < critconteo:
            suppress.append(i)

    idxs = list(idxs)
    #print suppress
    #print idxs
    for i in suppress:
        for j in idxs:
            if i == j:
                idxs.remove(i)
    #print suppress
    #print idxs

    boxesReales = boxesReales[idxs]
    print("Algoritmo de limpieza 1")
    print ("Boxes Iniciales:",boxesIniciales,"Boxes Finales:",len(boxesReales))

    fig2,ax2 = plt.subplots(1)
    ax2.imshow(img, cmap = 'gray')
    plt.title("Resultados de las predicciones con NMS 1")
    plt.axis('off')
    for i in range(0,len(boxesReales)):
        rect = patches.Rectangle((boxesReales[i,0],boxesReales[i,1]),boxesReales[i,5],boxesReales[i,6],linewidth=1,edgecolor='r',facecolor='none')
        ax2.add_patch(rect)

    # Algoritmo de limpieza 2
    x1 = boxesReales[:,0]
    y1 = boxesReales[:,1]
    x2 = boxesReales[:,2]
    y2 = boxesReales[:,3]
    sc = boxesReales[:,4]
    boxesIniciales = len(boxesReales)
    idxs = np.argsort(boxesReales[:,4])
    area = (boxesReales[:,2] - boxesReales[:,0] + 1) * (boxesReales[:,3] - boxesReales[:,1] + 1)
    #print ("Indices a analizar:",idxs)

    while len(idxs) > 0:
        # Recorro BB de abajo hacia arriba (mayor a menor score)
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]

        for pos in range(0, last):

            # Itero sobre todos los BB que hay
            j = idxs[pos]
            #print ("Analizando traslape de imagen",i,"con",j)
            # Encuentro coordenadas para el traslape
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
			# Encuentro area de traslape
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            # Computo la relacion de area de traslape con area real
            overlap = float(w * h) / area[j]
            #print ("Overlap:",overlap)
			# Si se sobrepasa el area de traslape, borro esa caja
            if overlap > overlapThresh:
                #print("Se suprimio la imagen",i)
                scor1 = sc[i]
                scor2 = sc[j]
                if abs(scor1-scor2) >= args.diferencia:
                    suppress.append(pos)

        # Borro indices de boxes que ya no estan
        #print ("Supress:",suppress)
        #print ("IDSX:",idxs)
        idxs = np.delete(idxs, suppress)
        #print ("IDSX2:",idxs)
        #print ("PICK",pick)

    boxesFinales = len(boxesReales[pick])
    print ("Boxes Iniciales:",boxesIniciales,"Boxes Finales:",boxesFinales)
    boxesReales = boxesReales[pick]
    #print boxesReales

    fig2,ax2 = plt.subplots(1)
    ax2.imshow(img, cmap = 'gray')
    plt.title("Resultados de las predicciones con NMS 2")
    plt.axis('off')

    for i in range(0,len(boxesReales)):
        rect = patches.Rectangle((boxesReales[i,0],boxesReales[i,1]),boxesReales[i,5],boxesReales[i,6],linewidth=1,edgecolor='r',facecolor='none')
        ax2.add_patch(rect)

    return boxesReales
def validacion(boxesReales,ground):

    verPositivos = []
    falPositivos = []
    falNegativos = []

    # Lectura del txt con el Ground Truth
    c = open(ground,'r')

    coordenadas = c.readlines()
    coordenadas = np.asfarray(coordenadas)
    coordenadas = np.reshape(coordenadas,(-1,4))

    # Informacion del Ground Truth
    x1t = coordenadas[:,0]
    y1t = coordenadas[:,1]
    anchost = coordenadas[:,2]
    altost = coordenadas[:,3]
    x2t = x1t + anchost
    y2t = y1t + altost

    # Informacion de los BB detectados
    x1e = boxesReales[:,0]
    y1e = boxesReales[:,1]
    anchose = boxesReales[:,5]
    altose = boxesReales[:,6]
    x2e = boxesReales[:,2]
    y2e = boxesReales[:,3]

    # Re-acomodacion de boxesReales
    auxiliar = []

    for i in range(0,len(boxesReales)):
        auxiliar.append(boxesReales[i,0])
        auxiliar.append(boxesReales[i,1])
        auxiliar.append(boxesReales[i,5])
        auxiliar.append(boxesReales[i,6])
    boxesReales = np.asfarray(auxiliar)
    boxesReales = np.reshape(boxesReales,(-1,4))

    # Calculo centros de los BB teoricos
    centrosXt = (x1t + (x1t + anchost))/2
    centrosYt = (y1t + (y1t + altost))/2

    # Calculo centros de los BB experimentales
    centrosXe = (x1e + (x1e + anchose))/2
    centrosYe = (y1e + (y1e + altose))/2

    # Caso 1: No se predijo ninguna persona cuando en realidad si hay
    # verPositivos = 0, falPositivos = 0, falNegativos = por lo menos len(teoricos)

    if len(boxesReales)==0 and len(coordenadas)!=0:
        print ("\n\nCASO 1: NO SE PREDICEN PERSONAS CUANDO EN REALIDAD HAY")
        verPositivos = []
        falPositivos = []
        razones = []
        falNegativos = len(coordenadas)
        return verPositivos,falPositivos,falNegativos,coordenadas,razones

    # Caso 2: Se predicen personas cuando en realidad no hay
    # verPositivos = 0,falPositivos = boxesReales, falNegativos=len(coordenadas)
    if len(boxesReales)!=0 and len(coordenadas)==0:
        print ("\n\nCASO 2: SE PREDICEN PERSONAS CUANDO EN REALIDAD NO HAY")
        verPositivos = []
        falPositivos = boxesReales
        razones = []
        falNegativos = 0
        return verPositivos,falPositivos,falNegativos,coordenadas,razones

    # Caso 3: Se predicen mas personas de las que realmente hay
    if len(coordenadas) < len(boxesReales):
        print ("\n\nCASO 3: MAS PERSONAS DE LAS QUE REALMENTE HAY")
        indices = []
        razones = []
        yaEvaluados = []
        distancias = []

        # Algoritmo para determinar pares de BB mas cercanos y que se traslapen:
        for i in range(0,len(coordenadas)):
            indice = -1
            indice2 = -1
            razon = -1
            areat = -1
            # Recorro los BB teoricos
            Cxt = centrosXt[i]
            Cyt = centrosYt[i]
            dif = 10000000
            for j in range(0,len(boxesReales)):
                # Recorro los BB experimentales
                Cxe = centrosXe[j]
                Cye = centrosYe[j]
                if (math.sqrt(math.pow(abs(Cxt - Cxe),2) + math.pow(abs(Cyt - Cye),2)) <= dif):
                    # Encontrar el area de traslape
                    areat = (x2t[i]-x1t[i])*(y2t[i]-y1t[i])
                    areae = (x2e[j]-x1e[j])*(y2e[j]-y1e[j])
                    xx1 = max(x1t[i], x1e[j])
                    yy1 = max(y1t[i], y1e[j])
                    xx2 = min(x2t[i], x2e[j])
                    yy2 = min(y2t[i], y2e[j])
        			# Encuentro area de traslape
                    w = max(0, xx2 - xx1 + 1)
                    h = max(0, yy2 - yy1 + 1)
                    # Tienen que traslaparse para que sea valida la deteccion
                    if w*h != 0:
                        dif = math.sqrt(math.pow(abs(Cxt - Cxe),2) + math.pow(abs(Cyt - Cye),2))
                        razon =float(w*h)/(float(areat)+float(areae)-float(w*h))
                        indice = j
                        indice2 = i
                        d = dif

            if indice != -1:
                indices.append(indice) #experimental
                razones.append(razon)
                yaEvaluados.append(indice2) #ground truth
                distancias.append(d)

        # Algoritmo para descartar multiples traslapes en un mismo punto del GT
        nuevoEvaluados = []
        nuevoIndices = []
        nuevoDistancias = []
        nuevoRazones = []

        print("Indices predicciones",indices)
        print("Indices GT",yaEvaluados)
        print("Distancias",distancias)
        print("Razones",razones)

        for i in range(0,len(indices)):
            experimental = indices[i]
            if not experimental in nuevoEvaluados:
                nuevoEvaluados.append(experimental)
                nuevoIndices.append(yaEvaluados[i])
                nuevoDistancias.append(distancias[i])
                nuevoRazones.append(razones[i])
            else:
                # Buscar la escena
                for j in range(0,len(nuevoEvaluados)):
                    # Si la encuentra
                    if nuevoEvaluados[j] == experimental:
                        if nuevoDistancias[j] > distancias[i]:
                             nuevoEvaluados[j] = experimental
                             nuevoIndices[j] = indices[i]
                             nuevoDistancias[j] = distancias[i]
                             nuevoRazones[j]=razones[i]

        indices = nuevoEvaluados
        yaEvaluados = nuevoIndices
        distancias = nuevoDistancias
        razones = nuevoRazones

        print("Nuevos Indices predicciones",indices)
        print("Nuevos Indices GT",yaEvaluados)
        print("Nuevas Distancias",distancias)
        print("Nuevas Razones",razones)

        # Algoritmo para limpíar boxes
        auxiliar1 = []
        auxiliar2 = []
        auxiliar3 = []
        for i in range(0,len(razones)):
            if razones[i] > 0.3:
                auxiliar1.append(indices[i])
                auxiliar2.append(razones[i])
                auxiliar3.append(distancias[i])

        indices = auxiliar1
        razones = auxiliar2
        distancias = auxiliar3

        print("Nuevos Indices predicciones",indices)
        print("Nuevas Distancias",distancias)
        print("Nuevas Razones",razones)

        # Algoritmo que determina indices de los fallidos
        indices2 = []
        for i in range(0,len(boxesReales)):
            if not i in indices:
                indices2.append(i)

        #print ("indices",indices)
        #print ("boxes reales",boxesReales,boxesReales.shape)
        boxesAcertadas = boxesReales[indices]
        #print ("boxes acertadas",boxesAcertadas)
        boxesFallidas = boxesReales[indices2]
        #print ("boxes fallidas", boxesFallidas)
        verPositivos = boxesAcertadas
        falPositivos = boxesFallidas
        falNegativos = len(coordenadas) - len(verPositivos)
        return verPositivos,falPositivos,falNegativos,coordenadas,razones

    # Caso 4: Se predicen menos personas de las que realmente hay
    if len(coordenadas) > len(boxesReales):
        print ("\n\nCASO 4: MENOS PERSONAS DE LAS QUE REALMENTE HAY")
        indices = []
        razones = []
        yaEvaluados = []
        distancias = []
        # Algoritmo para determinar pares de BB mas cercanos y que se traslapen:
        for i in range(0,len(boxesReales)):
            indice = -1
            indice2 = -1
            razon = -1
            areat = -1
            # Recorro los BB experimentales
            Cxe = centrosXe[i]
            Cye = centrosYe[i]
            dif = 10000000
            for j in range(0,len(coordenadas)):
                # Recorro los BB teoricos
                Cxt = centrosXt[j]
                Cyt = centrosYt[j]
                #print "experimental",i,"teorica",j,"dife",math.sqrt(math.pow(abs(Cxt - Cxe),2) + math.pow(abs(Cyt - Cye),2)),"difac",dif
                if (math.sqrt(math.pow(abs(Cxt - Cxe),2) + math.pow(abs(Cyt - Cye),2)) <= dif):
                    # Encontrar el area de traslape
                    areat = (x2t[j]-x1t[j])*(y2t[j]-y1t[j])
                    areae = (x2e[i]-x1e[i])*(y2e[i]-y1e[i])
                    xx1 = max(x1t[j], x1e[i])
                    yy1 = max(y1t[j], y1e[i])
                    xx2 = min(x2t[j], x2e[i])
                    yy2 = min(y2t[j], y2e[i])
        			# Encuentro area de traslape
                    w = max(0, xx2 - xx1 + 1)
                    h = max(0, yy2 - yy1 + 1)
                    # Tienen que traslaparse para que sea valida la deteccion
                    if w*h != 0:
                        dif = math.sqrt(math.pow(abs(Cxt - Cxe),2) + math.pow(abs(Cyt - Cye),2))
                        indice2 = j
                        indice = i
                        d = dif
                        razon =float(w*h)/(float(areat)+float(areae)-float(w*h))
            if indice!= -1:
                indices.append(indice)
                yaEvaluados.append(indice2)
                distancias.append(d)
                razones.append(razon)


        # Algoritmo para descartar multiples traslapes en un mismo punto del GT
        nuevoEvaluados = []
        nuevoIndices = []
        nuevoDistancias = []
        nuevoRazones = []

        for i in range(0,len(indices)):
            teorico = yaEvaluados[i]
            if not teorico in nuevoEvaluados:
                nuevoEvaluados.append(teorico)
                nuevoIndices.append(indices[i])
                nuevoDistancias.append(distancias[i])
                nuevoRazones.append(razones[i])
            else:
                # Buscar la escena
                for j in range(0,len(nuevoEvaluados)):
                    # Si la encuentra
                    if nuevoEvaluados[j] == teorico:
                        if nuevoDistancias[j] > distancias[i]:
                             nuevoEvaluados[j] = teorico
                             nuevoIndices[j] = indices[i]
                             nuevoDistancias[j] = distancias[i]
                             nuevoRazones[j]=razones[i]

        indices = nuevoIndices
        yaEvaluados = nuevoEvaluados
        distancias = nuevoDistancias
        razones = nuevoRazones

        #print indices
        #print yaEvaluados
        #print distancias

        # Algoritmo para limpíar boxes
        auxiliar1 = []
        auxiliar2 = []
        auxiliar3 = []
        for i in range(0,len(razones)):
            if razones[i] > 0.3:
                auxiliar1.append(indices[i])
                auxiliar2.append(razones[i])
                auxiliar3.append(distancias[i])

        indices = auxiliar1
        razones = auxiliar2
        distancias = auxiliar3

        print("Nuevos Indices predicciones",indices)
        print("Nuevas Distancias",distancias)
        print("Nuevas Razones",razones)


        # Algoritmo que determina indices de los fallidos
        indices2 = []
        for i in range(0,len(boxesReales)):
            if not i in indices:
                indices2.append(i)

        #print ("indices",indices)
        #print ("boxes reales",boxesReales,boxesReales.shape)
        boxesAcertadas = boxesReales[indices]
        #print ("boxes acertadas",boxesAcertadas)
        boxesFallidas = boxesReales[indices2]
        #print ("boxes fallidas", boxesFallidas)
        verPositivos = boxesAcertadas
        falPositivos = boxesFallidas
        falNegativos = len(coordenadas) - len(verPositivos)
        return verPositivos,falPositivos,falNegativos,coordenadas,razones
def escribirReporte(therNMS,therDIF,VP,FP,FN,P,R,F1,folder,bandera):
    if bandera == 0:
        print ("Se escribió el reporte de esta escena")
        archivo = open(folder,'a')
        linea = str(therNMS) + ";" + str(therDIF) + ";" + str(VP) + ";" + str(FP) + ";" + str(FN) + ";" + str(P) + ";" + str(R) + ";" + str(F1) + "\n"
        archivo.write(linea)
def validacion2(verPositivos,falPositivos,falNegativos,coordenadas,razones,boxesReales):

    falPositivos = list(falPositivos)

    #Barrido
    nverPositivos1 = []
    nverPositivos2 = []
    nverPositivos3 = []
    nverPositivos4 = []
    nverPositivos5 = []
    nverPositivos6 = []
    nverPositivos7 = []
    nverPositivos8 = []
    nverPositivos9 = []
    nverPositivos10 = []
    nfalPositivos1 = len(falPositivos)
    nfalPositivos2 = len(falPositivos)
    nfalPositivos3 = len(falPositivos)
    nfalPositivos4 = len(falPositivos)
    nfalPositivos5 = len(falPositivos)
    nfalPositivos6 = len(falPositivos)
    nfalPositivos7 = len(falPositivos)
    nfalPositivos8 = len(falPositivos)
    nfalPositivos9 = len(falPositivos)
    nfalPositivos10 = len(falPositivos)
    nfalNegativos1 = falNegativos
    nfalNegativos2 = falNegativos
    nfalNegativos3 = falNegativos
    nfalNegativos4 = falNegativos
    nfalNegativos5 = falNegativos
    nfalNegativos6 = falNegativos
    nfalNegativos7 = falNegativos
    nfalNegativos8 = falNegativos
    nfalNegativos9 = falNegativos
    nfalNegativos10 = falNegativos



    for i in range(0,len(verPositivos)):
        if razones[i]>0.3:
            nverPositivos1.append(verPositivos[i])
        else:
            nfalPositivos1 = nfalPositivos1 + 1
            nfalNegativos1 = nfalNegativos1 + 1
        if razones[i]>0.35:
            nverPositivos2.append(verPositivos[i])
        else:
            nfalPositivos1 = nfalPositivos2 + 1
            nfalNegativos2 = nfalNegativos2 + 1
        if razones[i]>0.4:
            nverPositivos3.append(verPositivos[i])
        else:
            nfalPositivos1 = nfalPositivos3 + 1
            nfalNegativos3 = nfalNegativos3 + 1
        if razones[i]>0.45:
            nverPositivos4.append(verPositivos[i])
        else:
            nfalPositivos1 = nfalPositivos4 + 1
            nfalNegativos4 = nfalNegativos4 + 1
        if razones[i]>0.5:
            nverPositivos5.append(verPositivos[i])
        else:
            nfalPositivos1 = nfalPositivos5 + 1
            nfalNegativos5 = nfalNegativos5+ 1
        if razones[i]>0.55:
            nverPositivos6.append(verPositivos[i])
        else:
            nfalPositivos1 = nfalPositivos6 + 1
            nfalNegativos6 = nfalNegativos6 + 1
        if razones[i]>0.6:
            nverPositivos7.append(verPositivos[i])
        else:
            nfalPositivos1 = nfalPositivos7 + 1
            nfalNegativos7 = nfalNegativos7 + 1
        if razones[i]>0.65:
            nverPositivos8.append(verPositivos[i])
        else:
            nfalPositivos1 = nfalPositivos8 + 1
            nfalNegativos8 = nfalNegativos8 + 1
        if razones[i]>0.7:
            nverPositivos9.append(verPositivos[i])
        else:
            nfalPositivos1 = nfalPositivos9 + 1
            nfalNegativos9 = nfalNegativos9 + 1
        if razones[i]>0.75:
            nverPositivos10.append(verPositivos[i])
        else:
            nfalPositivos1 = nfalPositivos10 + 1
            nfalNegativos10 = nfalNegativos10 + 1

    verPositivos1 = nverPositivos1
    verPositivos2 = nverPositivos2
    verPositivos3 = nverPositivos3
    verPositivos4 = nverPositivos4
    verPositivos5 = nverPositivos5
    verPositivos6 = nverPositivos6
    verPositivos7 = nverPositivos7
    verPositivos8 = nverPositivos8
    verPositivos9 = nverPositivos9
    verPositivos10 = nverPositivos10

    pre1 = float(len(verPositivos1))/float(((len(verPositivos1))+ (nfalPositivos1)))
    re1 = float(len(verPositivos1))/float((len(verPositivos1)+float(nfalNegativos1)))
    #f11 = float((2*(re1*pre1)))/float((re1 +pre1))
    pre2 = float(len(verPositivos2))/float(((len(verPositivos2))+(nfalPositivos2)))
    re2 = float(len(verPositivos2))/float((len(verPositivos2)+float(nfalNegativos2)))
    #f12 = float((2*(re2*pre2)))/float((re2 +pre2))
    pre3 = float(len(verPositivos3))/float(((len(verPositivos3))+(nfalPositivos3)))
    re3 = float(len(verPositivos3))/float((len(verPositivos3)+float(nfalNegativos3)))
    #f13 = float((2*(re3*pre3)))/float((re3 +pre3))
    pre4 = float(len(verPositivos4))/float(((len(verPositivos4))+(nfalPositivos4)))
    re4 = float(len(verPositivos4))/float((len(verPositivos4)+float(nfalNegativos4)))
    #f14 = float((2*(re4*pre4)))/float((re4 +pre4))
    pre5 = float(len(verPositivos5))/float(((len(verPositivos5))+(nfalPositivos5)))
    re5 = float(len(verPositivos5))/float((len(verPositivos5)+float(nfalNegativos5)))
    #f15 = float((2*(re5*pre5)))/float((re5 +pre5))
    pre6 = float(len(verPositivos6))/float(((len(verPositivos6))+(nfalPositivos6)))
    re6 = float(len(verPositivos6))/float((len(verPositivos6)+float(nfalNegativos6)))
    #f16 = float((2*(re6*pre6)))/float((re6 +pre6))
    pre7 = float(len(verPositivos7))/float(((len(verPositivos7))+(nfalPositivos7)))
    re7 = float(len(verPositivos7))/float((len(verPositivos7)+float(nfalNegativos7)))
    #f17 = float((2*(re7*pre7)))/float((re7 +pre7))
    pre8 = float(len(verPositivos8))/float(((len(verPositivos8))+(nfalPositivos8)))
    re8 = float(len(verPositivos8))/float((len(verPositivos8)+float(nfalNegativos8)))
    #f18 = float((2*(re8*pre8)))/float((re8 +pre8))
    pre9 = float(len(verPositivos9))/float(((len(verPositivos9))+(nfalPositivos9)))
    re9 = float(len(verPositivos9))/float((len(verPositivos9)+float(nfalNegativos9)))
    #f19 = float((2*(re9*pre9)))/float((re9 +pre9))
    pre10 = float(len(verPositivos10))/float(((len(verPositivos10))+(nfalPositivos10)))
    re10 = float(len(verPositivos10))/float((len(verPositivos10)+float(nfalNegativos10)))
    #f110 = float((2*(re10*pre10)))/float((re10 +pre10))

    archivo = open("C:\Deteccion_Conteo_Personas_Escenas_PUJ-1703\DATOS.txt",'a')
    re = str(re1) + ";" + str(re2) + ";" + str(re3) + ";" + str(re4) + ";" + str(re5) + ";" + str(re6) + ";" + str(re7) + ";" + str(re8) + ";" + str(re9) + ";" + str(re10) + "\n"
    archivo.write(re)
    pre = str(pre1) + ";" + str(pre2) + ";" + str(pre3) + ";" + str(pre4) + ";" + str(pre5) + ";" + str(pre6) + ";" + str(pre7) + ";" + str(pre8) + ";" + str(pre9) + ";" + str(pre10) + "\n"
    archivo.write(pre)

    print ("E",re1,re2,re3,re4,re5,re6,re7,re8,re9,re10)
    print ("P",pre1,pre2,pre3,pre4,pre5,pre6,pre7,pre8,pre9,pre10)

if __name__ == '__main__':

    # Obtencion de parametros
    args = get_args()

    # Lectura de scores y predicciones
    a  = open(args.dirscores, 'r')
    b  = open(args.dirpred, 'r')
    scores1  = a.readlines()
    preds1  = b.readlines()

    scores1 = np.asfarray(scores1,dtype='f')
    preds1 = np.asfarray(preds1,dtype='i')
    print (scores1)
    print (preds1)
    # Lectura del archivo de descripcion
    ancho,alto,coordenadas,coordenadas2 = leerDescripcion(args.tdes)
    # Non Maxima Supression
    boxesReales = NMS(ancho,alto,coordenadas,coordenadas2,scores1,preds1,args.thereshold,args.visualize,args.frame)
    # Validacion
    if len(boxesReales!=0):
        verPositivos,falPositivos,falNegativos,coordenadas,razones = validacion(boxesReales,args.ground)
        umbral = round(0.35 * len(falPositivos))
        falPositivos = falPositivos[0:umbral]
        validacion2(verPositivos,falPositivos,falNegativos,coordenadas,razones,boxesReales)
    else:
        c = open(args.ground,'r')
        coordenadas = c.readlines()
        coordenadas = np.asfarray(coordenadas)
        coordenadas = np.reshape(coordenadas,(-1,4))
        verPositivos = []
        falPositivos = []
        falNegativos = len(coordenadas)
        razones = []

    # Calculo de Errores
    print (verPositivos,falPositivos,falNegativos)
    if len(verPositivos) == 0 and len(falPositivos) == 0 and falNegativos != 0:
        pre = -1.0
        re = 0.0
        f1 = -1.0
    elif len(verPositivos) == 0 and falNegativos == 0 and len(falPositivos) != 0:
        pre = 0.0
        re = -1.0
        f1 = -1.0
    elif len(verPositivos) == 0 and falNegativos == 0 and len(falPositivos) == 0:
        pre = 1.0
        re = 1.0
        f1 = 1.0
    else:
        pre = float(len(verPositivos))/float(((len(verPositivos))+(len(falPositivos))))
        re = float(len(verPositivos))/float((len(verPositivos)+float(falNegativos)))
        f1 = float((2*(re*pre)))/float((re +pre))

    # Impresion de Resultados
    print("Personas REALES en la escena:",len(coordenadas))
    print("Personas totales CONTADAS en la escena:",len(verPositivos) + len(falPositivos))
    print("Personas correctas CONTADAS en la escena:",len(verPositivos))
    print("Personas incorrectas CONTADAS en la escena:",len(falPositivos))
    print("Personas que faltaron por contar:",falNegativos)
    print("Precision:",pre)
    print("Recall:",re)
    print("F1 SCORE:",f1)

    # Dibujar BB

    fig,ax = plt.subplots(1)
    plt.title("Resultados del conteo")
    img = cv2.imread(args.frame,0)
    img = cv2.resize(img, (args.ancho, args.alto))
    ax.imshow(img, cmap = 'gray')

    for i in range(0,len(verPositivos)):
        rect = patches.Rectangle((verPositivos[i,0],verPositivos[i,1]),verPositivos[i,2],verPositivos[i,3],linewidth=1,edgecolor='g',facecolor='none')
        ax.add_patch(rect)
    for i in range(0,len(falPositivos)):
        rect = patches.Rectangle((falPositivos[i,0],falPositivos[i,1]),falPositivos[i,2],falPositivos[i,3],linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    for i in range(0,len(coordenadas)):
        rect = patches.Rectangle((coordenadas[i,0],coordenadas[i,1]),coordenadas[i,2],coordenadas[i,3],linewidth=1,edgecolor='b',facecolor='none')
        ax.add_patch(rect)
    for i in range(0,len(razones)):
        ax.text(verPositivos[i,0],verPositivos[i,1],str(round(razones[i],2)),bbox={'facecolor':'white', 'alpha':0.8, 'pad':3},fontsize=8)

    green_patch = patches.Patch(color='g', label='Positivos Verdaderos')
    red_patch = patches.Patch(color='r', label='Positivos Falsos')
    blue_patch = patches.Patch(color='b', label='Ground Truth')
    plt.axis('off')
    plt.legend(handles=[red_patch,blue_patch,green_patch])
    ax.text(args.ancho/2, args.alto+20,"Precision:" + str(pre))
    ax.text(args.ancho/2, args.alto+40,"Recall:" + str(re))
    ax.text(args.ancho/2, args.alto+60,"F1 Score:" + str(f1))
    ax.text(0, args.alto+20,"Personas correctas totales: " + str(len(coordenadas)))
    ax.text(0, args.alto+40,"Personas correctas contadas: " + str(len(verPositivos)))
    ax.text(0, args.alto+60,"Personas incorrectas contadas: " + str(len(falPositivos)))
    ax.text(0, args.alto+80,"Personas que faltaron por contar:" + str(falNegativos))

    if args.visualize == 1:
        plt.show();
    escribirReporte(args.thereshold,args.diferencia,len(verPositivos),len(falPositivos),falNegativos,pre,re,f1,'C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/ModelosClasificacion/Reporte/ReporteCNN.csv',args.visualize)
