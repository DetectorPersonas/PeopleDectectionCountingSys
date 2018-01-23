import sys, glob, argparse,os
import numpy as np
import math, cv2
from scipy.stats import multivariate_normal
import time
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from cyvlfeat.gmm.cygmm import cy_gmm
from cyvlfeat.fisher.cyfisher import cy_fisher
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
import matplotlib.patches as patches

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d'  , "--dir"   , help="Directorio con los modelos"       , default='E:/DocumentosOSX/Desktop/fisherpython/ModeloFV/')
    parser.add_argument('-dg'  , "--dirGMM"   , help="Directorio del GMM"       , default='E:/DocumentosOSX/Desktop/fisherpython/ModeloFV/')
    parser.add_argument('-t'  , "--tag"   , help="tag banco imagenes"       , default='E:/DocumentosOSX/Desktop/fisherpython/ModeloFV/')
    parser.add_argument('-dib' , "--dirbordes"  , help="Directorio con las imagenes TEST BORDES" , default='E:/DocumentosOSX/TrabajoGrado/TecnicasBarrido/Bordes/BarridoBordes/Candidatos')
    parser.add_argument('-diex' , "--direx"  , help="Directorio con las imagenes TEST EX" , default='E:/DocumentosOSX/TrabajoGrado/TecnicasBarrido/Exhaustivo/BarridoVentana/Candidatos')
    parser.add_argument('-ver' , "--version", help="Version del modelo"             , default = '2000')
    parser.add_argument('-tipo' , "--tipovent"  , help="Ventaneo a utilizar 0(Exhaustivo) - 1(Bordes)"          , default=1, type=int)
    parser.add_argument('-tdb' , "--tdesbordes", help="Directory of txt with description BORDES" , default='E:/DocumentosOSX/TrabajoGrado/TecnicasBarrido/Bordes/BarridoBordes/DescripcionBarridoBordes.txt')
    parser.add_argument('-tde' , "--tdesex", help="Directory of txt with description EXHAUSTIVO" , default='E:/DocumentosOSX/TrabajoGrado/TecnicasBarrido/Exhaustivo/BarridoVentana/DescripcionBarridoExhaustivo.txt')
    parser.add_argument('-gra' , "--visualize"  , help="Mostrar resultados graficos 0(No) - 1(Si)"          , default=0, type=int)
    parser.add_argument('-ther' , "--thereshold"  , help="Thereshold para NMS"          , default=0.5, type=float)
    parser.add_argument('-fr' , "--frame"  , help="Directorio del frame"          , default='E:/DocumentosOSX/TrabajoGrado/TecnicasBarrido/Exhaustivo/BarridoVentana/Factor1.JPG')
    parser.add_argument('-gr' , "--ground"  , help="Directorio del txt con ground truth"          , default='E:\DocumentosOSX\TrabajoGrado/coord.txt')
    parser.add_argument('-dif' , "--diferencia"  , help="Thereshold para diferencia BBs"          , default=0.35, type=float)
    args = parser.parse_args()
    return args
def fisher(x, means, covariances, priors, normalized=True, square_root=True,
           improved=True, fast=False, verbose=True):
    r"""
    Computes the Fisher vector encoding of the vectors ``x`` relative to
    the diagonal covariance Gaussian mixture model with ``means``,
    ``covariances``, and prior mode probabilities ``priors``.
    By default, the standard Fisher vector is computed.
    Parameters
    ----------
    x : [D, N]  `float32` `ndarray`
        One column per data vector (e.g. a SIFT descriptor)
    means :  [F, N]  `float32` `ndarray`
        One column per GMM component.
    covariances :  [F, N]  `float32` `ndarray`
        One column per GMM component (covariance matrices are assumed diagonal,
        hence these are simply the variance of each data dimension).
    priors :  [F, N]  `float32` `ndarray`
        Equal to the number of GMM components.
    normalized : `bool`, optional
        If ``True``, L2 normalize the Fisher vector.
    square_root : `bool`, optional
        If ``True``, the signed square root function is applied to the return
        vector before normalization.
    improved : `bool`, optional
        If ``True``, compute the improved variant of the Fisher Vector. This is
        equivalent to specifying the ``normalized`` and ``square_root` options.
    fast : `bool`, optional
        If ``True``, uses slightly less accurate computations but significantly
        increase the speed in some cases (particularly with a large number of
        Gaussian modes).
    verbose: `bool`, optional
        If ``True``, print information.
    Returns
    -------
    enc : [k, 1] `float32` `ndarray`
        A vector of size equal to the product of
        ``k = 2 * the n_data_dimensions * n_components``.
    """
    # check for None
    if x is None or means is None or covariances is None or priors is None:
        raise ValueError('A required input is None')

    # validate the gmm parameters
    D = means.shape[0]  # the feature dimensionality
    K = means.shape[1]  # the number of GMM modes
    # N = x.shape[1] is the number of samples
    if covariances.shape[0] != D:
        raise ValueError('covariances and means do not have the same '
                         'dimensionality')

    if priors.ndim != 1:
        raise ValueError('priors has an unexpected shape')

    if covariances.shape[1] != K or priors.shape[0] != K:
        raise ValueError('covariances or priors does not have the correct '
                         'number of modes')

    if x.shape[0] != D:
        raise ValueError('x and means do not have the same dimensionality')

    return cy_fisher(x, means, covariances, priors,
                     np.int32(normalized), np.int32(square_root),
                     np.int32(improved), np.int32(fast), np.int32(verbose))
def image_descriptors(file):
    img = cv2.imread(file, 0)
    img = cv2.resize(img, (256, 256))
    surf = cv2.xfeatures2d.SIFT_create()
    _ , descriptors = surf.detectAndCompute(img,None)
    if descriptors is None:
        print("Esta imagen no encontro puntos SIFT")
        return ""
    return descriptors
def SIFTDescriptors(folder):
    caracteristicas = []
    bandera = []
    files = glob.glob(folder + "/*.JPG")
    print ("Se han encontrado",len(files),"imagenes")
    for file in files:
        print ("Leyendo",file)
        c = image_descriptors(file)
        if c != "":
            bandera.append(0)
            caracteristicas.append(c)
        else:
            bandera.append(1)
    return caracteristicas,bandera
def getFisher(means,covs,priors,caracteristicas):
    fisherVectors = []
    contador = 0
    for features in caracteristicas:
        features = np.transpose(features)
        print("Calculando Vector de Fisher de la imagen",contador)
        #print features.shape
        v = fisher(features,means,covs,priors,verbose = False)
        fisherVectors.append(v)
        contador = contador + 1
    fisherVectors = np.asfarray(fisherVectors, dtype='f')
    print (fisherVectors.shape)
    return fisherVectors,len(fisherVectors)
def load_gmm(folder,version):
    means = np.load(folder + "MeansSVM"  + version + ".npy")
    covs = np.load(folder + "CovsSVM" + version + ".npy")
    priors = np.load(folder + "PriorsSVM" + version + ".npy")
    return means,covs,priors
def predecir(classifier,carHOG,numimag):

    cont = 0
    probs = []
    preds = []
    for i in range(0,numimag):

        v = carHOG[i,:]
        v = v.reshape(1,-1)

        #Calculando prediccion de la imagen
        pred = classifier.predict(v)
        pred = pred[0]
        preds.append(pred)

        if pred ==0:
            resultado = 'NO PERSONA'
        else:
            resultado = 'PERSONA'
            cont = cont + 1

        prob = classifier.predict_proba(v)
        ppersona = prob[0,0]
        pnopersona = prob[0,1]

        print ("Imagen",i+1,resultado,prob)

        np.reshape(prob,(1,-1))
        probs.append(prob)

    print("PERSONAS:",cont,"NO PERSONAS:",numimag - cont)
    preds = np.asfarray(preds,dtype='i')
    return probs,preds
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
def NMS(ancho,alto,coordenadas,scores,preds,th,visualize,frame):

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

        factor = math.pow(2,i)
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
    ax.imshow(img, cmap = 'gray')
    #print ("LENGHTTT:",len(boxes))
    for i in range(0,len(boxes)):

        if preds[i] == 1:

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
            personaScore = max(scor[0,1],scor[0,0])

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
                if abs(scor1-scor2) > args.diferencia:
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

    for i in range(0,len(boxesReales)):
        rect = patches.Rectangle((boxesReales[i,0],boxesReales[i,1]),boxesReales[i,5],boxesReales[i,6],linewidth=1,edgecolor='r',facecolor='none')
        ax2.add_patch(rect)

    return boxesReales
def acomodar(coordenadas,bandera):
    coord = []
    ncoord = []
    contador = 0
    for i in range(0,len(coordenadas)):
        esc = coordenadas[i]
        for j in esc:
            if bandera[contador] == 0:
                #print "PASO", j
                coord.append(j)
            contador = contador + 1

        coord = np.asfarray(coord)
        ncoord.append(coord)
        coord = []

    print (len(ncoord),len(ncoord[0])+len(ncoord[1]))
    return ncoord
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
        falNegativos = len(coordenadas)
        return verPositivos,falPositivos,falNegativos,coordenadas

    # Caso 2: Se predicen personas cuando en realidad no hay
    # verPositivos = 0,falPositivos = boxesReales, falNegativos=len(coordenadas)
    if len(boxesReales)!=0 and len(coordenadas)==0:
        print ("\n\nCASO 2: SE PREDICEN PERSONAS CUANDO EN REALIDAD NO HAY")
        verPositivos = []
        falPositivos = boxesReales
        falNegativos = 0
        return verPositivos,falPositivos,falNegativos,coordenadas

    # Caso 3: Se predicen mas personas de las que realmente hay
    if len(coordenadas) < len(boxesReales):
        print ("\n\nCASO 3: MAS PERSONAS DE LAS QUE REALMENTE HAY")
        indices = [];
        # Algoritmo para determinar pares de BB mas cercanos y que se traslapen:
        for i in range(0,len(coordenadas)):
            indice = -1
            # Recorro los BB teoricos
            Cxt = centrosXt[i]
            Cyt = centrosYt[i]
            dif = 10000000
            for j in range(0,len(boxesReales)):
                # Recorro los BB experimentales
                Cxe = centrosXe[j]
                Cye = centrosYe[j]
                if (math.sqrt(math.pow(abs(Cxt - Cxe),2) + math.pow(abs(Cyt - Cye),2)) < dif):
                    # Encontrar el area de traslape
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
                        indice = j
            if indice != -1:
                indices.append(indice)

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
        return verPositivos,falPositivos,falNegativos,coordenadas

    # Caso 4: Se predicen menos personas de las que realmente hay
    if len(coordenadas) > len(boxesReales):
        print ("\n\nCASO 4: MENOS PERSONAS DE LAS QUE REALMENTE HAY")
        indices = []
        yaEvaluados = []
        distancias = []
        # Algoritmo para determinar pares de BB mas cercanos y que se traslapen:
        for i in range(0,len(boxesReales)):
            indice = -1
            indice2 = -1
            # Recorro los BB experimentales
            Cxe = centrosXe[i]
            Cye = centrosYe[i]
            dif = 10000000
            for j in range(0,len(coordenadas)):
                # Recorro los BB teoricos
                Cxt = centrosXt[j]
                Cyt = centrosYt[j]
                #print "experimental",i,"teorica",j,"dife",math.sqrt(math.pow(abs(Cxt - Cxe),2) + math.pow(abs(Cyt - Cye),2)),"difac",dif
                if (math.sqrt(math.pow(abs(Cxt - Cxe),2) + math.pow(abs(Cyt - Cye),2)) < dif):
                    # Encontrar el area de traslape
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
            if indice!= -1:
                indices.append(indice)
                yaEvaluados.append(indice2)
                distancias.append(d)

        #print indices
        #print yaEvaluados
        #print distancias

        # Algoritmo para descartar multiples traslapes en un mismo punto del GT
        #print boxesReales
        #print coordenadas
        nuevoEvaluados = []
        nuevoIndices = []
        nuevoDistancias = []

        for i in range(0,len(indices)):
            teorico = yaEvaluados[i]
            if not teorico in nuevoEvaluados:
                nuevoEvaluados.append(teorico)
                nuevoIndices.append(indices[i])
                nuevoDistancias.append(distancias[i])
            else:
                # Buscar la escena
                for j in range(0,len(nuevoEvaluados)):
                    # Si la encuentra
                    if nuevoEvaluados[j] == teorico:
                        if nuevoDistancias[j] > distancias[i]:
                             nuevoEvaluados[j] = teorico
                             nuevoIndices[j] = indices[i]
                             nuevoDistancias[j] = distancias[i]

        indices = nuevoIndices
        yaEvaluados = nuevoEvaluados
        distancias = nuevoDistancias

        #print indices
        #print yaEvaluados
        #print distancias
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
        return verPositivos,falPositivos,falNegativos,coordenadas
def escribirReporte(therNMS,therDIF,VP,FP,FN,P,R,F1,folder,bandera):
    if bandera == 1:
        print ("Se escribió el reporte de esta escena")
        archivo = open(folder,'a')
        linea = str(therNMS) + ";" + str(therDIF) + ";" + str(VP) + ";" + str(FP) + ";" + str(FN) + ";" + str(P) + ";" + str(R) + ";" + str(F1) + "\n"
        archivo.write(linea)

if __name__ == '__main__':

    # Obtencion de parametros
    args = get_args()

    if args.tipovent == 1:

        # Ventaneo por Bordes
        # Extraccion de caracteristicas del directorio de bordes
        ancho,alto,coordenadas,coordenadas2 = leerDescripcion(args.tdesbordes)
        caracteristicas,bandera = SIFTDescriptors(args.dirbordes)

    else:

        # Ventaneo Exhaustivo
        # Extraccion de caracteristicas del directorio de Exhaustivo
        ancho,alto,coordenadas,coordenadas2 = leerDescripcion(args.tdesex)
        caracteristicas,bandera = SIFTDescriptors(args.direx)

    # Funcion para quitar de la descripcion los parches que no sirven
    coordenadas = acomodar(coordenadas,bandera)


    # Lectura del diccionario
    means,covs,priors = load_gmm(args.dirGMM,args.tag)
    # Calculo vectores de Fisher
    fisherVectors,numimag = getFisher(means,covs,priors,caracteristicas)

    # Importacion de los clasificadores
    classifier1 = joblib.load(args.dir + args.version)
    #classifier2 = joblib.load(args.dir + args.version)

    # Evaluacion de clasificadores
    scores1,preds1 = predecir(classifier1,fisherVectors,numimag)
    # Non Maxima Supression
    boxesReales = NMS(ancho,alto,coordenadas,scores1,preds1,args.thereshold,args.visualize,args.frame)
    # Validacion
    verPositivos,falPositivos,falNegativos,coordenadas = validacion(boxesReales,args.ground)

    # Calculo de Errores
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

    green_patch = patches.Patch(color='g', label='Positivos Verdaderos')
    red_patch = patches.Patch(color='r', label='Positivos Falsos')
    blue_patch = patches.Patch(color='b', label='Ground Truth')
    plt.axis('off')
    plt.legend(handles=[red_patch,blue_patch,green_patch])
    ax.text(500, 660,"Precision:" + str(pre))
    ax.text(500, 680,"Recall:" + str(re))
    ax.text(500, 700,"F1 Score:" + str(f1))
    ax.text(0, 660,"Personas correctas totales: " + str(len(coordenadas)))
    ax.text(0, 680,"Personas correctas contadas: " + str(len(verPositivos)))
    ax.text(0, 700,"Personas incorrectas contadas: " + str(len(falPositivos)))
    ax.text(0, 720,"Personas que faltaron por contar:" + str(falNegativos))

    if args.visualize == 1:
        plt.show();
    escribirReporte(args.thereshold,args.diferencia,len(verPositivos),len(falPositivos),falNegativos,pre,re,f1,'E:\DocumentosOSX\TrabajoGrado/Reporte.csv',args.visualize)
