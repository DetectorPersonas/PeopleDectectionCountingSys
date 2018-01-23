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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d'  , "--dir"   , help="Directorio con los modelos"       , default='/Volumes/Datos/DocumentosOSX/Desktop/fisherpython/ModeloFV/')
    parser.add_argument('-dD'  , "--dirDic"   , help="Directorio con el GMM"       , default='/Volumes/Datos/DocumentosOSX/Desktop/fisherpython/ModeloFV/')
    parser.add_argument('-di' , "--dir2"  , help="Directorio con las imagenes TEST positivas" , default='/Volumes/Datos/DocumentosOSX/Desktop/IMAGENESTESTReal')
    parser.add_argument('-di2' , "--dir3"  , help="Directorio con las imagenes TEST negativas" , default='/Volumes/Datos/DocumentosOSX/Desktop/IMAGENESTESTReal')
    parser.add_argument('-na' , "--name"  , help="Nombre archivo modelo entrenado" , default='')
    parser.add_argument('-ver' , "--version", help="Version del modelo"             , default = '')
    parser.add_argument('-limP' , "--nlimPos", help="Number of training images"                     , default = 1000, type=int)
    parser.add_argument('-limN' , "--nlimNeg", help="Number of training images"                     , default = 1000, type=int)

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
    img = cv2.resize(img, (64, 128))
    surf = cv2.xfeatures2d.SIFT_create()
    _ , descriptors = surf.detectAndCompute(img,None)
    if descriptors is None:
        print("Esta imagen no encontro puntos SIFT")
        return ""
    return descriptors
def SIFTDescriptors(folder,folder2):
    caracteristicas = []
    bandera = []
    contadorPos=0
    contadorNeg=0

    files = glob.glob(folder + "/*.jpg")
    print ("Se han encontrado",args.nlimPos,"imagenes positivas")
    for file in files:
        if contadorPos<args.nlimPos:
            print ("Leyendo",file)
            c = image_descriptors(file)
            if c != "":
                caracteristicas.append(c)
                contadorPos=contadorPos+1
                bandera.append(0)
            else:
                contadorPos=contadorPos+1
                bandera.append(1)

    files = glob.glob(folder2 + "/*.jpg")
    print ("Se han encontrado",args.nlimNeg,"imagenes negativas")
    for file in files:
        if contadorNeg<args.nlimNeg:
            print ("Leyendo",file)
            c = image_descriptors(file)
            if c != "":
                caracteristicas.append(c)
                contadorNeg=contadorNeg+1
                bandera.append(0)
            else:
                contadorNeg=contadorNeg+1
                bandera.append(1)
    print ("Dimension caracteristicas",len(caracteristicas))
    print ("Dimension bandera",len(bandera))
    return caracteristicas,bandera
def getLabels(folder,folder2):
    labels = []
    contadorPos=0
    contadorNeg=0

    files = glob.glob(folder + "/*.jpg")
    for file in files:
        if contadorPos<args.nlimPos:
            contadorPos=contadorPos+1
            start = file.find('Pos')
            print(start)
            if start==-1:
                # No persona
                labels.append('1')
            else:
                # Persona
                labels.append('0')
    files = glob.glob(folder2 + "/*.jpg")
    for file in files:
        if contadorNeg<args.nlimNeg:
            contadorNeg=contadorNeg+1
            start = file.find('Pos')
            print(start)
            if start==-1:
                # No persona
                labels.append('1')
            else:
                # Persona
                labels.append('0')
    labels = np.asfarray(labels, dtype='i')
    return labels
def getFisher(means,covs,priors,caracteristicas):
    fisherVectors = []
    contador = 0
    for features in caracteristicas:
        features = np.transpose(features)
        print("Calculando Vector de Fisher de la imagen",contador)
        fisherVectors.append(fisher(features,means,covs,priors,verbose = False))
        contador = contador + 1
    fisherVectors = np.asfarray(fisherVectors, dtype='f')
    print (fisherVectors.shape)
    return fisherVectors
def accur(classifier,fisher,label,bandera):
    cont = 0
    probs = []
    preds = []
    for i in range(len(label)):
        v = fisher[i,:]
        l = label[i]
        v = v.reshape(1,-1)
        #print ('Calculando prediccion de imagen',i)
        print("Clases",classifier.classes_)
        pred = classifier.predict(v)
        pred = pred[0]
        preds.append(pred)
        prob = classifier.predict_proba(v)

        ppersona = prob[0,0]
        pnopersona = prob[0,1]

        if pred == l:
            print ("Imagen",i,":Prediccion correcta",pred,l,cont)
            cont = cont + 1
        else:
            print ("Imagen",i,":Prediccion incorrecta",pred,l,cont)

        print("SCORE PERSONA:",ppersona,"SCORE NO PERSONA",pnopersona)
        np.reshape(prob,(1,-1))
        probs.append(prob)

    #print("Predicciones correctas:",cont)
    #print("Total predicciones:",len(label))
    #print("Precision:",float(cont)/float(len(label)))
    predCorr = cont
    totPred = len(label)
    prec = float(cont)/float(len(label))
    preds = np.asfarray(preds,dtype='i')

    if bandera == True:
        return probs,predCorr,totPred,prec,preds
    else:
        return -1,predCorr,totPred,prec
def load_gmm(folder,version):
    means = np.load(folder + "MeansSVM"  + version + ".npy")
    covs = np.load(folder + "CovsSVM" + version + ".npy")
    priors = np.load(folder + "PriorsSVM" + version + ".npy")
    return means,covs,priors
def acomodar(coordenadas,bandera):
    print("Dimension vieja",len(labels))
    nuevolabels = []
    for i in range(0,len(bandera)):
        if bandera[i]!=1:
            nuevolabels.append(coordenadas[i])
    print ("Nueva dimension de labels",len(nuevolabels))
    return nuevolabels

if __name__ == '__main__':

    # Obtencion de parametros
    args = get_args()

    # Extraccion de caracteristicas
    caracteristicas,bandera = SIFTDescriptors(args.dir2, args.dir3)

    # Lectura de Labels
    labels = getLabels(args.dir2, args.dir3)
    labels = acomodar(labels,bandera)
    # Lectura del diccionario
    means,covs,priors = load_gmm(args.dirDic,args.version)

    # Calculo vectores de Fisher
    fisherVectors = getFisher(means,covs,priors,caracteristicas)

    # Importacion de los clasificadores
    classifier1 = joblib.load(args.dir + args.name + args.version +".pkl")

    # Evaluacion de los clasificadores
    print ('Resultados SVM')
    scores1,predCorr1,totPred1,prec1,preds1 = accur(classifier1,fisherVectors,labels,True)


    # Impresion de Resultados
    target_names = ['Persona', 'No Persona']
    print("Resultados Clasificador SVM")
    print("Predicciones correctas:",predCorr1)
    print("Total predicciones:",totPred1)
    print ("Accuracy SVM: ",accuracy_score(labels, preds1))
    c1 = confusion_matrix(labels, preds1)
    print ("Positivos verdaderos:",c1[0,0])
    print ("Falsos negativos:",c1[1,0])
    print ("Falsos positivos:",c1[0,1])
    print ("Negativos verdaderos:",c1[1,1])
    print(classification_report(labels, preds1, target_names=target_names))

    input('Press enter to continue: ')
