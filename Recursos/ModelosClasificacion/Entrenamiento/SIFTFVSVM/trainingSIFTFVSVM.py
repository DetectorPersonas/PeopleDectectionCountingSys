import sys, glob, argparse
import numpy as np
import math, cv2
from scipy.stats import multivariate_normal
import time
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from cyvlfeat.gmm.cygmm import cy_gmm
from cyvlfeat.fisher.cyfisher import cy_fisher
from sklearn.externals import joblib

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp'  , "--dirPos"   , help="Directorio imagenes entrenamiento positivas"            , default='/Volumes/Datos/DocumentosOSX/Desktop/IMAGENESTRAINReal')
    parser.add_argument('-dn'  , "--dirNeg"   , help="Directorio imagenes entrenamiento negativas"            , default='/Volumes/Datos/DocumentosOSX/Desktop/IMAGENESTRAINReal')
    parser.add_argument('-nn' , "--clus"  , help="Numero de Clusters"                           , default=256, type=int)
    parser.add_argument('-dgmm', "--dirgmm", help="Directorio donde se guarda el diccionario"   , default='/Volumes/Datos/DocumentosOSX/Desktop/fisherpython/ModeloFV/')
    parser.add_argument('-dsvm', "--dirsvm", help="Directorio donde se guarda el SVM"           , default='/Volumes/Datos/DocumentosOSX/Desktop/fisherpython/ModeloFV/')
    parser.add_argument('-drf' , "--dirrf"  , help="Directorio donde se guarda el RF"           , default='/Volumes/Datos/DocumentosOSX/Desktop/fisherpython/ModeloFV/')
    parser.add_argument('-nmodsvm' , "--namemodelSVM"  , help="Nombre del modelo a guardar SVM" , default='SVMFisher500.pkl')
    parser.add_argument('-nmodrf' , "--namemodelRF"  , help="Nombre del modelo a guardar RF"    , default='RFFisher500.pkl')
    parser.add_argument('-dic' , "--verdic"  , help="Numero de la version del diccionario"      , default='2')
    parser.add_argument('-arb' , "--narb", help="Number of trees RF"                            , default = 50, type=int)
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
def gmm(X, n_clusters=128, max_num_iterations=100, covariance_bound=None,
        init_mode='kmeans', init_priors=None, init_means=None, init_covars=None,
        n_repetitions=1, verbose=False):
    """Fit a Gaussian mixture model
    Parameters
    ----------
    X : [n_samples, n_features] `float32/float64` `ndarray`
        The data to be fit. One data point per row.
    n_clusters : `int`, optional
        Number of output clusters.
    max_num_iterations : `int`, optional
        The maximum number of EM iterations.
    covariance_bound : `float` or `ndarray`, optional
        A lower bound on the value of the covariance. If a float is given
        then the same value is given for all features/dimensions. If an
        array is given it should have shape [n_features] and give the
        lower bound for each feature.
    init_mode: {'rand', 'kmeans', 'custom'}, optional
        The initialization mode:
          - rand: Initial mean positions are randomly  chosen among
                  data samples
          - kmeans: The K-Means algorithm is used to initialize the cluster
                    means
          - custom: The intial parameters are provided by the user, through
                    the use of ``init_priors``, ``init_means`` and
                    ``init_covars``. Note that if those arguments are given
                    then the ``init_mode`` value is always considered as
                    ``custom``
    init_priors : [n_clusters,] `ndarray`, optional
        The initial prior probabilities on each components
    init_means : [n_clusters, n_features] `ndarray`, optional
        The initial component means.
    init_covars : [n_clusters, n_features] `ndarray`, optional
        The initial diagonal values of the covariances for each component.
    n_repetitions : `int`, optional
        The number of times the fit is performed. The fit with the highest
        likelihood is kept.
    verbose : `bool`, optional
        If ``True``, display information about computing the mixture model.
    Returns
    -------
    priors : [n_clusters] `ndarray`
        The prior probability of each component
    means : [n_clusters, n_features] `ndarray`
        The means of the components
    covars : [n_clusters, n_features] `ndarray`
        The diagonal elements of the covariance matrix for each component.
    ll : `float`
        The found log-likelihood of the input data w.r.t the fitted model
    posteriors : [n_samples, n_clusters] `ndarray`
        The posterior probability of each cluster w.r.t each data points.
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]

    if X.shape[0] == 0:
        raise ValueError('X should contain at least one row')
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("X contains Nans or Infs.")

    if n_clusters <= 0 or n_clusters > n_samples:
        raise ValueError(
            'n_clusters {} must be a positive integer smaller than the '
            'number of data points {}'.format(n_clusters, n_samples)
        )

    if max_num_iterations < 0:
        raise ValueError('max_num_iterations must be non negative')
    if n_repetitions <= 0:
        raise ValueError('n_repetitions must be a positive integer')
    if init_mode not in {'rand', 'custom', 'kmeans'}:
        raise ValueError("init_mode must be one of {'rand', 'custom', 'kmeans'")

    # Make sure we have the correct types
    X = np.ascontiguousarray(X)
    if X.dtype not in [np.float32, np.float64]:
        raise ValueError('Input data matrix must be of type float32 or float64')

    if covariance_bound is not None:
        covariance_bound = np.asarray(covariance_bound, dtype=np.float)

    if init_priors is not None:
        init_priors = np.require(init_priors, requirements='C', dtype=X.dtype)
        if init_priors.shape != (n_clusters,):
            raise ValueError('init_priors does not have the correct size')
    if init_means is not None:
        init_means = np.require(init_means, requirements='C', dtype=X.dtype)
        if init_means.shape != (n_clusters, n_features):
            raise ValueError('init_means does not have the correct size')
    if init_covars is not None:
        init_covars = np.require(init_covars, requirements='C', dtype=X.dtype)
        if init_covars.shape != (n_clusters, n_features):
            raise ValueError('init_covars does not have the correct size')

    all_inits = (init_priors, init_means, init_covars)
    if any(all_inits) and not all(all_inits):
        raise ValueError('Either all or none of init_priors, init_means and '
                         'init_covars must be set.')

    if init_mode == "custom" and not all(all_inits):
        raise ValueError('init_mode==custom implies that all initial '
                         'parameters are given')

    return cy_gmm(X, n_clusters, max_num_iterations, init_mode.encode('utf8'),
                  n_repetitions, int(verbose),
                  covariance_bound=covariance_bound, init_priors=init_priors,
                  init_means=init_means, init_covars=init_covars)
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
            start = file.find('Pos')
            contadorPos=contadorPos+1
            if start==-1:
                # No persona
                labels.append('1')
            else:
                # Persona
                labels.append('0')

    files = glob.glob(folder2 + "/*.jpg")
    for file in files:
        if contadorNeg<args.nlimNeg:
            start = file.find('Pos')
            contadorNeg=contadorNeg+1
            if start==-1:
                # No persona
                labels.append('1')
            else:
                # Persona
                labels.append('0')
    labels = np.asfarray(labels, dtype='i')
    return labels
def diccionario(caracteristicas,clusters,directorio,version):
    dicc = []
    for i in range(0,len(caracteristicas)):
        carimagen = caracteristicas[i]
        for feature in carimagen:
            dicc.append(feature)
    dicc = np.asfarray(dicc, dtype='f')

    print ("Calculando el diccionario con",clusters,"clusters")
    [means,covs,priors,ll,posteriors] = gmm(dicc, n_clusters = clusters)
    means = np.transpose(means)
    #print ('MEANS',means.shape)
    covs = np.transpose(covs)
    #print ('COVS',covs.shape)

    np.save(directorio + "MeansSVM" + args.verdic , means)
    np.save(directorio + "CovsSVM" + args.verdic , covs)
    np.save(directorio + "PriorsSVM" + args.verdic , priors)

    return means,covs,priors
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
def train(features,labels):
    # Entrenamiento usando SVM

    clf = svm.SVC(C=500.0, cache_size=500, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=True, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

    print('Entrenando el clasificador SVM...')
    clf.fit(features,labels)
    return clf
def train2(features,labels,arb):
    # Entrenamiento usando Random Forest
    clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=arb, n_jobs=1,
            oob_score=False, random_state=0, verbose=0, warm_start=False)
    print('Entrenando el clasificador Random Forest...')
    clf.fit(features, labels)
    return clf
def accur(classifier,fisher,label,bandera):
    cont = 0
    probs = []
    for i in range(len(label)):
        v = fisher[i,:]
        l = label[i]
        v = v.reshape(1,-1)
        #print ('Calculando prediccion de imagen',i)
        pred = classifier.predict(v)
        prob = classifier.predict_proba(v)

        if pred == l:
            print ("Imagen",i,":Prediccion correcta",pred,l,cont)
            cont = cont + 1
        else:
            print ("Imagen",i,":Prediccion incorrecta",pred,l,cont)

        print("Scores:",prob)

    print("Predicciones correctas:",cont)
    print("Total predicciones:",len(label))
    print("Precision:",float(cont)/float(len(label)))

    if bandera == True:
        return probs
    else:
        return -1
def mezclar(a,b):
    c = zip(a,b)
    np.random.shuffle(c)
    a = []
    b = []

    for i in range (0,len(c)):
        aux = list(c[i])
        a.append(aux[0])
        b.append(aux[1])


    print (len(a))
    b = np.asfarray(b,dtype='i')
    return a,b
def acomodar(coordenadas,bandera):
    print("Dimension vieja",len(labels))
    nuevolabels = []
    for i in range(0,len(bandera)):
        if bandera[i]!=1:
            nuevolabels.append(coordenadas[i])
    print ("Nueva dimension de labels",len(nuevolabels))
    return nuevolabels

if __name__ == '__main__':
    # Obtencion de paramatros
    args = get_args()
    folder = args.dirPos
    folder2 = args.dirNeg
    clusters = args.clus
    dirdicc = args.dirgmm

    # Extraccion de caracteristicas
    caracteristicas,bandera = SIFTDescriptors(folder,folder2)

    # Lectura de Labels
    labels = getLabels(folder,folder2)
    labels = acomodar(labels,bandera)
    '''
    # Mezclar el orden
    caracteristicas,labels = mezclar(caracteristicas,labels)

    # Acotar conjunto de Entrenamiento
    caracteristicas = caracteristicas[0:]
    labels = labels[0:]
    '''
    # Creacion del diccionario
    means,covs,priors = diccionario(caracteristicas,clusters,dirdicc,args.verdic)

    # Calculo de vectores de Fisher
    fisherVectors = getFisher(means,covs,priors,caracteristicas)

    # Entrenamiento de clasificadores
    classifier1 = train(fisherVectors,labels)

    # Evaluacion de clasificadores
    print ('Resultados SVM')
    a1 = accur(classifier1,fisherVectors,labels,False)

    # Exportacion de los clasificadores
    joblib.dump(classifier1, args.dirsvm + args.namemodelSVM, compress=5)

    #input('Press enter to continue: ')
