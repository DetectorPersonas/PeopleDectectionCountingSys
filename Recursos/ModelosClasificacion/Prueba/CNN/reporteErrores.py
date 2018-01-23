import sys, glob, argparse
import numpy as np
import math, cv2
from scipy.stats import multivariate_normal
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d1' , "--dir1", help="Directorio con labels teoricos" , default='E:\DocumentosOSX\TrabajoGrado/teorico.txt')
    parser.add_argument("-d2" , "--dir2", help="Directorio con lables predecidos", default = 'E:\DocumentosOSX\TrabajoGrado/predecido.txt')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    # Obtencion de parametros
    args = get_args()

    target_names = ['persona', 'no persona']
    teoricos  = open(args.dir1, 'r')
    predecidos  = open(args.dir2, 'r')

    teoricos = teoricos.readlines()
    #teoricos = np.asfarray(teoricos)

    predecidos = predecidos.readlines()
    #predecidos = np.asfarray(predecidos)

    buenos = 0
    for i in range(0,len(teoricos)):
        if teoricos[i] == predecidos[i]:
            buenos = buenos + 1

    print("Reporte de errores del clasificador")
    print("Predicciones correctas:",buenos)
    print("Total predicciones:",len(teoricos))
    print ("Accuracy: ",accuracy_score(teoricos, predecidos))
    c1 = confusion_matrix(teoricos, predecidos)
    print ("Positivos verdaderos:",c1[0,0])
    print ("Falsos negativos:",c1[0,1])
    print ("Falsos positivos:",c1[1,0])
    print ("Negativos verdaderos:",c1[1,1])
    print(classification_report(teoricos, predecidos, target_names=target_names))

    file = open("C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/ModelosClasificacion/Prueba/CNN/Errores.txt","w")
    file.write("Predicciones correctas:"+str(buenos)+"\n")
    file.write("Total predicciones:"+str(len(teoricos))+"\n")
    file.write("Accuracy: "+str(accuracy_score(teoricos, predecidos))+"\n")
    file.write("Positivos verdaderos:"+str(c1[0,0])+"\n")
    file.write("Falsos negativos:"+str(c1[1,0])+"\n")
    file.write("Falsos positivos:"+str(c1[0,1])+"\n")
    file.write("Negativos verdaderos:"+str(c1[1,1])+"\n")
    file.write(classification_report(teoricos, predecidos, target_names=target_names))
    file.close()

    input('Press enter to continue: ')
