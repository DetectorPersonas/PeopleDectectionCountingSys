#include <iostream>
#include <string>
#include <windows.h>

using namespace std;

//Parametros archivo Python entrenamiento SIFT FV SVM
#define DIRECCION_PYTHON "C:/Users/user/Desktop/Recursos/ModelosClasificacion/Entrenamiento/SIFTFVSVM/trainingSIFT.py"
#define DIRECCION_IMAGENES_POSITIVAS "C:\\Users\\user\\Desktop\\Recursos\\BancosImagenes\\Personalizado\\Entrenamiento\\Persona"
#define DIRECCION_IMAGENES_NEGATIVAS "C:\\Users\\user\\Desktop\\Recursos\\BancosImagenes\\Personalizado\\Entrenamiento\\No_Persona"
#define NUM_CLUSTERS 256
#define DIRECTORIO_GUARDAR_GMM "/Users/user/Desktop/Recursos/ModelosClasificacion/ModelosEntrenados/DiccionarioGMM/"
#define DIRECTORIO_GUARDAR_MODELO "/Users/user/Desktop/Recursos/ModelosClasificacion/ModelosEntrenados/"
#define NOMBRE_MODELO "SIFTFVSVM"
#define VERSION "Personalizada"

int main() {

	string filename = DIRECCION_PYTHON;
	string command = "python ";
	command += filename;

	string dirImagPos = " --dirPos ";
	string dirImagNeg = " --dirNeg ";
	string numClus = " --clus ";
	string dirGMM = " --dirgmm ";
	string dirSVM = " --dirsvm ";
	string nameSVM = " --namemodelSVM ";
	string numImag = " --nlim ";
	string version = " --verdic ";

	dirImagPos += DIRECCION_IMAGENES_POSITIVAS;
	command += dirImagPos;
	dirImagNeg += DIRECCION_IMAGENES_NEGATIVAS;
	command += dirImagNeg;
	numClus += to_string(NUM_CLUSTERS);
	command += numClus;
	dirGMM += DIRECTORIO_GUARDAR_GMM;
	command += dirGMM;
	dirSVM += DIRECTORIO_GUARDAR_MODELO;
	command += dirSVM;

	string nombre = NOMBRE_MODELO;
	string BancoImag = "Personalizada.pkl";
	nombre += BancoImag;
	nameSVM += nombre;
	command += nameSVM;

	version += VERSION;
	command += version;

	system(command.c_str());
	Sleep(1000000);
}