#include <iostream>
#include <string>
#include <windows.h>

using namespace std;

//Parametros archivo Python entrenamiento SIFT FV SVM
#define DIRECCION_PYTHON "C:/Users/user/Desktop/Recursos/ModelosClasificacion/Entrenamiento/SIFTFVRF/trainingSIFT.py"
#define DIRECCION_IMAGENES_POSITIVAS "C:\\Users\\user\\Desktop\\Recursos\\BancosImagenes\\Personalizado\\Entrenamiento\\Persona"
#define DIRECCION_IMAGENES_NEGATIVAS "C:\\Users\\user\\Desktop\\Recursos\\BancosImagenes\\Personalizado\\Entrenamiento\\No_Persona"
#define NUM_CLUSTERS 256
#define DIRECTORIO_GUARDAR_GMM "/Users/user/Desktop/Recursos/ModelosClasificacion/ModelosEntrenados/DiccionarioGMM/"
#define DIRECTORIO_GUARDAR_MODELO "/Users/user/Desktop/Recursos/ModelosClasificacion/ModelosEntrenados/"
#define NOMBRE_MODELO "SIFTFVRF"
#define NUM_ARBOLES 50
#define VERSION "Personalizada"

int main() {

	string filename = DIRECCION_PYTHON;
	string command = "python ";
	command += filename;

	string dirImagPos = " --dirPos ";
	string dirImagNeg = " --dirNeg ";
	string numClus = " --clus ";
	string dirGMM = " --dirgmm ";
	string dirRF = " --dirrf ";
	string nameRF = " --namemodelRF ";
	string numImag = " --nlim ";
	string numArb = " --narb ";
	string version = " --verdic ";

	dirImagPos += DIRECCION_IMAGENES_POSITIVAS;
	command += dirImagPos;
	dirImagNeg += DIRECCION_IMAGENES_NEGATIVAS;
	command += dirImagNeg;
	numClus += to_string(NUM_CLUSTERS);
	command += numClus;
	dirGMM += DIRECTORIO_GUARDAR_GMM;
	command += dirGMM;
	dirRF += DIRECTORIO_GUARDAR_MODELO;
	command += dirRF;

	string nombre = NOMBRE_MODELO;
	string BancoImag = "Personalizada.pkl";
	nombre += BancoImag;
	nameRF += nombre;
	command += nameRF;

	numArb += to_string(NUM_ARBOLES);
	command += numArb;

	version += VERSION;
	command += version;

	system(command.c_str());
	Sleep(1000000);
}