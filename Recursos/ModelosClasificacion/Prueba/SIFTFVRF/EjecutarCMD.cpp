#include <iostream>
#include <string>
#include <windows.h>

using namespace std;

//Parametros archivo Python entrenamiento SIFT FV SVM
#define DIRECCION_PYTHON "C:/Users/user/Desktop/Recursos/ModelosClasificacion/Prueba/SIFTFVRF/testingSIFT.py"
#define DIRECCION_IMAGENES_POSITIVAS "C:\\Users\\user\\Desktop\\Recursos\\BancosImagenes\\Personalizado\\Prueba\\Persona"
#define DIRECCION_IMAGENES_NEGATIVAS "C:\\Users\\user\\Desktop\\Recursos\\BancosImagenes\\Personalizado\\Prueba\\No_Persona"
#define DIRECTORIO_CARGAR_GMM "/Users/user/Desktop/Recursos/ModelosClasificacion/ModelosEntrenados/DiccionarioGMM/"
#define DIRECTORIO_CARGAR_MODELO "/Users/user/Desktop/Recursos/ModelosClasificacion/ModelosEntrenados/"
#define NOMBRE_MODELO "SIFTFVRF"
#define VERSION "Personalizada"

int main() {

	string filename = DIRECCION_PYTHON;
	string command = "python ";
	command += filename;

	string dirImagPos = " --dir2 ";
	string dirImagNeg = " --dir3 ";
	string dirMod = " --dir ";
	string nameSVM = " --name ";
	string version = " --version ";
	string dirDic = " --dirDic ";

	dirImagPos += DIRECCION_IMAGENES_POSITIVAS;
	command += dirImagPos;
	dirImagNeg += DIRECCION_IMAGENES_NEGATIVAS;
	command += dirImagNeg;
	dirMod += DIRECTORIO_CARGAR_MODELO;
	command += dirMod;
	nameSVM += NOMBRE_MODELO;
	command += nameSVM;
	version += VERSION;
	command += version;
	dirDic += DIRECTORIO_CARGAR_GMM;
	command += dirDic;

	system(command.c_str());
	Sleep(1000000);
}