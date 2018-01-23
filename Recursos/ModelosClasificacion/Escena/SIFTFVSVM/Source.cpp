////////////////////////////////////////////////////////////////////////////////
//PROCESO DE ENTRENAMIENTO MÁQUINAS ARTIFICIALES UTILIZANDO HOG
//CREACIÓN DE FICHEROS CON CARACTERÍSTICAS Y LABELS PARA LECTURA EN PYTHON
////////////////////////////////////////////////////////////////////////////////

//Librerías utilizadas
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <Windows.h>

//Definición de clases estandar
using namespace cv;
using namespace std;

//Parametros archivo Python prueba escena SIFT
#define DIRECCION_PYTHON "C:/Users/user/Desktop/Recursos/ModelosClasificacion/Escena/SIFTFVSVM/testingEscenaSIFTFVSVM.py"
#define DIRECTORIO_CARGAR_MODELO "/Users/user/Desktop/Recursos/ModelosClasificacion/ModelosEntrenados/"
#define DIRECCION_IMAGENES_CANDIDATOS "C:\\Users\\user\\Desktop\\Recursos\\TecnicasBarrido\\Candidatos"
#define NOMBRE_MODELO "SIFTFVSVM"
#define TAGENTRENAMIENTO "Personalizada"
#define TIPO_BARRIDO 1 //Bordes (1) - Exhaustivo (0)
#define DIR_DESCRIPCIONBARRIDO "C:/Users/user/Desktop/Recursos/TecnicasBarrido/DescripcionBarrido/DescripcionBarrido.txt"
#define GRAFICAR 1 //Mostrar resultados graficos
#define THRESHOLD_NMS 0.15
#define DIR_ESCENA "C:/Users/user/Desktop/Recursos/BancoEscenas/Otras/Escena.jpg"
#define GROUNDTRUTH "C:/Users/user/Desktop/Recursos/BancoEscenas/Otras/GroundTruthEscena.txt"
#define DIRGMM "/Users/user/Desktop/Recursos/ModelosClasificacion/ModelosEntrenados/DiccionarioGMM/"
#define DIFERENCIABB 0.005

int main(int argc, char** argv)
{
	string filename = DIRECCION_PYTHON;
	string command = "python ";
	command += filename;

	string dirModelo = " --dir ";
	string dirCandBordes = " --dirbordes ";
	string dirCandExhaus = " --direx ";
	string versionModelo = " --version ";
	string tipoVentaneo = " --tipovent ";
	string descripBordes = " --tdesbordes ";
	string descripExhaus = " --tdesex ";
	string graficar = " --visualize ";
	string Threshold = " --thereshold ";
	string frame = " --frame ";
	string groundTruth = " --ground ";
	string dirGMM = " --dirGMM ";
	string tag = " --tag ";
	string diferencia = " --diferencia ";
	
	string nombre = NOMBRE_MODELO;
	string BancoImag = TAGENTRENAMIENTO;
	nombre += BancoImag;
	versionModelo += nombre;
	versionModelo += ".pkl";
	command += versionModelo;

	dirModelo += DIRECTORIO_CARGAR_MODELO;
	command += dirModelo;
	dirCandBordes += DIRECCION_IMAGENES_CANDIDATOS;
	command += dirCandBordes;
	dirCandExhaus += DIRECCION_IMAGENES_CANDIDATOS;
	command += dirCandExhaus;
	tipoVentaneo += to_string(TIPO_BARRIDO);
	command += tipoVentaneo;
	descripBordes += DIR_DESCRIPCIONBARRIDO;
	command += descripBordes;
	descripExhaus += DIR_DESCRIPCIONBARRIDO;
	command += descripExhaus;
	graficar += to_string(GRAFICAR);
	command += graficar;
	Threshold += to_string(THRESHOLD_NMS);
	command += Threshold;
	frame += DIR_ESCENA;
	command += frame;
	groundTruth += GROUNDTRUTH;
	command += groundTruth;
	dirGMM += DIRGMM;
	command += dirGMM;
	tag += TAGENTRENAMIENTO;
	command += tag;
	diferencia += to_string(DIFERENCIABB);
	command += diferencia;

	system(command.c_str());
	Sleep(1000000);

	return 0;
}
