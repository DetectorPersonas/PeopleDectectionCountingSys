#include <iostream>
#include <windows.h>
#include <string>

using namespace std;

#define CARPETA_IMAGENES "/Users/user/Desktop/Recursos/BancosImagenes/"
#define BANCOIMAGENES "Personalizada"
#define LEARNINGRATE 0.02
#define EPOCAS 100
#define NUMEJEMPLOS 5000
#define BATCHSIZE 20
#define NUMNEURONS 20

int main() {
	string filename = "C:/Users/user/Desktop/Recursos/ModelosClasificacion/Entrenamiento/ANN/trainingANN.py";
	string command = "python ";
	command += filename;

	string dirTrain = " --dir ";
	string bancoUtilizado = " --banco ";
	string learningRate = " --learning ";
	string epocas = " --epocas ";
	string numEjemplos = " --numExam ";
	string batchSize = " --batch ";
	string numNeuronas = " --numNeurons ";

	dirTrain += CARPETA_IMAGENES;
	command += dirTrain;
	bancoUtilizado += BANCOIMAGENES;
	command += bancoUtilizado;
	learningRate += to_string(LEARNINGRATE);
	command += learningRate;
	epocas += to_string(EPOCAS);
	command += epocas;
	numEjemplos += to_string(NUMEJEMPLOS);
	command += numEjemplos;
	batchSize += to_string(BATCHSIZE);
	command += batchSize;
	numNeuronas += to_string(NUMNEURONS);
	command += numNeuronas;

	system(command.c_str());
	Sleep(50000);
}