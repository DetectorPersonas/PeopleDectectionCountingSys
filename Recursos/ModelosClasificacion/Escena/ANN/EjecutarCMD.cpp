#include <iostream>
#include <windows.h>
#include <string>

using namespace std;

#define CARPETA_IMAGENES "/Users/user/Desktop/Recursos/TecnicasBarrido/Candidatos/"
#define GRAFICA_ENTRENADA "/Users/user/Desktop/Recursos/ModelosClasificacion/ModelosEntrenados/ANN/"
#define BANCOIMAGENES "Daimler"
#define DIR_DESCRIPCIONBARRIDO "/Users/user/Desktop/Recursos/TecnicasBarrido/DescripcionBarrido/DescripcionBarrido.txt"
#define DIR_ESCENA "C:/Users/user/Desktop/Recursos/BancoEscenas/Otras/Escena.jpg"
#define GROUNDTRUTH "C:/Users/user/Desktop/Recursos/BancoEscenas/Otras/GroundTruthEscena.txt"
#define GRAFICAR 1 //Mostrar resultados graficos
#define THRESHOLD_NMS 0.35
#define DIR_SCORES "C:/Users/user/Desktop/Recursos/ModelosClasificacion/Escena/ANN/Scores.txt"
#define DIR_PREDICCIONES "C:/Users/user/Desktop/Recursos/ModelosClasificacion/Escena/CNN/Predicciones.txt"
#define DIFERENCIABB 0.2

int main() {
	string filename;
	string command;

	remove("Predicciones.txt");
	remove("Scores.txt");

	string dirimag;
	string bancoUtilizado;

	int numCandidatos = 0;

	//LECTURA FICHERO CON NUMERO DE CANDIDATOS
	FILE *Fichero;
	Fichero = fopen("C:/Users/user/Desktop/Recursos/TecnicasBarrido/DescripcionBarrido/NumeroCandidatos.txt", "r");
	
	if (Fichero == NULL){
		printf("\nError al crear el archivo de texto NumeroCandidatos.txt... \n");
	}
	else{
		fscanf(Fichero, "%d", &numCandidatos);
	}
	fclose(Fichero);

	cout << "INICIO LECTURA CANDIDATOS" << endl << endl;

	for (int i = 1; i < numCandidatos + 1; i++){
		filename = "C:/Users/user/Desktop/Recursos/ModelosClasificacion/Escena/ANN/testingANN.py";
		command = "python ";
		dirimag = " --image ";
		bancoUtilizado = " --banco ";

		command += filename;

		dirimag += CARPETA_IMAGENES;
		dirimag += to_string(i);
		dirimag += ".jpg";
		command += dirimag;
		bancoUtilizado += BANCOIMAGENES;
		command += bancoUtilizado;

		cout << "Prediccion: " << i << endl;

		system(command.c_str());
	}
	
	string descripFrame = " --tdes ";
	string dirScores = " --dirscores ";
	string dirPredic = " --dirpred ";
	string visua = " --visualize ";
	string threshold = " --thereshold ";
	string frame = " --frame ";
	string ground = " --ground ";
	string diferencia = " --diferencia ";

	filename = "C:/Users/user/Desktop/Recursos/ModelosClasificacion/Escena/ANN/testingEscenaANN.py";
	command = "python ";

	command += filename;
	
	descripFrame += DIR_DESCRIPCIONBARRIDO;
	command += descripFrame;
	dirScores += DIR_SCORES;
	command += dirScores;
	dirPredic += DIR_PREDICCIONES;
	command += dirPredic;
	visua += to_string(GRAFICAR); 
	command += visua;
	threshold += to_string(THRESHOLD_NMS);
	command += threshold;
	frame += DIR_ESCENA;
	command += frame;
	ground += GROUNDTRUTH;
	command += ground;
	diferencia += to_string(DIFERENCIABB);
	command += diferencia;

	cout << endl << endl << "REPORTE DE ERRORRES" << endl << endl;
	system(command.c_str());
	
	Sleep(50000);
}

