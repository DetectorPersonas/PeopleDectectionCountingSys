#include <iostream>
#include <windows.h>
#include <string>

using namespace std;

#define CARPETA_IMAGENES "/Users/user/Desktop/Recursos/BancosImagenes/"
#define NUMPOSITIVAS 1
#define NUMNEGATIVAS 1
#define GRAFICA_ENTRENADA "/Users/user/Desktop/Recursos/ModelosClasificacion/ModelosEntrenados/CNN/Daimler/output_graph.pb"
#define LABELS "/Users/user/Desktop/Recursos/ModelosClasificacion/ModelosEntrenados/CNN/Daimler/output_labels.txt"
#define BANCOIMAGENES "Daimler"
#define DIR_GROUNDTRUTH "/Users/user/Desktop/Recursos/ModelosClasificacion/Prueba/CNN/GroundTruth.txt"
#define DIR_PREDICCIONES "/Users/user/Desktop/Recursos/ModelosClasificacion/Prueba/CNN/Predicciones.txt"

int main() {

	string filename;
	string command;

	string dirimag;
	string outGraph;
	string outLabels;

	remove("Predicciones.txt");

	//CREACIÓN DEL FICHERO PARA MIRAR LOS ERRORES
	FILE *Fichero;
	Fichero = fopen("GroundTruth.txt", "w");
	if (Fichero == NULL){
		printf("\nError al crear el archivo de texto GroundTruth.txt... \n");
	}
	else{
		cout << "INICIO IMAGENES POSITIVAS" << endl << endl;

		for (int i = 1; i < NUMPOSITIVAS+1; i++){
			filename = "C:/Users/user/Desktop/Recursos/ModelosClasificacion/Prueba/CNN/testingCNN.py";
			command = "python ";
			dirimag = " --image ";
			outGraph = " --graph ";
			outLabels = " --labels ";

			command += filename;

			outGraph += GRAFICA_ENTRENADA;
			command += outGraph;
			outLabels += LABELS;
			command += outLabels;

			dirimag += CARPETA_IMAGENES;
			dirimag += BANCOIMAGENES;
			dirimag += "/Prueba/Persona/Pos";
			dirimag += to_string(i);
			dirimag += ".jpg";
			command += dirimag;

			fprintf(Fichero, "%s\n", "persona"); //Escribe la clase real

			cout << "Prediccion: " << i << endl;

			system(command.c_str());
		}

		cout << endl << endl <<"INICIO IMAGENES NEGATIVAS" << endl << endl;

		for (int i = 1; i < NUMNEGATIVAS+1; i++){
			filename = "C:/Users/user/Desktop/Recursos/ModelosClasificacion/Prueba/CNN/testingCNN.py";
			command = "python ";
			dirimag = " --image ";
			outGraph = " --graph ";
			outLabels = " --labels ";

			command += filename;

			outGraph += GRAFICA_ENTRENADA;
			command += outGraph;
			outLabels += LABELS;
			command += outLabels;

			dirimag += CARPETA_IMAGENES;
			dirimag += BANCOIMAGENES;
			dirimag += "/Prueba/No_Persona/neg";
			dirimag += to_string(i);
			dirimag += ".jpg";
			command += dirimag;

			fprintf(Fichero, "%s\n", "no persona"); //Escribe la clase real

			cout << "Prediccion: " << i << endl;

			system(command.c_str());
		}
	}
	fclose(Fichero);

	string dirTruth = " --dir1 ";
	string dirPredict = " --dir2 ";

	filename = "C:/Users/user/Desktop/Recursos/ModelosClasificacion/Prueba/CNN/reporteErrores.py";
	command = "python ";

	command += filename;

	dirTruth += DIR_GROUNDTRUTH;
	command += dirTruth;
	dirPredict += DIR_PREDICCIONES;
	command += dirPredict;

	cout << endl << endl << "REPORTE DE ERRORRES" << endl << endl;
	system(command.c_str());
	
	Sleep(50000);
}

