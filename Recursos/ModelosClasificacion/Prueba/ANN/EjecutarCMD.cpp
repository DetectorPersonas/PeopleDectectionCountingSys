#include <iostream>
#include <windows.h>
#include <string>

using namespace std;

#define DIRIMAGENES "/Users/user/Desktop/Recursos/BancosImagenes/"
#define NUMPOSITIVAS 1000
#define NUMNEGATIVAS 1000
#define BANCOIMAGENES "Daimler"
#define DIR_GROUNDTRUTH "/Users/user/Desktop/Recursos/ModelosClasificacion/Prueba/ANN/GroundTruth.txt"
#define DIR_PREDICCIONES "/Users/user/Desktop/Recursos/ModelosClasificacion/Prueba/ANN/Predicciones.txt"

int main() {

	string filename;
	string command;

	string dirImage;
	string bancoUtilizado;

	remove("Predicciones.txt");

	//CREACIÓN DEL FICHERO PARA MIRAR LOS ERRORES
	FILE *Fichero;
	Fichero = fopen("GroundTruth.txt", "w");
	if (Fichero == NULL){
		printf("\nError al crear el archivo de texto GroundTruth.txt... \n");
	}
	else{
		cout << "INICIO IMAGENES POSITIVAS" << endl << endl;

		for (int i = 1; i < NUMPOSITIVAS + 1; i++){
			filename = "C:/Users/user/Desktop/Recursos/ModelosClasificacion/Prueba/ANN/testingANN.py";
			command = "python ";
			dirImage = " --image ";
			bancoUtilizado = " --banco ";

			command += filename;

			dirImage += DIRIMAGENES;
			dirImage += BANCOIMAGENES;
			dirImage += "/Prueba/Persona/Pos";
			dirImage += to_string(i);
			dirImage += ".jpg";
			command += dirImage;
			bancoUtilizado += BANCOIMAGENES;
			command += bancoUtilizado;

			fprintf(Fichero, "%s\n", "persona"); //Escribe la clase real

			cout << "Prediccion: " << i << endl;

			system(command.c_str());
		}

		cout << endl << endl << "INICIO IMAGENES NEGATIVAS" << endl << endl;

		for (int i = 1; i < NUMNEGATIVAS + 1; i++){
			filename = "C:/Users/user/Desktop/Recursos/ModelosClasificacion/Prueba/ANN/testingANN.py";
			command = "python ";
			dirImage = " --image ";
			bancoUtilizado = " --banco ";

			command += filename;

			dirImage += DIRIMAGENES;
			dirImage += BANCOIMAGENES;
			dirImage += "/Prueba/No_Persona/neg";
			dirImage += to_string(i);
			dirImage += ".jpg";
			command += dirImage;
			bancoUtilizado += BANCOIMAGENES;
			command += bancoUtilizado;

			fprintf(Fichero, "%s\n", "no persona"); //Escribe la clase real

			cout << "Prediccion: " << i << endl;

			system(command.c_str());
		}
	}
	fclose(Fichero);

	string dirTruth = " --dir1 ";
	string dirPredict = " --dir2 ";

	filename = "C:/Users/user/Desktop/Recursos/ModelosClasificacion/Prueba/ANN/reporteErrores.py";
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