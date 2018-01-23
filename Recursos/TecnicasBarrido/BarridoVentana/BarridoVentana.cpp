#include <Windows.h>
#include <math.h>
#include <direct.h>
#include <stdio.h>
#include <string>

#include "opencv2\highgui.hpp"
#include "opencv2\imgproc.hpp"

using namespace cv;
using namespace std;

int main(int argc, const char** argv)
{
	//Elimina el directorio con candidatos de la escena anterior
	string Path = "C:\\Users\\user\\Desktop\\Recursos\\TecnicasBarrido\\Candidatos";
	string folderRemoveCommand = "rmdir /S /Q " + Path;
	system(folderRemoveCommand.c_str());

	//Elimina el fichero con la información sobre las coordenadas de los candidatos
	remove("C:\\Users\\user\\Desktop\\Recursos\\TecnicasBarrido\\DescripcionBarrido.txt");

	//Creación del directorio para guardar ventanas
	string folderName = "C:\\Users\\user\\Desktop\\Recursos\\TecnicasBarrido\\Candidatos";
	string folderCreateCommand = "mkdir " + folderName;
	system(folderCreateCommand.c_str());

	folderName = "C:\\Users\\user\\Desktop\\Recursos\\TecnicasBarrido\\\BarridoVentana\\Resultados";
	folderCreateCommand = "mkdir " + folderName;
	system(folderCreateCommand.c_str());

	//Parametros con el tamaño real de la imagen
	int Real_Columnas = 960;
	int Real_Filas = 640;

	//Parametros para la ventana
	int windows_n_cols = 80;
	int windows_n_rows = 160; //Siempre debe ser el doble del ancho
	int StepSlide = 20;

	int NumeroNombre = 1;
	int ContadorCandidatos = 1;

	Mat LoadedImage;

	LoadedImage = imread("C:\\Users\\user\\Desktop\\Recursos\\BancoEscenas\\Otras\\Escena.jpg", IMREAD_COLOR);

	//CREACIÓN DEL FICHERO PARA BLOQUE POST-PROCESAMIENTO
	FILE *Fichero;
	Fichero = fopen("C:\\Users\\user\\Desktop\\Recursos\\TecnicasBarrido\\DescripcionBarrido\\DescripcionBarrido.txt", "w");
	if (Fichero == NULL){
		printf("\nError al crear el archivo de texto DescripcionBarridoExhaustivo.txt... \n");
	}
	else{
		fprintf(Fichero, "%d\n", windows_n_cols); //Escribe el ancho de la ventana
		fprintf(Fichero, "%d\n", windows_n_rows); //Escribe el alto de la ventana
		fprintf(Fichero, "%c\n", '*'); //Indica el final info sobre ventana

		for (int Factor = 0; Factor < 2; Factor++) {

			int Escala = pow(2, Factor);
			printf("Factor: %d \n", Escala);

			resize(LoadedImage, LoadedImage, Size(Real_Columnas / Escala, Real_Filas / Escala));
			//namedWindow(cv::format("Escala %d", Escala), WINDOW_AUTOSIZE);
			//imshow(cv::format("Escala %d", Escala), LoadedImage);
			imwrite(cv::format("Resultados/Factor %d.JPG", Escala), LoadedImage);
			waitKey(1000);

			Mat DrawResultGrid = LoadedImage.clone();

			for (int row = 0; row <= LoadedImage.rows - windows_n_rows; row += StepSlide)
			{
				for (int col = 0; col <= LoadedImage.cols - windows_n_cols; col += StepSlide)
				{
					//INFORMACIÓN COORDENADAS DE CANDIDATO ACTUAL
					fprintf(Fichero, "%d\n", row); //Escribe coordenada en x
					fprintf(Fichero, "%d\n", col); //Escribe coordenada en y

					// resulting window   
					Rect windows(col, row, windows_n_cols, windows_n_rows);

					Mat DrawResultHere = LoadedImage.clone();

					// Draw only rectangle
					rectangle(DrawResultHere, windows, Scalar(255), 1, 8, 0);
					// Draw grid
					rectangle(DrawResultGrid, windows, Scalar(255), 1, 8, 0);

					// Show  rectangle
					namedWindow("Ventana en movimiento", WINDOW_AUTOSIZE);
					imshow("Ventana en movimiento", DrawResultHere);

					// Show grid
					//namedWindow("Barrido realizado", WINDOW_AUTOSIZE);
					//imshow("Barrido realizado", DrawResultGrid);
					imwrite(cv::format("Resultados/Barrido factor %d.JPG", Escala), DrawResultGrid);

					//Ver ROI
					Mat Roi = LoadedImage(windows);
					//namedWindow("Ventana segmentada", WINDOW_AUTOSIZE);
					//imshow("Ventana segmentada", Roi);
					waitKey(2);
					resize(Roi, Roi, Size(64, 128));
					imwrite(cv::format("C:\\Users\\user\\Desktop\\Recursos\\TecnicasBarrido\\Candidatos\\%d.JPG", NumeroNombre), Roi);
					NumeroNombre++;
					ContadorCandidatos++;
				}
			}
			fprintf(Fichero, "%c\n", '*'); //Indica el final de candidatos en una escala
		}
		fprintf(Fichero, "%c", '-'); //Indica el final de todo el proceso de barrido
	}
	fclose(Fichero);

	FILE *Fichero2;
	Fichero2 = fopen("C:\\Users\\user\\Desktop\\Recursos\\TecnicasBarrido\\DescripcionBarrido\\NumeroCandidatos.txt", "w");
	if (Fichero2 == NULL){
		printf("\nError al crear el archivo de texto NumeroCandidatos.txt... \n");
	}
	else{
		fprintf(Fichero2, "%d", (NumeroNombre - 1)); //Indica el final de todo el proceso de barrido
	}
}