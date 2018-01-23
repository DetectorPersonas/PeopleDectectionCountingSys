#include "opencv2\ximgproc.hpp"
#include "opencv2\highgui.hpp"
#include "opencv2\opencv.hpp"

using namespace std;
using namespace cv;
using namespace cv::ximgproc;


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

	folderName = "C:\\Users\\user\\Desktop\\Recursos\\TecnicasBarrido\\\BarridoBordes\\Resultados";
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

	//Lectura de la escena
	Mat LoadedImage, image_f, edges_canny, bw, gray;
	LoadedImage = imread("C:\\Users\\user\\Desktop\\Recursos\\BancoEscenas\\Otras\\Escena.jpg", IMREAD_COLOR);

	if (LoadedImage.empty())
	{
		printf("Cannot read image file");
		return -1;
	}

	//Creación de las ventanas para visualizar
	//namedWindow("RF edges", 1);
	//namedWindow("Canny edges", 1);

	//Extracción de bordes
	LoadedImage.convertTo(image_f, cv::DataType<float>::type, 1 / 255.0);
	cv::Mat edges(image_f.size(), image_f.type());
	Ptr<StructuredEdgeDetection> pDollar = createStructuredEdgeDetection("model.yml");
	pDollar->detectEdges(image_f, edges);
	//edges.convertTo(edges, CV_8U, 255.0);
	edges.convertTo(edges, CV_8U, 512.0);
	Mat edges_color;
	//applyColorMap(edges, edges_color, COLORMAP_JET);
	applyColorMap(edges, edges_color, COLORMAP_COOL);
	Canny(edges, bw, 100, 200, 3);
	threshold(edges, bw, 64, 255, CV_THRESH_BINARY);
	cvtColor(LoadedImage, gray, COLOR_BGR2GRAY);
	Canny(gray, edges_canny, 60, 180, 3);
	//cv::imshow("RF edges", edges_color);
	//cv::imshow("RF edges", bw);
	//cv::imshow("Canny edges", edges_canny);

	//CREACIÓN DEL FICHERO PARA BLOQUE POST-PROCESAMIENTO
	FILE *Fichero;
	Fichero = fopen("C:\\Users\\user\\Desktop\\Recursos\\TecnicasBarrido\\DescripcionBarrido\\DescripcionBarrido.txt", "w");
	if (Fichero == NULL){
		printf("\nError al crear el archivo de texto DescripcionBarridoBordes.txt... \n");
	}
	else{
		fprintf(Fichero, "%d\n", windows_n_cols); //Escribe el ancho de la ventana
		fprintf(Fichero, "%d\n", windows_n_rows); //Escribe el alto de la ventana
		fprintf(Fichero, "%c\n", '*'); //Indica el final info sobre ventana

		//Barrido completo de la escena
		for (int Factor = 0; Factor < 2; Factor++) {

			int Escala = pow(2, Factor);
			printf("Factor: %d \n", Escala);

			resize(bw, bw, Size(Real_Columnas / Escala, Real_Filas / Escala));
			//namedWindow(cv::format("Escala %d", Escala), WINDOW_AUTOSIZE);
			//imshow(cv::format("Escala %d", Escala), bw);
			imwrite(cv::format("Resultados/Factor %d.JPG", Escala), bw);
			waitKey(1000);

			Mat DrawResultGrid = bw.clone();

			for (int row = 0; row <= bw.rows - windows_n_rows; row += StepSlide)
			{
				for (int col = 0; col <= bw.cols - windows_n_cols; col += StepSlide)
				{
					// resulting window   
					Rect windows(col, row, windows_n_cols, windows_n_rows);

					Mat DrawResultHere = bw.clone();

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
					Mat Roi = bw(windows);
					//namedWindow("Ventana segmentada", WINDOW_AUTOSIZE);
					//imshow("Ventana segmentada", Roi);
					waitKey(1);
					int Suma = (int)cv::sum(Roi)[0];
					cout << Suma << endl;
					if (Suma > 920000){
						Mat RoiColor = LoadedImage(windows);
						resize(RoiColor, RoiColor, Size(64, 128));
						#//imwrite(cv::format("C:\\Users\\user\\Desktop\\Recursos\\TecnicasBarrido\\Candidatos\\%d-%d.JPG", NumeroNombre, Suma), RoiColor);
						imwrite(cv::format("C:\\Users\\user\\Desktop\\Recursos\\TecnicasBarrido\\Candidatos\\%d.JPG", NumeroNombre), RoiColor);
						NumeroNombre++;

						//INFORMACIÓN COORDENADAS DE CANDIDATO ACTUAL QUE CUMPLE LA REGLA
						fprintf(Fichero, "%d\n", row); //Escribe coordenada en x
						fprintf(Fichero, "%d\n", col); //Escribe coordenada en y
					}
				}
			}
			fprintf(Fichero, "%c\n", '*'); //Indica el final de candidatos en una escala
		}
		fprintf(Fichero, "%c\n", '-'); //Indica el final de todo el proceso de barrido
	}
	fclose(Fichero);

	FILE *Fichero2;
	Fichero2 = fopen("C:\\Users\\user\\Desktop\\Recursos\\TecnicasBarrido\\DescripcionBarrido\\NumeroCandidatos.txt", "w");
	if (Fichero2 == NULL){
		printf("\nError al crear el archivo de texto NumeroCandidatos.txt... \n");
	}
	else{
		fprintf(Fichero2, "%d", (NumeroNombre-1)); //Indica el final de todo el proceso de barrido
	}

}
