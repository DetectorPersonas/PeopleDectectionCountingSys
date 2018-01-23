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

//Definición de directorios para lectura de bases de datos y creación de ficheros
#define DIRECCION_IMAGENES_POSITIVAS "C:\\Users\\user\\Desktop\\Recursos\\BancosImagenes\\Personalizado\\Prueba\\Persona"
#define DIRECCION_IMAGENES_NEGATIVAS "C:\\Users\\user\\Desktop\\Recursos\\BancosImagenes\\Personalizado\\Prueba\\No_Persona"

//Parametros archivo Python entrenamiento SVM
#define DIRECCION_FICHERO_CARAC_TEST "/Users/user/Desktop/Recursos/ModelosClasificacion/Prueba/HoGSVM/CaracteristicasHOGPrueba.txt"
#define DIRECCION_FICHERO_LABELS_TEST "/Users/user/Desktop/Recursos/ModelosClasificacion/Prueba/HoGSVM/LabelsHOGPrueba.txt"
#define NUM_CARACTERISTICAS 3780
#define DIRECTORIO_CARGAR_MODELO "/Users/user/Desktop/Recursos/ModelosClasificacion/ModelosEntrenados/"
#define NOMBRE_MODELO "HOGSVM"
#define DIRECCION_PYTHON "C:/Users/user/Desktop/Recursos/ModelosClasificacion/Prueba/HoGSVM/testingHOG.py"

//Definición de parametros para el extractor de características HoG
#define	TAMANO_IMAGEN Size(64, 128) 
#define CELL_SIZE Size(8, 8)
#define BLOCK_SIZE Size(16, 16)
#define BLOCK_STRIDE Size(8, 8)
#define NUMBER_BINS 9

//Prototipo de las funciones utilizadas
void CargarImagenes(string Directorio, vector<Mat>& Imagenes);
vector<string> ArchivosEnDirectorio(string Directorio);
void CaracteristicasHoG(const vector< Mat > & Imagenes, vector< Mat > & Gradientes, const Size & Tamano);
Mat VisualizacionHoG(const Mat& color_origImg, vector<float>& descriptorValues, const Size & size);
void CreacionFicherosPython(vector< Mat > & Gradientes, vector< int > Etiquetas, string Tipo);
void LlamarPython();

int numImagenes;

int main(int argc, char** argv)
{
	vector< Mat > ImagenesPositivas;
	vector< Mat > ImagenesNegativas;
	vector< Mat > GradientesImagenes;
	vector< int > Etiquetas;

	cout << "INICIO PROCESO EXTRACCION DE CARACTERISTICAS PRUEBA..." << endl<<endl;
	
	cout << "1. Lectura imagenes positivas y asignacion de label (+1)" << endl;
	CargarImagenes(DIRECCION_IMAGENES_POSITIVAS, ImagenesPositivas);
	Etiquetas.assign(ImagenesPositivas.size(), +1);

	cout << "2. Lectura imagenes negativas y asignacion de label (-1)" << endl << endl;
	CargarImagenes(DIRECCION_IMAGENES_NEGATIVAS, ImagenesNegativas);
	Etiquetas.insert(Etiquetas.end(), ImagenesNegativas.size(), -1);

	cout << "3. Extraccion caracteristicas (HoG) a las imagenes positivas (visualizacion)" << endl;
	CaracteristicasHoG(ImagenesPositivas, GradientesImagenes, TAMANO_IMAGEN);

	cout << "4. Extraccion caracteristicas (HoG) a las imagenes negativas (visualizacion)" << endl;
	CaracteristicasHoG(ImagenesNegativas, GradientesImagenes, TAMANO_IMAGEN);
	
	cout << "5. Creacion ficheros con descriptores y labels" << endl;
	cout << "Tamaño matriz caracteristicas Filas(numero de imagenes): " << GradientesImagenes.size() << " y Columnas(numero de caracteristicas): " << GradientesImagenes[0].rows << endl << endl;
	
	numImagenes = GradientesImagenes.size();

	CreacionFicherosPython(GradientesImagenes, Etiquetas, "Prueba");

	LlamarPython();

	waitKey(1000000);

	return 0;
}

void CargarImagenes(string Directorio, vector<Mat>& Imagenes)
{
	Mat imagen;
	vector<string> NombresArchivos;

	NombresArchivos = ArchivosEnDirectorio(Directorio);

	for (int i = 0; i < NombresArchivos.size(); ++i) {
		imagen = imread(NombresArchivos.at(i));
		if (imagen.empty()){
			cout << "ERROR: No se pudo leer la imagen " << i + 1 << " del directorio." << endl;
			continue;
		}
		imshow("Imagen cargada", imagen);
		resize(imagen, imagen, TAMANO_IMAGEN);
		cout << "Se cargo la imagen" << i + 1 << ". " << endl;
		imshow("imagen modificada", imagen);
		waitKey(10);
		Imagenes.push_back(imagen.clone());
	}
	cout << endl;
}

vector<string> ArchivosEnDirectorio(string Directorio)
{
	vector<string> NombresArchivos;
	char buf[256];
	string command;

	command = "dir /b /s " + Directorio;

	FILE* pipe = NULL;

	if (pipe = _popen(command.c_str(), "rt")){
		while (!feof(pipe)){
			if (fgets(buf, 256, pipe) != NULL) {
				string Archivo(buf);
				Archivo.pop_back();
				NombresArchivos.push_back(Archivo);
			}
		}
	}

	_pclose(pipe);

	return NombresArchivos;
}

void CaracteristicasHoG(const vector< Mat > & Imagenes, vector< Mat > & Gradientes, const Size & Tamano)
{
	HOGDescriptor hog(Tamano, BLOCK_SIZE, BLOCK_STRIDE, CELL_SIZE, NUMBER_BINS);

	Mat EscalaGrises;

	//vector< Point > location;
	vector< float > Descriptores;

	vector< Mat >::const_iterator img = Imagenes.begin();
	vector< Mat >::const_iterator end = Imagenes.end();

	for (; img != end; ++img)
	{
		cvtColor(*img, EscalaGrises, COLOR_BGR2GRAY);
		hog.compute(EscalaGrises, Descriptores, Size(0, 0), Size(0, 0));
		Gradientes.push_back(Mat(Descriptores).clone());
		imshow("Gradientes HoG", VisualizacionHoG(img->clone(), Descriptores, Tamano));
		waitKey(10);
	}

	cout << "Cantidad de caracteristicas por imagen: " << Descriptores.size() << endl <<endl;
}

//Tomado de: http://www.juergenwiki.de/work/wiki/doku.php?id=public:hog_descriptor_computation_and_visualization
Mat VisualizacionHoG(const Mat& color_origImg, vector<float>& descriptorValues, const Size & size)
{
	const int DIMX = size.width; //Dimensión de la imagen de entrada
	const int DIMY = size.height;
	float zoomFac = 5; //Factor para el zoom de la imagen de entrada
	Mat visu;
	resize(color_origImg, visu, Size((int)(color_origImg.cols*zoomFac), (int)(color_origImg.rows*zoomFac))); //	Aumento de la imagen de entrada según el factor de zoom

	int cellSize = CELL_SIZE.width; //Tamaño de la celda cuadrada
	int gradientBinSize = NUMBER_BINS; //Tamaño de bins del histograma
	float radRangeForOneBin = (float)(CV_PI / (float)gradientBinSize); // dividing 180° into 9 bins, how large (in rad) is one bin?

	//Cálculo de las celdas que tendrá la imagen en ambas orientaciones según la dimensión de la imagen y las celdas
	int cells_in_x_dir = DIMX / cellSize;
	int cells_in_y_dir = DIMY / cellSize;

	// prepare data structure: 9 orientation / gradient strenghts for each cell
	float*** gradientStrengths = new float**[cells_in_y_dir];
	int** cellUpdateCounter = new int*[cells_in_y_dir];
	for (int y = 0; y<cells_in_y_dir; y++)
	{
		gradientStrengths[y] = new float*[cells_in_x_dir];
		cellUpdateCounter[y] = new int[cells_in_x_dir];
		for (int x = 0; x<cells_in_x_dir; x++)
		{
			gradientStrengths[y][x] = new float[gradientBinSize];
			cellUpdateCounter[y][x] = 0;

			for (int bin = 0; bin<gradientBinSize; bin++)
				gradientStrengths[y][x][bin] = 0.0;
		}
	}

	// nr of blocks = nr of cells - 1
	// since there is a new block on each cell (overlapping blocks!) but the last one
	int blocks_in_x_dir = (DIMX / BLOCK_STRIDE.width) - 1;
	int blocks_in_y_dir = (DIMY / BLOCK_STRIDE.height) - 1;

	/*int blocks_in_x_dir = cells_in_x_dir - 1;
	int blocks_in_y_dir = cells_in_y_dir - 1;*/

	// compute gradient strengths per cell
	int descriptorDataIdx = 0;
	int cellx = 0;
	int celly = 0;

	for (int blockx = 0; blockx<blocks_in_x_dir; blockx++)
	{
		for (int blocky = 0; blocky<blocks_in_y_dir; blocky++)
		{
			// 4 cells per block ...
			for (int cellNr = 0; cellNr<4; cellNr++)
			{
				// compute corresponding cell nr
				cellx = blockx;
				celly = blocky;
				if (cellNr == 1) celly++;
				if (cellNr == 2) cellx++;
				if (cellNr == 3)
				{
					cellx++;
					celly++;
				}

				for (int bin = 0; bin<gradientBinSize; bin++)
				{
					float gradientStrength = descriptorValues[descriptorDataIdx];
					descriptorDataIdx++;

					gradientStrengths[celly][cellx][bin] += gradientStrength;

				} // for (all bins)


				// note: overlapping blocks lead to multiple updates of this sum!
				// we therefore keep track how often a cell was updated,
				// to compute average gradient strengths
				cellUpdateCounter[celly][cellx]++;

			} // for (all cells)


		} // for (all block x pos)
	} // for (all block y pos)


	// compute average gradient strengths
	for (celly = 0; celly<cells_in_y_dir; celly++)
	{
		for (cellx = 0; cellx<cells_in_x_dir; cellx++)
		{

			float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];

			// compute average gradient strenghts for each gradient bin direction
			for (int bin = 0; bin<gradientBinSize; bin++)
			{
				gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
			}
		}
	}

	// draw cells
	for (celly = 0; celly<cells_in_y_dir; celly++)
	{
		for (cellx = 0; cellx<cells_in_x_dir; cellx++)
		{
			//Coordenadas para dibujar por cada celda
			int drawX = cellx * cellSize;
			int drawY = celly * cellSize;

			//Coordenadas para dibujar vectores en el medio de cada celda
			int mx = drawX + cellSize / 2;
			int my = drawY + cellSize / 2;

			//Función para dibujar el rectángulo negro de cada celda, de modo tal que se vea la grilla según la división de celdas
			rectangle(visu, Point((int)(drawX*zoomFac), (int)(drawY*zoomFac)), Point((int)((drawX + cellSize)*zoomFac), (int)((drawY + cellSize)*zoomFac)), Scalar(100, 100, 100), 1);

			// draw in each cell all 9 gradient strengths
			for (int bin = 0; bin<gradientBinSize; bin++)
			{
				float currentGradStrength = gradientStrengths[celly][cellx][bin];

				// no line to draw?
				if (currentGradStrength == 0)
					continue;

				float currRad = bin * radRangeForOneBin + radRangeForOneBin / 2;

				float dirVecX = cos(currRad);
				float dirVecY = sin(currRad);
				float maxVecLen = (float)(cellSize / 2.f);
				float scale = 2.5; // just a visualization scale, to see the lines better

				// compute line coordinates
				float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
				float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
				float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
				float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

				// draw gradient visualization
				line(visu, Point((int)(x1*zoomFac), (int)(y1*zoomFac)), Point((int)(x2*zoomFac), (int)(y2*zoomFac)), Scalar(0, 255, 0), 1);

			} // for (all bins)

		} // for (cellx)
	} // for (celly)


	// don't forget to free memory allocated by helper data structures!
	for (int y = 0; y<cells_in_y_dir; y++)
	{
		for (int x = 0; x<cells_in_x_dir; x++)
		{
			delete[] gradientStrengths[y][x];
		}
		delete[] gradientStrengths[y];
		delete[] cellUpdateCounter[y];
	}
	delete[] gradientStrengths;
	delete[] cellUpdateCounter;

	return visu;

}

void CreacionFicherosPython(vector< Mat > & Gradientes, vector< int > Etiquetas, string Tipo)
{
	//Creación fichero para caracteristicas
	FILE *Fichero;

	string Nombre = "CaracteristicasHOG";
	string Extension = ".txt";
	Nombre += Tipo;
	Nombre += Extension;

	Fichero = fopen(Nombre.c_str(), "w");

	if (Fichero == NULL){
		printf("\nError al crear el archivo de texto CaracteristicasHOG.txt... \n");
	}
	else{
		for (int i = 0; i<Gradientes.size(); i++){
			for (int j = 0; j<Gradientes[0].rows; j++){
				fprintf(Fichero, "%f\n", Gradientes[i].at<float>(j, 0));
			}
		}
	}
	fclose(Fichero);
	
	//Creación fichero para labels

	Nombre = "LabelsHOG";
	Nombre += Tipo;
	Nombre += Extension;

	Fichero = fopen(Nombre.c_str(), "w");
	if (Fichero == NULL){
		printf("\nError al crear el archivo de texto LabelsHOG.txt... \n");
	}
	else{
		for (int i = 0; i < Etiquetas.size(); i++){
			fprintf(Fichero, "%d\n", Etiquetas.at(i));
		}
	}
	fclose(Fichero);
}

void LlamarPython()
{
	string filename = DIRECCION_PYTHON;
	string command = "python ";
	command += filename;

	string dirCTr = " --tcar " ;
	string dirLTr = " --tlab " ;
	string numTr = " --tnimag ";
	string numCa = " --ncha ";
	string dirSVM = " --dirsvm ";
	string nameSVM = " --namemodelSVM ";

	dirCTr += DIRECCION_FICHERO_CARAC_TEST;
	command += dirCTr;
	dirLTr += DIRECCION_FICHERO_LABELS_TEST;
	command += dirLTr;
	numTr += to_string(numImagenes);
	command += numTr;
	numCa += to_string(NUM_CARACTERISTICAS);
	command += numCa;
	dirSVM += DIRECTORIO_CARGAR_MODELO;
	command += dirSVM;

	string nombre = NOMBRE_MODELO;
	string BancoImag = "Personalizada.pkl";
	nombre += BancoImag;
	nameSVM += nombre;
	command += nameSVM;

	system(command.c_str());
	Sleep(1000000);
}