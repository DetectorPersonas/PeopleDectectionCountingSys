#include "detector_personas.h"
#include "ui_detector_personas.h"

#include<QFileDialog>
#include<QtCore>
#include <QMessageBox>

using namespace cv::ximgproc;

//Definición de directorios y constantes
#define DIR_BANCO_IMAG "C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/BancosImagenes/"
#define DIRECTORIO_GUARDAR_MODELO "C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/ModelosClasificacion/ModelosEntrenados/"
#define DIR_BANCOS_LECTURA "C:\\Deteccion_Conteo_Personas_Escenas_PUJ-1703\\Recursos\\BancosImagenes\\"
#define DIR_GUARDAR_CARAC_LABELS "C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/ModelosClasificacion/"
#define PATH_CANDIDATOS "C:\\Deteccion_Conteo_Personas_Escenas_PUJ-1703\\Recursos\\TecnicasBarrido\\Candidatos"
#define PATH_DESCRIPCIONBARRIDOS "C:\\Deteccion_Conteo_Personas_Escenas_PUJ-1703\\Recursos\\TecnicasBarrido\\DescripcionBarrido\\"
#define PATH_RESULTADOSEXHAUS "C:\\Deteccion_Conteo_Personas_Escenas_PUJ-1703\\Recursos\\TecnicasBarrido\\BarridoVentana\\Resultados"
#define PATH_RESULTADOSBORDES "C:\\Deteccion_Conteo_Personas_Escenas_PUJ-1703\\Recursos\\TecnicasBarrido\\BarridoBordes\\Resultados"

//Definición de parametros para el extractor de características HoG
#define	TAMANO_IMAGEN cv::Size(64, 128)
#define CELL_SIZE cv::Size(8, 8)
#define BLOCK_SIZE cv::Size(16, 16)
#define BLOCK_STRIDE cv::Size(8, 8)
#define NUMBER_BINS 9
#define NUM_CARACTERISTICAS 3780

//Definición variables globales
cv::Mat EjemPos;
cv::Mat EjemNeg;
cv::Mat frameCargado;
QString strFileName;
int tipoBarrido;

Detector_Personas::Detector_Personas(QWidget *parent) : QMainWindow(parent), ui(new Ui::Detector_Personas){

    ui->setupUi(this);

    ui->ListaModelos->addItem("HoG - Random Forest");
    ui->ListaModelos->addItem("HoG - SVM");
    ui->ListaModelos->addItem("SIFT - FV - Random Forest");
    ui->ListaModelos->addItem("SIFT - FV - SVM");
    ui->ListaModelos->addItem("ANN - 2 hidden layer");
    ui->ListaModelos->addItem("CNN - Inception");

    ui->ListaBancos->addItem("Daimler");
    ui->ListaBancos->addItem("INRIA");
    ui->ListaBancos->addItem("MIT");
    ui->ListaBancos->addItem("Oclusion");
    ui->ListaBancos->addItem("PUJ");
    ui->ListaBancos->addItem("Personalizado");

    ui->ListaBarridos->addItem("Método exhaustivo");
    ui->ListaBarridos->addItem("Método de bordes");

    ui->ListaEntrenamientos->addItem("Daimler");
    ui->ListaEntrenamientos->addItem("INRIA");
    ui->ListaEntrenamientos->addItem("MIT");
    ui->ListaEntrenamientos->addItem("Oclusion");
    ui->ListaEntrenamientos->addItem("PUJ");
    ui->ListaEntrenamientos->addItem("Personalizado");

    ui->ListaModelo->addItem("HoG - Random Forest");
    ui->ListaModelo->addItem("HoG - SVM");
    ui->ListaModelo->addItem("SIFT - FV - Random Forest");
    ui->ListaModelo->addItem("SIFT - FV - SVM");
    ui->ListaModelo->addItem("ANN - 2 hidden layer");
    ui->ListaModelo->addItem("CNN - Inception");

    ui->modelos->addItem("HoG - Random Forest");
    ui->modelos->addItem("HoG - SVM");
    ui->modelos->addItem("SIFT - FV - Random Forest");
    ui->modelos->addItem("SIFT - FV - SVM");
    ui->modelos->addItem("ANN - 2 hidden layer");
    ui->modelos->addItem("CNN - Inception");

    ui->bancosTrain->addItem("Daimler");
    ui->bancosTrain->addItem("INRIA");
    ui->bancosTrain->addItem("MIT");
    ui->bancosTrain->addItem("Oclusion");
    ui->bancosTrain->addItem("PUJ");
    ui->bancosTrain->addItem("Personalizado");

    ui->bancosTest->addItem("Daimler");
    ui->bancosTest->addItem("INRIA");
    ui->bancosTest->addItem("MIT");
    ui->bancosTest->addItem("Oclusion");
    ui->bancosTest->addItem("PUJ");
    ui->bancosTest->addItem("Personalizado");

    ui->condicional->addItem("Si");
    ui->condicional->addItem("No");

    ui->iniciarBusqueda->setStyleSheet("background-color: rgb(255,125,100)");
    ui->btnEntrenar->setStyleSheet("background-color: rgb(255,125,100)");
    ui->iniciarReporte->setStyleSheet("background-color: rgb(200,255,100)");
    ui->iniciarTest->setStyleSheet("background-color: rgb(255,125,100)");
}

Detector_Personas::~Detector_Personas(){
    delete ui;
}

QImage Detector_Personas::convertOpenCVMatToQtQImage(cv::Mat mat) {

    if(mat.channels() == 1) {                   // if grayscale image
        return QImage((uchar*)mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Indexed8);     // declare and return a QImage
    } else if(mat.channels() == 3) {            // if 3 channel color image
        cv::cvtColor(mat, mat, CV_BGR2RGB);     // invert BGR to RGB
        return QImage((uchar*)mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);       // declare and return a QImage
    } else {
        qDebug() << "in convertOpenCVMatToQtQImage, image was not 1 channel or 3 channel, should never get here";
    }
    return QImage();        // return a blank QImage if the above did not work
}

std::vector<std::string> Detector_Personas::ArchivosEnDirectorio(std::string Directorio){
    std::vector<std::string> NombresArchivos;
    char buf[256];
    std::string command;

    command = "dir /b /s ";
    command += Directorio;

    FILE* pipe = NULL;

    if (pipe = _popen(command.c_str(), "rt")){
       while (!feof(pipe)){
            if (fgets(buf, 256, pipe) != NULL) {
                std::string Archivo(buf);
                Archivo.pop_back();
                NombresArchivos.push_back(Archivo);
            }
        }

    }

    _pclose(pipe);

    return NombresArchivos;

}

void Detector_Personas::CargarImagenes(std::string Directorio, std::vector<cv::Mat>& Imagenes, int Pos, int Prueba)
{
    cv::Mat imagen;
    std::vector<std::string> NombresArchivos;

    NombresArchivos = Detector_Personas::ArchivosEnDirectorio(Directorio);

    int ejemplos;

    if(Prueba==0){
        if(Pos==1){
            ejemplos = (ui->numPos->toPlainText()).toInt();
        }
        else{
            ejemplos = (ui->numNeg->toPlainText()).toInt();
        }
    }
    else{
        if(Pos==1){
            ejemplos = (ui->posTest->toPlainText()).toInt();
        }
        else{
            ejemplos = (ui->negTest->toPlainText()).toInt();
        }
    }

    for (int i = 0; i < ejemplos ; ++i) {
        imagen = cv::imread(NombresArchivos.at(i));
        if (imagen.empty()){
            std::cout << "ERROR: No se pudo leer la imagen " << i + 1 << " del directorio.\n";
            continue;
        }
        //imshow("Imagen cargada", imagen);
        cv::resize(imagen, imagen, TAMANO_IMAGEN);
        std::cout << "Se cargo la imagen" << i + 1 << ". \n";
        //imshow("imagen modificada", imagen);
        Imagenes.push_back(imagen.clone());
    }
    std::cout << "\n";

}

cv::Mat Detector_Personas::VisualizacionHoG(const cv::Mat& color_origImg, std::vector<float>& descriptorValues, const cv::Size & size){ //Tomado de: http://www.juergenwiki.de/work/wiki/doku.php?id=public:hog_descriptor_computation_and_visualization
    const int DIMX = size.width; //Dimensión de la imagen de entrada
    const int DIMY = size.height;
    float zoomFac = 2; //Factor para el zoom de la imagen de entrada

    cv::Mat visu;
    cv::resize(color_origImg, visu, cv::Size((int)(color_origImg.cols*zoomFac), (int)(color_origImg.rows*zoomFac))); //	Aumento de la imagen de entrada según el factor de zoom

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
            cv::rectangle(visu, cv::Point((int)(drawX*zoomFac), (int)(drawY*zoomFac)), cv::Point((int)((drawX + cellSize)*zoomFac), (int)((drawY + cellSize)*zoomFac)), cv::Scalar(100, 100, 100), 1);

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
                line(visu, cv::Point((int)(x1*zoomFac), (int)(y1*zoomFac)), cv::Point((int)(x2*zoomFac), (int)(y2*zoomFac)), cv::Scalar(0, 255, 0), 1);

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

void Detector_Personas::CaracteristicasHoG(const std::vector< cv::Mat > & Imagenes, std::vector< cv::Mat > & Gradientes, const cv::Size & Tamano){
    cv::HOGDescriptor hog(Tamano, BLOCK_SIZE, BLOCK_STRIDE, CELL_SIZE, NUMBER_BINS);

    cv::Mat EscalaGrises;

    //vector< Point > location;
    std::vector< float > Descriptores;

    std::vector< cv::Mat >::const_iterator img = Imagenes.begin();
    std::vector< cv::Mat >::const_iterator end = Imagenes.end();

    for (; img != end; ++img)
    {
        cv::cvtColor(*img, EscalaGrises, cv::COLOR_BGR2GRAY);
        hog.compute(EscalaGrises, Descriptores, cv::Size(0, 0), cv::Size(0, 0));
        Gradientes.push_back(cv::Mat(Descriptores).clone());
        //imshow("Gradientes HoG", VisualizacionHoG(img->clone(), Descriptores, Tamano));
    }

    std::cout << "Cantidad de caracteristicas por imagen: " << Descriptores.size() << "\n \n";
}

void Detector_Personas::CreacionFicherosPython(std::vector< cv::Mat > & Gradientes, std::vector< int > Etiquetas, std::string Tipo, std::string Classifier){
    //Creación fichero para caracteristicas
    FILE *Fichero;

    std::string dir = DIR_GUARDAR_CARAC_LABELS;
    dir += Tipo;
    dir += "/";
    dir += Classifier;
    std::string Nombre = "/CaracteristicasHOG";
    std::string Extension = ".txt";
    Nombre += Tipo;
    Nombre += Extension;
    dir += Nombre;

    Fichero = fopen(dir.c_str(), "w");

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

    dir = DIR_GUARDAR_CARAC_LABELS;
    dir += Tipo;
    dir += "/";
    dir += Classifier;
    Nombre = "/LabelsHOG";
    Nombre += Tipo;
    Nombre += Extension;
    dir += Nombre;

    Fichero = fopen(dir.c_str(), "w");
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

void Detector_Personas::LlamarPythonHOGRF(int numImagenes, std::string Tipo, std::string Classifier){

    std::string filename = DIR_GUARDAR_CARAC_LABELS;
    std::string command = "python ";
    command += filename;
    command += Tipo;
    command += "/";
    command += Classifier;
    command += "/";

    if(strcmp(Tipo.c_str(),"Entrenamiento")==0){
        command += "trainingHOGRF.py";

        std::string dirCTr = " --car " ;
        std::string dirLTr = " --lab " ;
        std::string numTr = " --nimag ";
        std::string numCa = " --ncha ";
        std::string dirRF = " --dirsvm ";
        std::string nameRF = " --namemodelRF ";
        std::string numTrees = " --narb ";

        std::string dir = DIR_GUARDAR_CARAC_LABELS;
        dir += Tipo;
        dir += "/";
        dir += Classifier;
        std::string Nombre = "/CaracteristicasHOG";
        std::string Extension = ".txt";
        Nombre += Tipo;
        Nombre += Extension;
        dir += Nombre;

        dirCTr += dir;
        command += dirCTr;

        dir = DIR_GUARDAR_CARAC_LABELS;
        dir += Tipo;
        dir += "/";
        dir += Classifier;
        Nombre = "/LabelsHOG";
        Nombre += Tipo;
        Nombre += Extension;
        dir += Nombre;

        dirLTr += dir;
        command += dirLTr;

        numTr += std::to_string(numImagenes);
        command += numTr;

        numCa += std::to_string(NUM_CARACTERISTICAS);
        command += numCa;

        dirRF += DIRECTORIO_GUARDAR_MODELO;
        command += dirRF;

        std::string nombre = Classifier;
        std::string BancoImag = (ui->ListaBancos->currentText()).toStdString();
        nombre += BancoImag;
        nombre += ".pkl";
        nameRF += nombre;
        command += nameRF;

        numTrees += (ui->numTrees->toPlainText()).toStdString();
        command += numTrees;

        system(command.c_str());
    }
    else if(strcmp(Tipo.c_str(),"Prueba")==0){
        command += "testingHOG.py";

        std::string dirCTr = " --tcar " ;
        std::string dirLTr = " --tlab " ;
        std::string numTr = " --tnimag ";
        std::string numCa = " --ncha ";
        std::string dirRF = " --dirrf ";
        std::string nameRF = " --namemodelRF ";

        std::string dir = DIR_GUARDAR_CARAC_LABELS;
        dir += Tipo;
        dir += "/";
        dir += Classifier;
        std::string Nombre = "/CaracteristicasHOG";
        std::string Extension = ".txt";
        Nombre += Tipo;
        Nombre += Extension;
        dir += Nombre;

        dirCTr += dir;
        command += dirCTr;

        dir = DIR_GUARDAR_CARAC_LABELS;
        dir += Tipo;
        dir += "/";
        dir += Classifier;
        Nombre = "/LabelsHOG";
        Nombre += Tipo;
        Nombre += Extension;
        dir += Nombre;

        dirLTr += dir;
        command += dirLTr;

        numTr += std::to_string(numImagenes);
        command += numTr;

        numCa += std::to_string(NUM_CARACTERISTICAS);
        command += numCa;

        dirRF += DIRECTORIO_GUARDAR_MODELO;
        command += dirRF;

        std::string nombre = Classifier;
        std::string BancoImag = (ui->bancosTrain->currentText()).toStdString();
        nombre += BancoImag;
        nombre += ".pkl";
        nameRF += nombre;
        command += nameRF;

        system(command.c_str());
    }
}

void Detector_Personas::LlamarPythonHOGSVM(int numImagenes, std::string Tipo, std::string Classifier){

    FILE* F;
    F = fopen("HOGSVMTrain.txt","w");

    std::string filename = DIR_GUARDAR_CARAC_LABELS;
    std::string command = "python ";
    command += filename;
    command += Tipo;
    command += "/";
    command += Classifier;
    command += "/";

    if(strcmp(Tipo.c_str(),"Entrenamiento")==0){
        command += "trainingHOGSVM.py";
        std::string dirCTr = " --car " ;
        std::string dirLTr = " --lab " ;
        std::string numTr = " --nimag ";
        std::string numCa = " --ncha ";
        std::string dirRF = " --dirsvm ";
        std::string nameRF = " --namemodelSVM ";

        std::string dir = DIR_GUARDAR_CARAC_LABELS;
        dir += Tipo;
        dir += "/";
        dir += Classifier;
        std::string Nombre = "/CaracteristicasHOG";
        std::string Extension = ".txt";
        Nombre += Tipo;
        Nombre += Extension;
        dir += Nombre;

        dirCTr += dir;
        command += dirCTr;

        dir = DIR_GUARDAR_CARAC_LABELS;
        dir += Tipo;
        dir += "/";
        dir += Classifier;
        Nombre = "/LabelsHOG";
        Nombre += Tipo;
        Nombre += Extension;
        dir += Nombre;

        dirLTr += dir;
        command += dirLTr;

        numTr += std::to_string(numImagenes);
        command += numTr;

        numCa += std::to_string(NUM_CARACTERISTICAS);
        command += numCa;

        dirRF += DIRECTORIO_GUARDAR_MODELO;
        command += dirRF;

        std::string nombre = Classifier;
        std::string BancoImag = (ui->ListaBancos->currentText()).toStdString();
        nombre += BancoImag;
        nombre += ".pkl";
        nameRF += nombre;
        command += nameRF;

        fprintf(F,"%s \n",command.c_str());
        fclose(F);

        system(command.c_str());
    }
    else if(strcmp(Tipo.c_str(),"Prueba")==0){

        command += "testingHOG.py";

        std::string dirCTr = " --tcar " ;
        std::string dirLTr = " --tlab " ;
        std::string numTr = " --tnimag ";
        std::string numCa = " --ncha ";
        std::string dirRF = " --dirsvm ";
        std::string nameRF = " --namemodelSVM ";

        std::string dir = DIR_GUARDAR_CARAC_LABELS;
        dir += Tipo;
        dir += "/";
        dir += Classifier;
        std::string Nombre = "/CaracteristicasHOG";
        std::string Extension = ".txt";
        Nombre += Tipo;
        Nombre += Extension;
        dir += Nombre;

        dirCTr += dir;
        command += dirCTr;

        dir = DIR_GUARDAR_CARAC_LABELS;
        dir += Tipo;
        dir += "/";
        dir += Classifier;
        Nombre = "/LabelsHOG";
        Nombre += Tipo;
        Nombre += Extension;
        dir += Nombre;

        dirLTr += dir;
        command += dirLTr;

        numTr += std::to_string(numImagenes);
        command += numTr;

        numCa += std::to_string(NUM_CARACTERISTICAS);
        command += numCa;

        dirRF += DIRECTORIO_GUARDAR_MODELO;
        command += dirRF;

        std::string nombre = Classifier;
        std::string BancoImag = (ui->bancosTrain->currentText()).toStdString();
        nombre += BancoImag;
        nombre += ".pkl";
        nameRF += nombre;
        command += nameRF;

        system(command.c_str());
    }
}

void Detector_Personas::LlamarPythonSIFTFVRF(std::string Tipo, std::string Classifier){

    FILE* F;
    F =fopen("SIFTRFTrain.txt","w");

    std::string filename = DIR_GUARDAR_CARAC_LABELS;
    std::string command = "python ";
    command += filename;
    command += Tipo;
    command += "/";
    command += Classifier;
    command += "/";

    if(strcmp(Tipo.c_str(),"Entrenamiento")==0){
        command += "trainingSIFTFVRF.py";
        std::string dirImagPos = " --dirPos ";
        std::string dirImagNeg = " --dirNeg ";
        std::string numClus = " --clus ";
        std::string dirGMM = " --dirgmm ";
        std::string dirRF = " --dirrf ";
        std::string nameRF = " --namemodelRF ";
        std::string numImagPos = " --nlimPos ";
        std::string numImagNeg = " --nlimNeg ";
        std::string numArb = " --narb ";
        std::string version = " --verdic ";

        std::string dirPos = DIR_BANCO_IMAG;
        dirPos += (ui->ListaBancos->currentText()).toStdString();
        dirPos += "/";
        dirPos += Tipo;
        dirPos += "/Persona";
        dirImagPos += dirPos;
        command += dirImagPos;

        std::string dirNeg = DIR_BANCO_IMAG;
        dirNeg += (ui->ListaBancos->currentText()).toStdString();
        dirNeg += "/";
        dirNeg += Tipo;
        dirNeg += "/No_Persona";
        dirImagNeg += dirNeg;
        command += dirImagNeg;

        numClus +=(ui->numClus->toPlainText()).toStdString();
        command += numClus;

        std::string dirGuardarGMM = DIRECTORIO_GUARDAR_MODELO;
        dirGuardarGMM += "DiccionarioGMM/";
        dirGMM += dirGuardarGMM;
        command += dirGMM;

        dirRF += DIRECTORIO_GUARDAR_MODELO;
        command += dirRF;

        std::string nombre = Classifier;
        std::string BancoImag = (ui->ListaBancos->currentText()).toStdString();
        nombre += BancoImag;
        nombre += ".pkl";
        nameRF += nombre;
        command += nameRF;

        numImagPos += (ui->numPos->toPlainText()).toStdString();
        command += numImagPos;

        numImagNeg += (ui->numNeg->toPlainText()).toStdString();
        command += numImagNeg;

        numArb += (ui->numTrees->toPlainText()).toStdString();
        command += numArb;

        version += (ui->ListaBancos->currentText()).toStdString();
        command += version;

        fprintf(F,"%s \n",command.c_str());
        fclose(F);

        system(command.c_str());
    }
    else if(strcmp(Tipo.c_str(),"Prueba")==0){
        command += "testingSIFT.py";

        std::string dirImagPos = " --dir2 ";
        std::string dirImagNeg = " --dir3 ";
        std::string dirMod = " --dir ";
        std::string nameSVM = " --name ";
        std::string version = " --version ";
        std::string dirDic = " --dirDic ";
        std::string numImagPos = " --nlimPos ";
        std::string numImagNeg = " --nlimNeg ";

        std::string dirPos = DIR_BANCO_IMAG;
        dirPos += (ui->bancosTest->currentText()).toStdString();
        dirPos += "/";
        dirPos += Tipo;
        dirPos += "/Persona";
        dirImagPos += dirPos;
        command += dirImagPos;

        std::string dirNeg = DIR_BANCO_IMAG;
        dirNeg += (ui->bancosTest->currentText()).toStdString();
        dirNeg += "/";
        dirNeg += Tipo;
        dirNeg += "/No_Persona";
        dirImagNeg += dirNeg;
        command += dirImagNeg;

        numImagPos += (ui->posTest->toPlainText()).toStdString();
        command += numImagPos;

        numImagNeg += (ui->negTest->toPlainText()).toStdString();
        command += numImagNeg;

        dirMod +=DIRECTORIO_GUARDAR_MODELO;
        command += dirMod;

        std::string dirGuardarGMM = DIRECTORIO_GUARDAR_MODELO;
        dirGuardarGMM += "DiccionarioGMM/";
        dirDic += dirGuardarGMM;
        command += dirDic;

        nameSVM += Classifier;
        command += nameSVM;

        version += (ui->bancosTrain->currentText()).toStdString();
        command += version;

        system(command.c_str());
    }

}

void Detector_Personas::LlamarPythonSIFTFVRFEscena(std::string Tipo, std::string Classifier, int Visua, int Barrido){
    FILE* F;
    F=fopen("SIFTRFEscena.txt","w");

    std::string filename = DIR_GUARDAR_CARAC_LABELS;
    std::string command = "python ";
    command += filename;
    command += Tipo;
    command += "/";
    command += Classifier;
    command += "/";
    command += "testingEscenaSIFTFVRF.py";

    std::string dirModelo = " --dir ";
    std::string dirCandBordes = " --dirbordes ";
    std::string dirCandExhaus = " --direx ";
    std::string versionModelo = " --version ";
    std::string tipoVentaneo = " --tipovent ";
    std::string descripBordes = " --tdesbordes ";
    std::string descripExhaus = " --tdesex ";
    std::string graficar = " --visualize ";
    std::string Threshold = " --thereshold ";
    std::string frame = " --frame ";
    std::string groundTruth = " --ground ";
    std::string dirGMM = " --dirGMM ";
    std::string tag = " --tag ";
    std::string diferencia = " --diferencia ";
    std::string ancho = " --ancho ";
    std::string alto = " --alto ";

    std::string critConteo = " --critConteo ";
    std::string critScore = " --critPredic ";
    std::string scale = " --escala ";

    scale += (ui->factorVentana->toPlainText()).toStdString();
    command += scale;

    critConteo += (ui->criterioCont->toPlainText()).toStdString();
    command += critConteo;

    critScore += (ui->criterioPredic->toPlainText()).toStdString();
    command += critScore;

    std::string nombre = Classifier;
    std::string BancoImag = (ui->ListaEntrenamientos->currentText()).toStdString();
    nombre += BancoImag;
    versionModelo += nombre;
    versionModelo += ".pkl";
    command += versionModelo;

    dirModelo += DIRECTORIO_GUARDAR_MODELO;
    command += dirModelo;
    dirCandBordes += PATH_CANDIDATOS;
    command += dirCandBordes;
    dirCandExhaus += PATH_CANDIDATOS;
    command += dirCandExhaus;
    graficar += (QString::number(Visua)).toStdString();
    command += graficar;
    tipoVentaneo += (QString::number(Barrido)).toStdString();
    command += tipoVentaneo;
    descripBordes += PATH_DESCRIPCIONBARRIDOS;
    descripBordes += "DescripcionBarrido.txt";
    command += descripBordes;
    descripExhaus += PATH_DESCRIPCIONBARRIDOS;
    descripExhaus += "DescripcionBarrido.txt";
    command += descripExhaus;
    Threshold += (ui->criterioNMS->toPlainText()).toStdString();
    command += Threshold;
    diferencia += (ui->criterioNMS->toPlainText()).toStdString();
    command += diferencia;
    tag += (ui->ListaEntrenamientos->currentText()).toStdString();
    command += tag;
    frame += strFileName.toStdString();
    command += frame;

    ancho += (ui->anchoEscena->toPlainText()).toStdString();
    command += ancho;

    alto += (ui->altoEscena->toPlainText()).toStdString();
    command += alto;

    std::size_t pos = (strFileName.toStdString()).find("BancoEscenas");
    std::string str3 = (strFileName.toStdString()).substr (pos);     // Otras/Escena.pjg
    std::size_t pos2 = (str3).find("/");
    str3 = (str3).substr (pos2+1);     // Otras/1.jpg
    std::size_t pos3 = (str3).find("/");
    str3 = (str3).substr (pos3+1);     // 1.jpg
    std::size_t pos4 = (str3).find(".");

    std::string dirPath = (strFileName.toStdString()).substr (0,pos+pos2+pos3+2);
    std::string nom = (str3).substr (0,pos4); //1

    dirPath += nom;
    dirPath += ".txt";

    groundTruth += dirPath;
    command += groundTruth;

    dirGMM += DIRECTORIO_GUARDAR_MODELO;
    dirGMM += "DiccionarioGMM/";
    command += dirGMM;

    fprintf(F,"%s \n",command.c_str());
    fclose(F);

    system(command.c_str());

}

void Detector_Personas::LlamarPythonSIFTFVSVMEscena(std::string Tipo, std::string Classifier, int Visua, int Barrido){

    std::string filename = DIR_GUARDAR_CARAC_LABELS;
    std::string command = "python ";
    command += filename;
    command += Tipo;
    command += "/";
    command += Classifier;
    command += "/";
    command += "testingEscenaSIFTFVSVM.py";

    std::string dirModelo = " --dir ";
    std::string dirCandBordes = " --dirbordes ";
    std::string dirCandExhaus = " --direx ";
    std::string versionModelo = " --version ";
    std::string tipoVentaneo = " --tipovent ";
    std::string descripBordes = " --tdesbordes ";
    std::string descripExhaus = " --tdesex ";
    std::string graficar = " --visualize ";
    std::string Threshold = " --thereshold ";
    std::string frame = " --frame ";
    std::string groundTruth = " --ground ";
    std::string dirGMM = " --dirGMM ";
    std::string tag = " --tag ";
    std::string diferencia = " --diferencia ";
    std::string ancho = " --ancho ";
    std::string alto = " --alto ";

    std::string critConteo = " --critConteo ";
    std::string critScore = " --critPredic ";
    std::string scale = " --escala ";

    FILE* F;
    F=fopen("SIFTSVMEscena.txt","w");

    scale += (ui->factorVentana->toPlainText()).toStdString();
    command += scale;

    critConteo += (ui->criterioCont->toPlainText()).toStdString();
    command += critConteo;

    critScore += (ui->criterioPredic->toPlainText()).toStdString();
    command += critScore;

    std::string nombre = Classifier;
    std::string BancoImag = (ui->ListaEntrenamientos->currentText()).toStdString();
    nombre += BancoImag;
    versionModelo += nombre;
    versionModelo += ".pkl";
    command += versionModelo;

    dirModelo += DIRECTORIO_GUARDAR_MODELO;
    command += dirModelo;
    dirCandBordes += PATH_CANDIDATOS;
    command += dirCandBordes;
    dirCandExhaus += PATH_CANDIDATOS;
    command += dirCandExhaus;
    graficar += (QString::number(Visua)).toStdString();
    command += graficar;
    tipoVentaneo += (QString::number(Barrido)).toStdString();
    command += tipoVentaneo;
    descripBordes += PATH_DESCRIPCIONBARRIDOS;
    descripBordes += "DescripcionBarrido.txt";
    command += descripBordes;
    descripExhaus += PATH_DESCRIPCIONBARRIDOS;
    descripExhaus += "DescripcionBarrido.txt";
    command += descripExhaus;
    Threshold += (ui->criterioNMS->toPlainText()).toStdString();
    command += Threshold;
    diferencia += (ui->criterioNMS->toPlainText()).toStdString();
    command += diferencia;
    tag += (ui->ListaEntrenamientos->currentText()).toStdString();
    command += tag;
    frame += strFileName.toStdString();
    command += frame;

    ancho += (ui->anchoEscena->toPlainText()).toStdString();
    command += ancho;

    alto += (ui->altoEscena->toPlainText()).toStdString();
    command += alto;

    std::size_t pos = (strFileName.toStdString()).find("BancoEscenas");
    std::string str3 = (strFileName.toStdString()).substr (pos);     // Otras/Escena.pjg
    std::size_t pos2 = (str3).find("/");
    str3 = (str3).substr (pos2+1);     // Otras/1.jpg
    std::size_t pos3 = (str3).find("/");
    str3 = (str3).substr (pos3+1);     // 1.jpg
    std::size_t pos4 = (str3).find(".");

    std::string dirPath = (strFileName.toStdString()).substr (0,pos+pos2+pos3+2);
    std::string nom = (str3).substr (0,pos4); //1

    dirPath += nom;
    dirPath += ".txt";

    groundTruth += dirPath;
    command += groundTruth;

    dirGMM += DIRECTORIO_GUARDAR_MODELO;
    dirGMM += "DiccionarioGMM/";
    command += dirGMM;

    fprintf(F, "%s \n",command.c_str());
    fclose(F);
    system(command.c_str());

}

void Detector_Personas::LlamarPythonSIFTFVSVM(std::string Tipo, std::string Classifier){
    FILE* F;
    F =fopen("SIFTSVMTrain.txt","w");

    std::string filename = DIR_GUARDAR_CARAC_LABELS;
    std::string command = "python ";
    command += filename;
    command += Tipo;
    command += "/";
    command += Classifier;
    command += "/";

    if(strcmp(Tipo.c_str(),"Entrenamiento")==0){
        command += "trainingSIFTFVSVM.py";

        std::string dirImagPos = " --dirPos ";
        std::string dirImagNeg = " --dirNeg ";
        std::string numClus = " --clus ";
        std::string dirGMM = " --dirgmm ";
        std::string dirRF = " --dirsvm ";
        std::string nameRF = " --namemodelSVM ";
        std::string numImagPos = " --nlimPos ";
        std::string numImagNeg = " --nlimNeg ";
        std::string numArb = " --narb ";
        std::string version = " --verdic ";

        std::string dirPos = DIR_BANCO_IMAG;
        dirPos += (ui->ListaBancos->currentText()).toStdString();
        dirPos += "/";
        dirPos += Tipo;
        dirPos += "/Persona";
        dirImagPos += dirPos;
        command += dirImagPos;

        std::string dirNeg = DIR_BANCO_IMAG;
        dirNeg += (ui->ListaBancos->currentText()).toStdString();
        dirNeg += "/";
        dirNeg += Tipo;
        dirNeg += "/No_Persona";
        dirImagNeg += dirNeg;
        command += dirImagNeg;

        numClus +=(ui->numClus->toPlainText()).toStdString();
        command += numClus;

        std::string dirGuardarGMM = DIRECTORIO_GUARDAR_MODELO;
        dirGuardarGMM += "DiccionarioGMM/";
        dirGMM += dirGuardarGMM;
        command += dirGMM;

        dirRF += DIRECTORIO_GUARDAR_MODELO;
        command += dirRF;

        std::string nombre = Classifier;
        std::string BancoImag = (ui->ListaBancos->currentText()).toStdString();
        nombre += BancoImag;
        nombre += ".pkl";
        nameRF += nombre;
        command += nameRF;

        numImagPos += (ui->numPos->toPlainText()).toStdString();
        command += numImagPos;

        numImagNeg += (ui->numNeg->toPlainText()).toStdString();
        command += numImagNeg;

        numArb += (ui->numTrees->toPlainText()).toStdString();
        command += numArb;

        version += (ui->ListaBancos->currentText()).toStdString();
        command += version;

        fprintf(F,"%s \n",command.c_str());
        fclose(F);
        system(command.c_str());
    }
    else if(strcmp(Tipo.c_str(),"Prueba")==0){
        command += "testingSIFT.py";

        std::string dirImagPos = " --dir2 ";
        std::string dirImagNeg = " --dir3 ";
        std::string dirMod = " --dir ";
        std::string nameSVM = " --name ";
        std::string version = " --version ";
        std::string dirDic = " --dirDic ";
        std::string numImagPos = " --nlimPos ";
        std::string numImagNeg = " --nlimNeg ";

        std::string dirPos = DIR_BANCO_IMAG;
        dirPos += (ui->bancosTest->currentText()).toStdString();
        dirPos += "/";
        dirPos += Tipo;
        dirPos += "/Persona";
        dirImagPos += dirPos;
        command += dirImagPos;

        std::string dirNeg = DIR_BANCO_IMAG;
        dirNeg += (ui->bancosTest->currentText()).toStdString();
        dirNeg += "/";
        dirNeg += Tipo;
        dirNeg += "/No_Persona";
        dirImagNeg += dirNeg;
        command += dirImagNeg;

        dirMod +=DIRECTORIO_GUARDAR_MODELO;
        command += dirMod;

        std::string dirGuardarGMM = DIRECTORIO_GUARDAR_MODELO;
        dirGuardarGMM += "DiccionarioGMM/";
        dirDic += dirGuardarGMM;
        command += dirDic;

        nameSVM += Classifier;
        command += nameSVM;

        version += (ui->bancosTrain->currentText()).toStdString();
        command += version;

        numImagPos += (ui->posTest->toPlainText()).toStdString();
        command += numImagPos;

        numImagNeg += (ui->negTest->toPlainText()).toStdString();
        command += numImagNeg;

        system(command.c_str());
    }

}

void Detector_Personas::LlamarPythonANN(std::string Tipo, std::string Classifier){

    FILE* F;
    F=fopen("ANNTrain.txt","w");

    std::string filename = DIR_GUARDAR_CARAC_LABELS;
    std::string command = "python ";
    command += filename;
    command += Tipo;
    command += "/";
    command += Classifier;
    command += "/";

    if(strcmp(Tipo.c_str(),"Entrenamiento")==0){
        command += "trainingANN.py";
    }

    std::string dirTrain = " --dir ";
    std::string bancoUtilizado = " --banco ";
    std::string learnRate = " --learning ";
    std::string epocas = " --epocas ";
    std::string numEjemPos = " --numExamPos ";
    std::string numEjemNeg = " --numExamNeg ";
    std::string batchSize = " --batch ";
    std::string numNeuronas = " --numNeurons ";
    std::string dirGuardar = " --dirModelo ";

    dirTrain += DIR_BANCO_IMAG;
    command += dirTrain;

    bancoUtilizado += (ui->ListaBancos->currentText()).toStdString();
    command += bancoUtilizado;

    learnRate +=(ui->learningRate->toPlainText()).toStdString();
    command += learnRate;

    epocas +=(ui->epochs->toPlainText()).toStdString();
    command += epocas;

    batchSize +=(ui->batch->toPlainText()).toStdString();
    command += batchSize;

    numNeuronas +=(ui->numNeurons->toPlainText()).toStdString();
    command += numNeuronas;

    numEjemPos +=(ui->numPos->toPlainText()).toStdString();
    command += numEjemPos;

    numEjemNeg +=(ui->numNeg->toPlainText()).toStdString();
    command += numEjemNeg;

    dirGuardar += DIRECTORIO_GUARDAR_MODELO;
    command += dirGuardar;

    fprintf(F,"%s \n",command.c_str());
    fclose(F);

    system(command.c_str());
}

void Detector_Personas::LlamarPythonANNEscena(int Visua){
    FILE* F1;

    F1 = fopen("ANNCandidatos.txt","w");

    std::string filename;
    std::string command;

    remove("C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/ModelosClasificacion/Escena/ANN/Predicciones.txt");
    remove("C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/ModelosClasificacion/Escena/ANN/Scores.txt");

    std::string dirimag;
    std::string bancoUtilizado;

    int numCandidatos = 0;

    //LECTURA FICHERO CON NUMERO DE CANDIDATOS
    FILE *Fichero;
    Fichero = fopen("C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/TecnicasBarrido/DescripcionBarrido/NumeroCandidatos.txt", "r");

    if (Fichero == NULL){
        printf("\nError al abrir el archivo de texto NumeroCandidatos.txt... \n");
    }
    else{
        fscanf(Fichero, "%d", &numCandidatos);
    }
    fclose(Fichero);

    std::cout << "INICIO LECTURA CANDIDATOS" <<"\n \n";

    for (int i = 1; i < numCandidatos + 1; i++){
        filename = "C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/ModelosClasificacion/Escena/ANN/testingANN.py";
        command = "python ";
        dirimag = " --image ";
        bancoUtilizado = " --banco ";

        command += filename;

        dirimag += PATH_CANDIDATOS;
        dirimag += "\\";
        dirimag += (QString::number(i)).toStdString();
        dirimag += ".jpg";
        command += dirimag;
        bancoUtilizado += (ui->ListaEntrenamientos->currentText()).toStdString();
        command += bancoUtilizado;

        std::cout << "Prediccion: " << i << "\n";

        fprintf(F1,"%s \n",command.c_str());
        fclose(F1);

        system(command.c_str());
    }

    std::string descripFrame = " --tdes ";
    std::string dirScores = " --dirscores ";
    std::string dirPredic = " --dirpred ";
    std::string visua = " --visualize ";
    std::string threshold = " --thereshold ";
    std::string frame = " --frame ";
    std::string ground = " --ground ";
    std::string diferencia = " --diferencia ";
    std::string ancho = " --ancho ";
    std::string alto = " --alto ";
    std::string critConteo = " --critConteo ";
    std::string critScore = " --critPredic ";
    std::string scale = " --escala ";


    filename = "C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/ModelosClasificacion/Escena/ANN/testingEscenaANN.py";
    command = "python ";

    command += filename;

    critConteo += (ui->criterioCont->toPlainText()).toStdString();
    command += critConteo;

    critScore += (ui->criterioPredic->toPlainText()).toStdString();
    command += critScore;

    scale += (ui->factorVentana->toPlainText()).toStdString();
    command += scale;

    descripFrame += "/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/TecnicasBarrido/DescripcionBarrido/DescripcionBarrido.txt";
    command += descripFrame;
    dirScores += "C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/ModelosClasificacion/Escena/ANN/Scores.txt";
    command += dirScores;
    dirPredic += "C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/ModelosClasificacion/Escena/ANN/Predicciones.txt";
    command += dirPredic;
    visua += (QString::number(Visua)).toStdString();
    command += visua;
    threshold += (ui->criterioNMS->toPlainText()).toStdString();
    command += threshold;
    frame += (strFileName).toStdString();
    command += frame;

    ancho += (ui->anchoEscena->toPlainText()).toStdString();
    command += ancho;

    alto += (ui->altoEscena->toPlainText()).toStdString();
    command += alto;

    std::size_t pos = (strFileName.toStdString()).find("BancoEscenas");
    std::string str3 = (strFileName.toStdString()).substr (pos);     // Otras/Escena.pjg
    std::size_t pos2 = (str3).find("/");
    str3 = (str3).substr (pos2+1);     // Otras/1.jpg
    std::size_t pos3 = (str3).find("/");
    str3 = (str3).substr (pos3+1);     // 1.jpg
    std::size_t pos4 = (str3).find(".");

    std::string dirPath = (strFileName.toStdString()).substr (0,pos+pos2+pos3+2);
    std::string nom = (str3).substr (0,pos4); //1

    dirPath += nom;
    dirPath += ".txt";

    ground += dirPath;
    command += ground;

    diferencia += (ui->criterioDif->toPlainText()).toStdString();
    command += diferencia;

    std::cout << "\n \n"<< "REPORTE DE ERRORRES" << "\n \n";

    system(command.c_str());
}

void Detector_Personas::LlamarPythonCNNEscena(int Visua){

    FILE* F1;

    F1 = fopen("CNNCandidatos.txt","w");

    FILE* F2;

    F2 = fopen("CNNReporte.txt","w");

    std::string filename;
    std::string command;

    remove("C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/ModelosClasificacion/Escena/CNN/Predicciones.txt");
    remove("C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/ModelosClasificacion/Escena/CNN/Scores.txt");

    std::string dirimag;
    std::string outGraph;
    std::string outLabels;

    int numCandidatos = 0;

    //LECTURA FICHERO CON NUMERO DE CANDIDATOS
    FILE *Fichero;
    Fichero = fopen("C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/TecnicasBarrido/DescripcionBarrido/NumeroCandidatos.txt", "r");

    if (Fichero == NULL){
        printf("\nError al abrir el archivo de texto NumeroCandidatos.txt... \n");
    }
    else{
        fscanf(Fichero, "%d", &numCandidatos);
    }
    fclose(Fichero);

    std::cout << "INICIO LECTURA CANDIDATOS" <<"\n \n";

    for (int i = 1; i < numCandidatos + 1; i++){
        filename = "C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/ModelosClasificacion/Escena/CNN/testingCNN.py";
        command = "python ";
        dirimag = " --image ";
        outGraph = " --graph ";
        outLabels = " --labels ";

        command += filename;

        dirimag += PATH_CANDIDATOS;
        dirimag += "\\";
        dirimag += (QString::number(i)).toStdString();
        dirimag += ".jpg";
        command += dirimag;

        outGraph += "C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/ModelosClasificacion/ModelosEntrenados/CNN/";
        outGraph += (ui->ListaEntrenamientos->currentText()).toStdString();
        outGraph += "/output_graph.pb";
        command += outGraph;

        outLabels += "C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/ModelosClasificacion/ModelosEntrenados/CNN/";
        outLabels += (ui->ListaEntrenamientos->currentText()).toStdString();
        outLabels += "/output_labels.txt";
        command += outLabels;

        std::cout << "Prediccion: " << i << "\n";

        fprintf(F1,"%s \n",command.c_str());
        fclose(F1);
        system(command.c_str());
    }

    std::string descripFrame = " --tdes ";
    std::string dirScores = " --dirscores ";
    std::string dirPredic = " --dirpred ";
    std::string visua = " --visualize ";
    std::string threshold = " --thereshold ";
    std::string frame = " --frame ";
    std::string ground = " --ground ";
    std::string diferencia = " --diferencia ";
    std::string ancho = " --ancho ";
    std::string alto = " --alto ";

    std::string critConteo = " --critConteo ";
    std::string critScore = " --critPredic ";
    std::string scale = " --escala ";


    filename = "C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/ModelosClasificacion/Escena/CNN/testingEscenaCNN.py";
    command = "python ";

    command += filename;

    critConteo += (ui->criterioCont->toPlainText()).toStdString();
    command += critConteo;

    critScore += (ui->criterioPredic->toPlainText()).toStdString();
    command += critScore;

    scale += (ui->factorVentana->toPlainText()).toStdString();
    command += scale;

    descripFrame += "/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/TecnicasBarrido/DescripcionBarrido/DescripcionBarrido.txt";
    command += descripFrame;
    dirScores += "C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/ModelosClasificacion/Escena/CNN/Scores.txt";
    command += dirScores;
    dirPredic += "C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/ModelosClasificacion/Escena/CNN/Predicciones.txt";
    command += dirPredic;
    visua += (QString::number(Visua)).toStdString();
    command += visua;
    threshold += (ui->criterioNMS->toPlainText()).toStdString();
    command += threshold;
    frame += (strFileName).toStdString();
    command += frame;

    std::size_t pos = (strFileName.toStdString()).find("BancoEscenas");
    std::string str3 = (strFileName.toStdString()).substr (pos);     // Otras/Escena.pjg
    std::size_t pos2 = (str3).find("/");
    str3 = (str3).substr (pos2+1);     // Otras/1.jpg
    std::size_t pos3 = (str3).find("/");
    str3 = (str3).substr (pos3+1);     // 1.jpg
    std::size_t pos4 = (str3).find(".");

    std::string dirPath = (strFileName.toStdString()).substr (0,pos+pos2+pos3+2);
    std::string nom = (str3).substr (0,pos4); //1

    dirPath += nom;
    dirPath += ".txt";

    ground += dirPath;
    command += ground;

    diferencia += (ui->criterioDif->toPlainText()).toStdString();
    command += diferencia;

    ancho += (ui->anchoEscena->toPlainText()).toStdString();
    command += ancho;

    alto += (ui->altoEscena->toPlainText()).toStdString();
    command += alto;

    std::cout << "\n \n"<< "REPORTE DE ERRORRES" << "\n \n";

    fprintf(F2,"%s \n",command.c_str());
    fclose(F2);

    system(command.c_str());
}

void Detector_Personas::LlamarPythonCNN(std::string Tipo, std::string Classifier){

    FILE* F;
    F=fopen("CNNTrain.txt","w");

    std::string filename = DIR_GUARDAR_CARAC_LABELS;
    std::string command = "python ";
    command += filename;
    command += Tipo;
    command += "/";
    command += Classifier;
    command += "/";

    if(strcmp(Tipo.c_str(),"Entrenamiento")==0){
        command += "trainingCNN.py";
    }

    std::string dirimag = " --image_dir ";
    std::string outGraph = " --output_graph ";
    std::string outLabels = " --output_labels ";
    std::string summ = " --summaries_dir ";
    std::string bottle = " --bottleneck_dir ";
    std::string trainSteps = " --how_many_training_steps ";
    std::string learnRate = " --learning_rate ";
    std::string batchSize = " --train_batch_size ";
    std::string id = " --id ";

    batchSize +=(ui->batch->toPlainText()).toStdString();
    command += batchSize;

    learnRate +=(ui->learningRate->toPlainText()).toStdString();
    command += learnRate;

    trainSteps +=(ui->epochs->toPlainText()).toStdString();
    command += trainSteps;

    id += Classifier;
    id += "/";
    command += id;

    dirimag += DIR_BANCO_IMAG;
    dirimag += (ui->ListaBancos->currentText()).toStdString();
    dirimag += "/";
    dirimag += Tipo;
    dirimag += "/";
    command += dirimag;

    outGraph += DIRECTORIO_GUARDAR_MODELO;
    outGraph += Classifier;
    outGraph += "/";
    outGraph += (ui->ListaBancos->currentText()).toStdString();
    outGraph += "/output_graph.pb";
    command += outGraph;

    outLabels += DIRECTORIO_GUARDAR_MODELO;
    outLabels += Classifier;
    outLabels += "/";
    outLabels += (ui->ListaBancos->currentText()).toStdString();
    outLabels += "/output_labels.txt";
    command += outLabels;

    summ += DIRECTORIO_GUARDAR_MODELO;
    summ += Classifier;
    summ += "/";
    summ += (ui->ListaBancos->currentText()).toStdString();
    summ += "/logs";
    command += summ;

    bottle += DIRECTORIO_GUARDAR_MODELO;
    bottle += Classifier;
    bottle += "/";
    bottle += (ui->ListaBancos->currentText()).toStdString();
    bottle += "/bottleneck";
    command += bottle;

    fprintf(F,"%s \n",command.c_str());
    fclose(F);

    system(command.c_str());

}

void Detector_Personas::CargarImagenesEscena(std::string Directorio, std::vector<cv::Mat>& Imagenes)
{
    cv::Mat imagen;
    std::vector<std::string> NombresArchivos;

    NombresArchivos = Detector_Personas::ArchivosEnDirectorio(Directorio);

    for (int i = 0; i < NombresArchivos.size() ; ++i) {
        imagen = cv::imread(NombresArchivos.at(i));
        if (imagen.empty()){
            std::cout << "ERROR: No se pudo leer la imagen " << i + 1 << " del directorio.\n";
            continue;
        }
        //imshow("Imagen cargada", imagen);
        cv::resize(imagen, imagen, TAMANO_IMAGEN);
        std::cout << "Se cargo la imagen" << i + 1 << ". \n";
        //imshow("imagen modificada", imagen);
        Imagenes.push_back(imagen.clone());
    }
    std::cout << "\n";

}

void Detector_Personas::CreacionFicherosPythonEscena(std::vector< cv::Mat > & Gradientes, std::string Tipo, std::string Classifier){
    //Creación fichero para caracteristicas
    FILE *Fichero;

    std::string dir = DIR_GUARDAR_CARAC_LABELS;
    dir += Tipo;
    dir += "/";
    dir += Classifier;
    std::string Nombre = "/CaracteristicasHOG";
    std::string Extension = ".txt";
    Nombre += Tipo;
    Nombre += Extension;
    dir += Nombre;

    Fichero = fopen(dir.c_str(), "w");

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
}

void Detector_Personas::LlamarPythonHOGRFEscena(std::string Tipo, std::string Classifier, int Visua, int Barrido){
    FILE* Fich;
    Fich = fopen("HOGRFEscena.txt","w");

    std::string filename = DIR_GUARDAR_CARAC_LABELS;
    std::string command = "python ";
    command += filename;
    command += Tipo;
    command += "/";
    command += Classifier;
    command += "/";
    command += "testingEscena";
    command += Classifier;
    command += ".py";

    std::string nameRF = " --namemodelRF ";
    std::string dirCaracBordes = " --tcarbordes ";
    std::string dirCaracExhaus = " --tcarex ";
    std::string descripBordes = " --tdesbordes ";
    std::string descripExhaus = " --tdesex ";
    std::string numCarac = " --ncha ";
    std::string dirModeloRF = " --dirrf ";
    std::string tipoVentaneo = " --tipovent ";
    std::string graficar = " --visualize ";
    std::string Threshold = " --thereshold ";
    std::string frame = " --frame ";
    std::string groundTruth = " --ground ";
    std::string diferencia = " --diferencia ";
    std::string ancho = " --ancho ";
    std::string alto = " --alto ";
    std::string scale = " --escala ";

    scale += (ui->factorVentana->toPlainText()).toStdString();
    command += scale;

    std::string critConteo = " --critConteo ";
    std::string critScore = " --critPredic ";

    critConteo += (ui->criterioCont->toPlainText()).toStdString();
    command += critConteo;

    critScore += (ui->criterioPredic->toPlainText()).toStdString();
    command += critScore;

    std::string nombre = Classifier;
    std::string BancoImag = (ui->ListaEntrenamientos->currentText()).toStdString();
    BancoImag += ".pkl";
    nombre += BancoImag;
    nameRF += nombre;
    command += nameRF;

    std::string dir = DIR_GUARDAR_CARAC_LABELS;
    dir += Tipo;
    dir += "/";
    dir += Classifier;
    std::string Nombre = "/CaracteristicasHOG";
    std::string Extension = ".txt";
    Nombre += Tipo;
    Nombre += Extension;
    dir += Nombre;

    dirCaracBordes += dir;
    command += dirCaracBordes;

    dirCaracExhaus += dir;
    command += dirCaracExhaus;

    descripBordes += PATH_DESCRIPCIONBARRIDOS;
    descripBordes += "DescripcionBarrido.txt";
    command += descripBordes;

    descripExhaus += PATH_DESCRIPCIONBARRIDOS;
    descripExhaus += "DescripcionBarrido.txt";
    command += descripExhaus;

    numCarac += (QString::number(NUM_CARACTERISTICAS)).toStdString();
    command += numCarac;

    dirModeloRF += DIRECTORIO_GUARDAR_MODELO;
    command += dirModeloRF;

    tipoVentaneo += (QString::number(Barrido)).toStdString();
    command += tipoVentaneo;

    graficar += (QString::number(Visua)).toStdString();
    command += graficar;

    Threshold += (ui->criterioNMS->toPlainText()).toStdString();
    command += Threshold;

    frame += strFileName.toStdString();
    command += frame;

    ancho += (ui->anchoEscena->toPlainText()).toStdString();
    command += ancho;

    alto += (ui->altoEscena->toPlainText()).toStdString();
    command += alto;

    std::size_t pos = (strFileName.toStdString()).find("BancoEscenas");
    std::string str3 = (strFileName.toStdString()).substr (pos);     // Otras/Escena.pjg
    std::size_t pos2 = (str3).find("/");
    str3 = (str3).substr (pos2+1);     // Otras/1.jpg
    std::size_t pos3 = (str3).find("/");
    str3 = (str3).substr (pos3+1);     // 1.jpg
    std::size_t pos4 = (str3).find(".");

    std::string dirPath = (strFileName.toStdString()).substr (0,pos+pos2+pos3+2);
    std::string nom = (str3).substr (0,pos4); //1

    dirPath += nom;
    dirPath += ".txt";

    groundTruth += dirPath;
    command += groundTruth;

    diferencia += (ui->criterioDif->toPlainText()).toStdString();
    command += diferencia;
    fprintf(Fich,"%s \n",command.c_str());
    fclose(Fich);
    system(command.c_str());
}

void Detector_Personas::LlamarPythonHOGSVMEscena(std::string Tipo, std::string Classifier, int Visua, int Barrido){
    FILE* Fich;
    Fich = fopen("HOGSVMEscena.txt","w");

    std::string filename = DIR_GUARDAR_CARAC_LABELS;
    std::string command = "python ";
    command += filename;
    command += Tipo;
    command += "/";
    command += Classifier;
    command += "/";
    command += "testingEscena";
    command += Classifier;
    command += ".py";

    std::string nameRF = " --namemodelSVM ";
    std::string dirCaracBordes = " --tcarbordes ";
    std::string dirCaracExhaus = " --tcarex ";
    std::string descripBordes = " --tdesbordes ";
    std::string descripExhaus = " --tdesex ";
    std::string numCarac = " --ncha ";
    std::string dirModeloRF = " --dirsvm ";
    std::string tipoVentaneo = " --tipovent ";
    std::string graficar = " --visualize ";
    std::string Threshold = " --thereshold ";
    std::string frame = " --frame ";
    std::string groundTruth = " --ground ";
    std::string diferencia = " --diferencia ";
    std::string ancho = " --ancho ";
    std::string alto = " --alto ";

    std::string critConteo = " --critConteo ";
    std::string critScore = " --critPredic ";
    std::string scale = " --escala ";

    scale += (ui->factorVentana->toPlainText()).toStdString();
    command += scale;

    critConteo += (ui->criterioCont->toPlainText()).toStdString();
    command += critConteo;

    critScore += (ui->criterioPredic->toPlainText()).toStdString();
    command += critScore;

    std::string nombre = Classifier;
    std::string BancoImag = (ui->ListaEntrenamientos->currentText()).toStdString();
    BancoImag += ".pkl";
    nombre += BancoImag;
    nameRF += nombre;
    command += nameRF;

    std::string dir = DIR_GUARDAR_CARAC_LABELS;
    dir += Tipo;
    dir += "/";
    dir += Classifier;
    std::string Nombre = "/CaracteristicasHOG";
    std::string Extension = ".txt";
    Nombre += Tipo;
    Nombre += Extension;
    dir += Nombre;

    dirCaracBordes += dir;
    command += dirCaracBordes;

    dirCaracExhaus += dir;
    command += dirCaracExhaus;

    descripBordes += PATH_DESCRIPCIONBARRIDOS;
    descripBordes += "DescripcionBarrido.txt";
    command += descripBordes;

    descripExhaus += PATH_DESCRIPCIONBARRIDOS;
    descripExhaus += "DescripcionBarrido.txt";
    command += descripExhaus;

    numCarac += (QString::number(NUM_CARACTERISTICAS)).toStdString();
    command += numCarac;

    dirModeloRF += DIRECTORIO_GUARDAR_MODELO;
    command += dirModeloRF;

    tipoVentaneo += (QString::number(Barrido)).toStdString();
    command += tipoVentaneo;

    graficar += (QString::number(Visua)).toStdString();
    command += graficar;

    Threshold += (ui->criterioNMS->toPlainText()).toStdString();
    command += Threshold;

    frame += strFileName.toStdString();
    command += frame;

    ancho += (ui->anchoEscena->toPlainText()).toStdString();
    command += ancho;

    alto += (ui->altoEscena->toPlainText()).toStdString();
    command += alto;

    std::size_t pos = (strFileName.toStdString()).find("BancoEscenas");
    std::string str3 = (strFileName.toStdString()).substr (pos);     // Otras/Escena.pjg
    std::size_t pos2 = (str3).find("/");
    str3 = (str3).substr (pos2+1);     // Otras/1.jpg
    std::size_t pos3 = (str3).find("/");
    str3 = (str3).substr (pos3+1);     // 1.jpg
    std::size_t pos4 = (str3).find(".");

    std::string dirPath = (strFileName.toStdString()).substr (0,pos+pos2+pos3+2);
    std::string nom = (str3).substr (0,pos4); //1

    dirPath += nom;
    dirPath += ".txt";

    groundTruth += dirPath;
    command += groundTruth;

    diferencia += (ui->criterioDif->toPlainText()).toStdString();
    command += diferencia;

    fprintf(Fich,"%s \n",command.c_str());
    fclose(Fich);
    system(command.c_str());

}

void Detector_Personas::BarridoExhaustivo(){
    tipoBarrido=0;

    cv::Mat frameExhaustivo = frameCargado.clone();

    medianBlur(frameExhaustivo, frameExhaustivo, 7);

    //Elimina el directorio con candidatos de la escena anterior
    std::string Path = PATH_CANDIDATOS;
    std::string folderRemoveCommand = "rmdir /S /Q ";
    folderRemoveCommand += Path;
    system(folderRemoveCommand.c_str());

    //Elimina el directorio de resultados de la escena anterior
    Path = PATH_RESULTADOSEXHAUS;
    folderRemoveCommand = "rmdir /S /Q ";
    folderRemoveCommand += Path;
    system(folderRemoveCommand.c_str());

    //Elimina el fichero con la información sobre las coordenadas de los candidatos
    folderRemoveCommand = PATH_DESCRIPCIONBARRIDOS;
    folderRemoveCommand += "DescripcionBarrido.txt";
    remove(folderRemoveCommand.c_str());

    //Creación del directorio para guardar ventanas
    std::string folderName = PATH_CANDIDATOS;
    std::string folderCreateCommand = "mkdir ";
    folderCreateCommand += folderName;
    system(folderCreateCommand.c_str());

    folderName = PATH_RESULTADOSEXHAUS;
    folderCreateCommand = "mkdir ";
    folderCreateCommand += folderName;
    system(folderCreateCommand.c_str());

    //Parametros con el tamaño real de la imagen
    int Real_Columnas = (ui->anchoEscena->toPlainText()).toInt();
    int Real_Filas = (ui->altoEscena->toPlainText()).toInt();

    //Parametros para la ventana
    int windows_n_cols = (ui->anchoVentana->toPlainText()).toInt();
    int windows_n_rows = (ui->altoVentana->toPlainText()).toInt(); //Siempre debe ser el doble del ancho
    int StepSlide = (ui->stride->toPlainText()).toInt();
    float Scale = (ui->factorVentana->toPlainText()).toFloat();

    int NumeroNombre = 1;
    int ContadorCandidatos = 1;

    //CREACIÓN DEL FICHERO PARA BLOQUE POST-PROCESAMIENTO
    FILE *Fichero;
    std::string aux = PATH_DESCRIPCIONBARRIDOS;
    aux += "DescripcionBarrido.txt";
    Fichero = fopen(aux.c_str(), "w");
    if (Fichero == NULL){
        printf("\nError al crear el archivo de texto DescripcionBarridoExhaustivo.txt... \n");
    }
    else{
        fprintf(Fichero, "%d\n", windows_n_cols); //Escribe el ancho de la ventana
        fprintf(Fichero, "%d\n", windows_n_rows); //Escribe el alto de la ventana
        fprintf(Fichero, "%c\n", '*'); //Indica el final info sobre ventana

        for (int Factor = 0; Factor < (ui->escalas->toPlainText()).toInt(); Factor++) {

            float Escala = pow(Scale, Factor);
            printf("Factor: %f \n", Escala);

            cv::resize(frameExhaustivo, frameExhaustivo, cv::Size(Real_Columnas / Escala, Real_Filas / Escala));
            //namedWindow(cv::format("Escala %d", Escala), WINDOW_AUTOSIZE);
            //imshow(cv::format("Escala %d", Escala), frameExhaustivo);
            cv::imwrite(cv::format("C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/TecnicasBarrido/BarridoVentana/Resultados/Factor %f.JPG", Escala), frameExhaustivo);

            cv::Mat DrawResultGrid = frameExhaustivo.clone();

            for (float row = 0; row <= frameExhaustivo.rows - windows_n_rows*Escala; row += StepSlide)
            {
                for (float col = 0; col <= frameExhaustivo.cols - windows_n_cols*Escala; col += StepSlide)
                {
                    //INFORMACIÓN COORDENADAS DE CANDIDATO ACTUAL
                    fprintf(Fichero, "%f\n", row); //Escribe coordenada en x
                    fprintf(Fichero, "%f\n", col); //Escribe coordenada en y

                    // resulting window
                    cv::Rect windows(col, row, windows_n_cols, windows_n_rows);

                    cv::Mat DrawResultHere = frameExhaustivo.clone();

                    // Draw only rectangle
                    cv::rectangle(DrawResultHere, windows, cv::Scalar(255), 1, 8, 0);
                    // Draw grid
                    cv::rectangle(DrawResultGrid, windows, cv::Scalar(255), 1, 8, 0);

                    // Show  rectangle
                    namedWindow("Ventana en movimiento", cv::WINDOW_AUTOSIZE);
                    cv::imshow("Ventana en movimiento", DrawResultHere);

                    // Show grid
                    //namedWindow("Barrido realizado", WINDOW_AUTOSIZE);
                    //imshow("Barrido realizado", DrawResultGrid);
                    cv::imwrite(cv::format("C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/TecnicasBarrido/BarridoVentana/Resultados/Barrido factor %f.JPG", Escala), DrawResultGrid);

                    //Ver ROI
                    cv::Mat Roi = frameExhaustivo(windows);
                    //namedWindow("Ventana segmentada", WINDOW_AUTOSIZE);
                    //imshow("Ventana segmentada", Roi);
                    cv::waitKey(2);
                    cv::resize(Roi, Roi, cv::Size(64, 128));
                    std::string carpeta = PATH_CANDIDATOS;
                    carpeta += "\\%d.JPG";
                    cv::imwrite(cv::format("C:\\Deteccion_Conteo_Personas_Escenas_PUJ-1703\\Recursos\\TecnicasBarrido\\Candidatos\\%d.JPG", NumeroNombre), Roi);
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
    aux = PATH_DESCRIPCIONBARRIDOS;
    aux += "NumeroCandidatos.txt";
    Fichero2 = fopen(aux.c_str(), "w");
    if (Fichero2 == NULL){
        printf("\nError al crear el archivo de texto NumeroCandidatos.txt... \n");
    }
    else{
        fprintf(Fichero2, "%d", (NumeroNombre - 1)); //Indica el final de todo el proceso de barrido
    }
    fclose(Fichero2);
}

void Detector_Personas::BarridoBordes(){
    tipoBarrido=1;

    cv::Mat frameBordes= frameCargado.clone();

    medianBlur(frameBordes, frameBordes, 7);

    //Elimina el directorio con candidatos de la escena anterior
    std::string Path = PATH_CANDIDATOS;
    std::string folderRemoveCommand = "rmdir /S /Q ";
    folderRemoveCommand += Path;
    system(folderRemoveCommand.c_str());

    //Elimina el directorio de resultados de la escena anterior
    Path = PATH_RESULTADOSBORDES;
    folderRemoveCommand = "rmdir /S /Q ";
    folderRemoveCommand += Path;
    system(folderRemoveCommand.c_str());

    //Elimina el fichero con la información sobre las coordenadas de los candidatos
    folderRemoveCommand = PATH_DESCRIPCIONBARRIDOS;
    folderRemoveCommand += "DescripcionBarrido.txt";
    remove(folderRemoveCommand.c_str());

    //Creación del directorio para guardar ventanas
    std::string folderName = PATH_CANDIDATOS;
    std::string folderCreateCommand = "mkdir ";
    folderCreateCommand += folderName;
    system(folderCreateCommand.c_str());

    folderName = PATH_RESULTADOSBORDES;
    folderCreateCommand = "mkdir ";
    folderCreateCommand += folderName;
    system(folderCreateCommand.c_str());

    //Parametros con el tamaño real de la imagen
    int Real_Columnas = (ui->anchoEscena->toPlainText()).toInt();
    int Real_Filas = (ui->altoEscena->toPlainText()).toInt();

    //Parametros para la ventana
    int windows_n_cols = (ui->anchoVentana->toPlainText()).toInt();
    int windows_n_rows = (ui->altoVentana->toPlainText()).toInt(); //Siempre debe ser el doble del ancho
    int StepSlide = (ui->stride->toPlainText()).toInt();
    float Scale = (ui->factorVentana->toPlainText()).toFloat();
    int Regla;

    int NumeroNombre = 1;

    //Lectura de la escena
    cv::Mat image_f, edges_canny, bw, gray;
    //frameCargado = cv::imread(strFileName.toStdString(),cv::IMREAD_COLOR);        // open image

    //Creación de las ventanas para visualizar
    //cv::namedWindow("RF edges", 1);
    //namedWindow("Canny edges", 1);

    //Extracción de bordes
    frameBordes.convertTo(image_f, cv::DataType<float>::type, 1 / 255.0);
    cv::Mat edges(image_f.size(), image_f.type());
    cv::Ptr<cv::ximgproc::StructuredEdgeDetection> pDollar = cv::ximgproc::createStructuredEdgeDetection("C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/TecnicasBarrido/BarridoBordes/model.yml");
    pDollar->detectEdges(image_f, edges);
    //edges.convertTo(edges, CV_8U, 255.0);
    edges.convertTo(edges, CV_8U, 512.0);
    cv::Mat edges_color;
    //applyColorMap(edges, edges_color, COLORMAP_JET);
    cv::applyColorMap(edges, edges_color, cv::COLORMAP_COOL);
    cv::Canny(edges, bw, 100, 200, 3);
    cv::threshold(edges, bw, 64, 255, CV_THRESH_BINARY);
    cv::cvtColor(frameBordes, gray, cv::COLOR_BGR2GRAY);
    cv::Canny(gray, edges_canny, 60, 180, 3);
    //cv::imshow("RF edges", edges_color);
    //cv::imshow("RF edges", bw);
    //cv::imshow("Canny edges", edges_canny);

    //CREACIÓN DEL FICHERO PARA BLOQUE POST-PROCESAMIENTO
    FILE *Fichero;
    std::string aux = PATH_DESCRIPCIONBARRIDOS;
    aux += "DescripcionBarrido.txt";
    Fichero = fopen(aux.c_str(), "w");
    if (Fichero == NULL){
        printf("\nError al crear el archivo de texto DescripcionBarridoBordes.txt... \n");
    }
    else{
        fprintf(Fichero, "%d\n", windows_n_cols); //Escribe el ancho de la ventana
        fprintf(Fichero, "%d\n", windows_n_rows); //Escribe el alto de la ventana
        fprintf(Fichero, "%c\n", '*'); //Indica el final info sobre ventana

        //Barrido completo de la escena
        for (int Factor = 0; Factor < (ui->escalas->toPlainText()).toInt(); Factor++) {

            if(Factor==0){
                Regla=900000;
            }
            else if(Factor==1){
                Regla=1900000;
            }
            else if(Factor==2){
                Regla=5000000;
            }

            float Escala = pow(Scale, Factor);
            printf("Factor: %f \n", Escala);

            cv::resize(frameBordes, frameBordes, cv::Size(Real_Columnas / Escala, Real_Filas / Escala));
            cv::resize(bw, bw, cv::Size(Real_Columnas / Escala, Real_Filas / Escala));

            //namedWindow(cv::format("Escala %d", Escala), WINDOW_AUTOSIZE);
            //imshow(cv::format("Escala %d", Escala), bw);
            cv::imwrite(cv::format("C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/TecnicasBarrido/BarridoBordes/Resultados/Factor %f.JPG", Escala), bw);

            cv::Mat DrawResultGrid = bw.clone();

            for (float row = 0; row <= bw.rows - windows_n_rows*Escala; row += StepSlide)
            {
                for (float col = 0; col <= bw.cols - windows_n_cols*Escala; col += StepSlide)
                {
                    // resulting window
                    cv::Rect windows(col, row, windows_n_cols, windows_n_rows);

                    cv::Mat DrawResultHere = bw.clone();

                    // Draw only rectangle
                    cv::rectangle(DrawResultHere, windows, cv::Scalar(255), 1, 8, 0);
                    // Draw grid
                    cv::rectangle(DrawResultGrid, windows, cv::Scalar(255), 1, 8, 0);

                    // Show  rectangle
                    cv::namedWindow("Ventana en movimiento", cv::WINDOW_AUTOSIZE);
                    cv::imshow("Ventana en movimiento", DrawResultHere);

                    // Show grid
                    //namedWindow("Barrido realizado", WINDOW_AUTOSIZE);
                    //imshow("Barrido realizado", DrawResultGrid);
                    cv::imwrite(cv::format("C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/TecnicasBarrido/BarridoBordes/Resultados/Barrido factor %f.JPG", Escala), DrawResultGrid);

                    //Ver ROI
                    cv::Mat Roi = bw(windows);
                    //namedWindow("Ventana segmentada", WINDOW_AUTOSIZE);
                    //imshow("Ventana segmentada", Roi);
                    cv::waitKey(1);
                    int Suma = (int)cv::sum(Roi)[0];
                    std::cout << Suma << "\n";
                    if (Suma > Regla){
                        cv::Mat RoiColor = frameBordes(windows);
                        cv::resize(RoiColor, RoiColor, cv::Size(64, 128));
                        //imwrite(cv::format("C:\\Users\\user\\Desktop\\Recursos\\TecnicasBarrido\\Candidatos\\%d-%d.JPG", NumeroNombre, Suma), RoiColor);
                        cv::imwrite(cv::format("C:\\Deteccion_Conteo_Personas_Escenas_PUJ-1703\\Recursos\\TecnicasBarrido\\Candidatos\\%d.JPG", NumeroNombre), RoiColor);
                        NumeroNombre++;

                        //INFORMACIÓN COORDENADAS DE CANDIDATO ACTUAL QUE CUMPLE LA REGLA
                        fprintf(Fichero, "%f\n", row); //Escribe coordenada en x
                        fprintf(Fichero, "%f\n", col); //Escribe coordenada en y
                    }
                }
            }
            fprintf(Fichero, "%c\n", '*'); //Indica el final de candidatos en una escala
        }
        fprintf(Fichero, "%c\n", '-'); //Indica el final de todo el proceso de barrido
    }
    fclose(Fichero);

    FILE *Fichero2;
    aux = PATH_DESCRIPCIONBARRIDOS;
    aux += "NumeroCandidatos.txt";
    Fichero2 = fopen(aux.c_str(), "w");
    if (Fichero2 == NULL){
        printf("\nError al crear el archivo de texto NumeroCandidatos.txt... \n");
    }
    else{
        fprintf(Fichero2, "%d", (NumeroNombre-1)); //Indica el final de todo el proceso de barrido
    }
    fclose(Fichero2);
}

void Detector_Personas::TestANN(){

    std::string filename;
    std::string command;
    std::string dirImage;
    std::string bancoUtilizado;

    remove("C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/ModelosClasificacion/Prueba/ANN/Predicciones.txt");

    //CREACIÓN DEL FICHERO PARA MIRAR LOS ERRORES
    FILE *Fichero;
    Fichero = fopen("C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/ModelosClasificacion/Prueba/ANN/GroundTruth.txt", "w");
    if (Fichero == NULL){
        printf("\nError al crear el archivo de texto GroundTruth.txt... \n");
    }
    else{
        std::cout << "INICIO IMAGENES POSITIVAS" << endl << endl;

        for (int i = 1; i < (ui->posTest->toPlainText()).toInt() + 1; i++){
            filename = "C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/ModelosClasificacion/Prueba/ANN/testingANN.py";
            command = "python ";
            dirImage = " --image ";
            bancoUtilizado = " --banco ";

            command += filename;

            dirImage += DIR_BANCO_IMAG;
            dirImage += (ui->bancosTest->currentText()).toStdString();
            dirImage += "/Prueba/Persona/Pos";
            dirImage += std::to_string(i);
            dirImage += ".jpg";
            command += dirImage;
            bancoUtilizado += (ui->bancosTrain->currentText()).toStdString();
            command += bancoUtilizado;

            fprintf(Fichero, "%s\n", "persona"); //Escribe la clase real

            std::cout << "Prediccion: " << i << endl;

            system(command.c_str());
        }

        std::cout << endl << endl << "INICIO IMAGENES NEGATIVAS" << endl << endl;

        for (int i = 1; i < (ui->negTest->toPlainText()).toInt() + 1; i++){
            filename = "C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/ModelosClasificacion/Prueba/ANN/testingANN.py";
            command = "python ";
            dirImage = " --image ";
            bancoUtilizado = " --banco ";

            command += filename;

            dirImage += DIR_BANCO_IMAG;
            dirImage += (ui->bancosTest->currentText()).toStdString();
            dirImage += "/Prueba/No_Persona/neg";
            dirImage += std::to_string(i);
            dirImage += ".jpg";
            command += dirImage;
            bancoUtilizado += (ui->bancosTrain->currentText()).toStdString();
            command += bancoUtilizado;

            fprintf(Fichero, "%s\n", "no persona"); //Escribe la clase real

            std::cout << "Prediccion: " << i << endl;

            system(command.c_str());
        }
    }
    fclose(Fichero);

    std::string dirTruth = " --dir1 ";
    std::string dirPredict = " --dir2 ";

    filename = "C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/ModelosClasificacion/Prueba/ANN/reporteErrores.py";
    command = "python ";

    command += filename;

    dirTruth += "C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/ModelosClasificacion/Prueba/ANN/GroundTruth.txt";
    command += dirTruth;
    dirPredict += "C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/ModelosClasificacion/Prueba/ANN/Predicciones.txt";
    command += dirPredict;

    std::cout << endl << endl << "REPORTE DE ERRORRES" << endl << endl;
    system(command.c_str());
}

void Detector_Personas::TestCNN(){

    std::string filename;
    std::string command;
    std::string dirimag;
    std::string outGraph;
    std::string outLabels;

    remove("C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/ModelosClasificacion/Prueba/CNN/Predicciones.txt");

    //CREACIÓN DEL FICHERO PARA MIRAR LOS ERRORES
    FILE *Fichero;
    Fichero = fopen("C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/ModelosClasificacion/Prueba/CNN/GroundTruth.txt", "w");
    if (Fichero == NULL){
        printf("\nError al crear el archivo de texto GroundTruth.txt... \n");
    }
    else{
        std::cout << "INICIO IMAGENES POSITIVAS" << endl << endl;

        for (int i = 1; i < (ui->posTest->toPlainText()).toInt()+1; i++){
            filename = "C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/ModelosClasificacion/Prueba/CNN/testingCNN.py";
            command = "python ";
            dirimag = " --image ";
            outGraph = " --graph ";
            outLabels = " --labels ";

            command += filename;

            outGraph += DIRECTORIO_GUARDAR_MODELO;
            outGraph += "CNN/";
            outGraph += (ui->bancosTrain->currentText()).toStdString();
            outGraph += "/output_graph.pb";
            command += outGraph;

            outLabels += DIRECTORIO_GUARDAR_MODELO;
            outLabels += "CNN/";
            outLabels += (ui->bancosTrain->currentText()).toStdString();
            outLabels += "/output_labels.txt";
            command += outLabels;

            dirimag += DIR_BANCO_IMAG;
            dirimag += (ui->bancosTest->currentText()).toStdString();
            dirimag += "/Prueba/Persona/Pos";
            dirimag += std::to_string(i);
            dirimag += ".jpg";
            command += dirimag;

            fprintf(Fichero, "%s\n", "persona"); //Escribe la clase real

            std::cout << "Prediccion: " << i << endl;

            system(command.c_str());
        }

        std::cout << endl << endl <<"INICIO IMAGENES NEGATIVAS" << endl << endl;

        for (int i = 1; i < (ui->negTest->toPlainText()).toInt()+1; i++){
            filename = "C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/ModelosClasificacion/Prueba/CNN/testingCNN.py";
            command = "python ";
            dirimag = " --image ";
            outGraph = " --graph ";
            outLabels = " --labels ";

            command += filename;

            outGraph += DIRECTORIO_GUARDAR_MODELO;
            outGraph += "CNN/";
            outGraph += (ui->bancosTrain->currentText()).toStdString();
            outGraph += "/output_graph.pb";
            command += outGraph;

            outLabels += DIRECTORIO_GUARDAR_MODELO;
            outLabels += "CNN/";
            outLabels += (ui->bancosTrain->currentText()).toStdString();
            outLabels += "/output_labels.txt";
            command += outLabels;

            dirimag += DIR_BANCO_IMAG;
            dirimag += (ui->bancosTest->currentText()).toStdString();
            dirimag += "/Prueba/No_Persona/neg";
            dirimag += std::to_string(i);
            dirimag += ".jpg";
            command += dirimag;

            fprintf(Fichero, "%s\n", "no persona"); //Escribe la clase real

            std::cout << "Prediccion: " << i << endl;

            system(command.c_str());
        }
    }
    fclose(Fichero);

    std::string dirTruth = " --dir1 ";
    std::string dirPredict = " --dir2 ";

    filename = "C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/ModelosClasificacion/Prueba/CNN/reporteErrores.py";
    command = "python ";

    command += filename;

    dirTruth += "C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/ModelosClasificacion/Prueba/CNN/GroundTruth.txt";
    command += dirTruth;
    dirPredict += "C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/ModelosClasificacion/Prueba/CNN/Predicciones.txt";
    command += dirPredict;

    std::cout << endl << endl << "REPORTE DE ERRORRES" << endl << endl;
    system(command.c_str());

}

void Detector_Personas::BarridoBordesRegiones(){
    tipoBarrido=1;

    cv::Mat frameBordes= frameCargado.clone();

    medianBlur(frameBordes, frameBordes, 7);

    //Elimina el directorio con candidatos de la escena anterior
    std::string Path = PATH_CANDIDATOS;
    std::string folderRemoveCommand = "rmdir /S /Q ";
    folderRemoveCommand += Path;
    system(folderRemoveCommand.c_str());

    //Elimina el directorio de resultados de la escena anterior
    Path = PATH_RESULTADOSBORDES;
    folderRemoveCommand = "rmdir /S /Q ";
    folderRemoveCommand += Path;
    system(folderRemoveCommand.c_str());

    //Elimina el fichero con la información sobre las coordenadas de los candidatos
    folderRemoveCommand = PATH_DESCRIPCIONBARRIDOS;
    folderRemoveCommand += "DescripcionBarrido.txt";
    remove(folderRemoveCommand.c_str());

    //Creación del directorio para guardar ventanas
    std::string folderName = PATH_CANDIDATOS;
    std::string folderCreateCommand = "mkdir ";
    folderCreateCommand += folderName;
    system(folderCreateCommand.c_str());

    folderName = PATH_RESULTADOSBORDES;
    folderCreateCommand = "mkdir ";
    folderCreateCommand += folderName;
    system(folderCreateCommand.c_str());

    //Parametros con el tamaño real de la imagen
    int Real_Columnas = (ui->anchoEscena->toPlainText()).toInt();
    int Real_Filas = (ui->altoEscena->toPlainText()).toInt();

    //Parametros para la ventana
    int windows_n_cols = (ui->anchoVentana->toPlainText()).toInt();
    int windows_n_rows = (ui->altoVentana->toPlainText()).toInt(); //Siempre debe ser el doble del ancho
    int StepSlide = (ui->stride->toPlainText()).toInt();
    float Scale = (ui->factorVentana->toPlainText()).toFloat();
    int Regla;

    int NumeroNombre = 1;

    //Lectura de la escena
    cv::Mat image_f, edges_canny, bw, gray;
    //frameCargado = cv::imread(strFileName.toStdString(),cv::IMREAD_COLOR);        // open image

    //Creación de las ventanas para visualizar
    //cv::namedWindow("RF edges", 1);
    //namedWindow("Canny edges", 1);

    //Extracción de bordes
    frameBordes.convertTo(image_f, cv::DataType<float>::type, 1 / 255.0);
    cv::Mat edges(image_f.size(), image_f.type());
    cv::Ptr<cv::ximgproc::StructuredEdgeDetection> pDollar = cv::ximgproc::createStructuredEdgeDetection("C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/TecnicasBarrido/BarridoBordes/model.yml");
    pDollar->detectEdges(image_f, edges);
    //edges.convertTo(edges, CV_8U, 255.0);
    edges.convertTo(edges, CV_8U, 512.0);
    cv::Mat edges_color;
    //applyColorMap(edges, edges_color, COLORMAP_JET);
    cv::applyColorMap(edges, edges_color, cv::COLORMAP_COOL);
    cv::Canny(edges, bw, 100, 200, 3);
    cv::threshold(edges, bw, 64, 255, CV_THRESH_BINARY);
    cv::cvtColor(frameBordes, gray, cv::COLOR_BGR2GRAY);
    cv::Canny(gray, edges_canny, 60, 180, 3);
    //cv::imshow("RF edges", edges_color);
    //cv::imshow("RF edges", bw);
    //cv::imshow("Canny edges", edges_canny);

    //CREACIÓN DEL FICHERO PARA BLOQUE POST-PROCESAMIENTO
    FILE *Fichero;
    std::string aux = PATH_DESCRIPCIONBARRIDOS;
    aux += "DescripcionBarrido.txt";
    Fichero = fopen(aux.c_str(), "w");
    if (Fichero == NULL){
        printf("\nError al crear el archivo de texto DescripcionBarridoBordes.txt... \n");
    }
    else{
        fprintf(Fichero, "%d\n", windows_n_cols); //Escribe el ancho de la ventana
        fprintf(Fichero, "%d\n", windows_n_rows); //Escribe el alto de la ventana
        fprintf(Fichero, "%c\n", '*'); //Indica el final info sobre ventana

        //Barrido completo de la escena
        for (int Factor = 0; Factor < (ui->escalas->toPlainText()).toInt(); Factor++) {

            if(Factor==0){
                Regla=900000;
            }
            else if(Factor==1){
                Regla=1900000;
            }
            else if(Factor==2){
                Regla=5000000;
            }

            float Escala = pow(Scale, Factor);
            printf("Factor: %f \n", Escala);

            //cv::resize(frameBordes, frameBordes, cv::Size(Real_Columnas / Escala, Real_Filas / Escala));
            //cv::resize(bw, bw, cv::Size(Real_Columnas / Escala, Real_Filas / Escala));

            cv::Mat DrawResultGrid = bw.clone();

            if(Factor==0){
                for (float row = 0; row <= 510 - windows_n_rows*Escala; row += StepSlide)
                {
                    for (float col = 0; col <= bw.cols - windows_n_cols*Escala; col += StepSlide)
                    {
                        // resulting window
                        cv::Rect windows(col, row, windows_n_cols*Escala, windows_n_rows*Escala);

                        cv::Mat DrawResultHere = bw.clone();

                        // Draw only rectangle
                        cv::rectangle(DrawResultHere, windows, cv::Scalar(255), 1, 8, 0);
                        // Draw grid
                        cv::rectangle(DrawResultGrid, windows, cv::Scalar(255), 1, 8, 0);

                        // Show  rectangle
                        cv::namedWindow("Ventana en movimiento", cv::WINDOW_AUTOSIZE);
                        cv::imshow("Ventana en movimiento", DrawResultHere);

                        // Show grid
                        cv::imwrite(cv::format("C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/TecnicasBarrido/BarridoBordes/Resultados/Barrido factor %f.JPG", Escala), DrawResultGrid);

                        //Ver ROI
                        cv::Mat Roi = bw(windows);
                        //namedWindow("Ventana segmentada", WINDOW_AUTOSIZE);
                        //imshow("Ventana segmentada", Roi);
                        cv::waitKey(1);
                        int Suma = (int)cv::sum(Roi)[0];
                        std::cout << Suma << "\n";
                        if (Suma > Regla){
                            cv::Mat RoiColor = frameBordes(windows);
                            cv::resize(RoiColor, RoiColor, cv::Size(64, 128));
                            //imwrite(cv::format("C:\\Users\\user\\Desktop\\Recursos\\TecnicasBarrido\\Candidatos\\%d-%d.JPG", NumeroNombre, Suma), RoiColor);
                            cv::imwrite(cv::format("C:\\Deteccion_Conteo_Personas_Escenas_PUJ-1703\\Recursos\\TecnicasBarrido\\Candidatos\\%d.JPG", NumeroNombre), RoiColor);
                            NumeroNombre++;

                            //INFORMACIÓN COORDENADAS DE CANDIDATO ACTUAL QUE CUMPLE LA REGLA
                            fprintf(Fichero, "%f\n", row/Escala); //Escribe coordenada en x
                            fprintf(Fichero, "%f\n", col/Escala); //Escribe coordenada en y
                        }
                    }
                }
            }
            else if(Factor==1){
                for (float row = 110; row <= 640 - windows_n_rows*Escala; row += StepSlide)
                {
                    for (float col = 0; col <= bw.cols - windows_n_cols*Escala; col += StepSlide)
                    {
                        // resulting window
                        cv::Rect windows(col, row, windows_n_cols*Escala, windows_n_rows*Escala);

                        cv::Mat DrawResultHere = bw.clone();

                        // Draw only rectangle
                        cv::rectangle(DrawResultHere, windows, cv::Scalar(255), 1, 8, 0);
                        // Draw grid
                        cv::rectangle(DrawResultGrid, windows, cv::Scalar(255), 1, 8, 0);

                        // Show  rectangle
                        cv::namedWindow("Ventana en movimiento", cv::WINDOW_AUTOSIZE);
                        cv::imshow("Ventana en movimiento", DrawResultHere);

                        // Show grid
                        cv::imwrite(cv::format("C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/TecnicasBarrido/BarridoBordes/Resultados/Barrido factor %f.JPG", Escala), DrawResultGrid);

                        //Ver ROI
                        cv::Mat Roi = bw(windows);
                        //namedWindow("Ventana segmentada", WINDOW_AUTOSIZE);
                        //imshow("Ventana segmentada", Roi);
                        cv::waitKey(1);
                        int Suma = (int)cv::sum(Roi)[0];
                        std::cout << Suma << "\n";
                        if (Suma > Regla){
                            cv::Mat RoiColor = frameBordes(windows);
                            cv::resize(RoiColor, RoiColor, cv::Size(64, 128));
                            //imwrite(cv::format("C:\\Users\\user\\Desktop\\Recursos\\TecnicasBarrido\\Candidatos\\%d-%d.JPG", NumeroNombre, Suma), RoiColor);
                            cv::imwrite(cv::format("C:\\Deteccion_Conteo_Personas_Escenas_PUJ-1703\\Recursos\\TecnicasBarrido\\Candidatos\\%d.JPG", NumeroNombre), RoiColor);
                            NumeroNombre++;

                            //INFORMACIÓN COORDENADAS DE CANDIDATO ACTUAL QUE CUMPLE LA REGLA
                            fprintf(Fichero, "%f\n", row/Escala); //Escribe coordenada en x
                            fprintf(Fichero, "%f\n", col/Escala); //Escribe coordenada en y
                        }
                    }
                }
            }
            else if(Factor==2){
                for (float row = 180; row <= 1080 - windows_n_rows*Escala; row += StepSlide)
                {
                    for (float col = 0; col <= bw.cols - windows_n_cols*Escala; col += StepSlide)
                    {
                        // resulting window
                        cv::Rect windows(col, row, windows_n_cols*Escala, windows_n_rows*Escala);

                        cv::Mat DrawResultHere = bw.clone();

                        // Draw only rectangle
                        cv::rectangle(DrawResultHere, windows, cv::Scalar(255), 1, 8, 0);
                        // Draw grid
                        cv::rectangle(DrawResultGrid, windows, cv::Scalar(255), 1, 8, 0);

                        // Show  rectangle
                        cv::namedWindow("Ventana en movimiento", cv::WINDOW_AUTOSIZE);
                        cv::imshow("Ventana en movimiento", DrawResultHere);

                        // Show grid
                        cv::imwrite(cv::format("C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/TecnicasBarrido/BarridoBordes/Resultados/Barrido factor %f.JPG", Escala), DrawResultGrid);

                        //Ver ROI
                        cv::Mat Roi = bw(windows);
                        //namedWindow("Ventana segmentada", WINDOW_AUTOSIZE);
                        //imshow("Ventana segmentada", Roi);
                        cv::waitKey(1);
                        int Suma = (int)cv::sum(Roi)[0];
                        std::cout << Suma << "\n";
                        if (Suma > Regla){
                            cv::Mat RoiColor = frameBordes(windows);
                            cv::resize(RoiColor, RoiColor, cv::Size(64, 128));
                            //imwrite(cv::format("C:\\Users\\user\\Desktop\\Recursos\\TecnicasBarrido\\Candidatos\\%d-%d.JPG", NumeroNombre, Suma), RoiColor);
                            cv::imwrite(cv::format("C:\\Deteccion_Conteo_Personas_Escenas_PUJ-1703\\Recursos\\TecnicasBarrido\\Candidatos\\%d.JPG", NumeroNombre), RoiColor);
                            NumeroNombre++;

                            //INFORMACIÓN COORDENADAS DE CANDIDATO ACTUAL QUE CUMPLE LA REGLA
                            fprintf(Fichero, "%f\n", row/Escala); //Escribe coordenada en x
                            fprintf(Fichero, "%f\n", col/Escala); //Escribe coordenada en y
                        }
                    }
                }
            }

            fprintf(Fichero, "%c\n", '*'); //Indica el final de candidatos en una escala
        }
        fprintf(Fichero, "%c\n", '-'); //Indica el final de todo el proceso de barrido
    }
    fclose(Fichero);

    FILE *Fichero2;
    aux = PATH_DESCRIPCIONBARRIDOS;
    aux += "NumeroCandidatos.txt";
    Fichero2 = fopen(aux.c_str(), "w");
    if (Fichero2 == NULL){
        printf("\nError al crear el archivo de texto NumeroCandidatos.txt... \n");
    }
    else{
        fprintf(Fichero2, "%d", (NumeroNombre-1)); //Indica el final de todo el proceso de barrido
    }
    fclose(Fichero2);
}

void Detector_Personas::BarridoExhaustivoRegiones(){
    tipoBarrido=0;

    cv::Mat frameExhaustivo = frameCargado.clone();

    medianBlur(frameExhaustivo, frameExhaustivo, 7);

    //Elimina el directorio con candidatos de la escena anterior
    std::string Path = PATH_CANDIDATOS;
    std::string folderRemoveCommand = "rmdir /S /Q ";
    folderRemoveCommand += Path;
    system(folderRemoveCommand.c_str());

    //Elimina el directorio de resultados de la escena anterior
    Path = PATH_RESULTADOSEXHAUS;
    folderRemoveCommand = "rmdir /S /Q ";
    folderRemoveCommand += Path;
    system(folderRemoveCommand.c_str());

    //Elimina el fichero con la información sobre las coordenadas de los candidatos
    folderRemoveCommand = PATH_DESCRIPCIONBARRIDOS;
    folderRemoveCommand += "DescripcionBarrido.txt";
    remove(folderRemoveCommand.c_str());

    //Creación del directorio para guardar ventanas
    std::string folderName = PATH_CANDIDATOS;
    std::string folderCreateCommand = "mkdir ";
    folderCreateCommand += folderName;
    system(folderCreateCommand.c_str());

    folderName = PATH_RESULTADOSEXHAUS;
    folderCreateCommand = "mkdir ";
    folderCreateCommand += folderName;
    system(folderCreateCommand.c_str());

    //Parametros con el tamaño real de la imagen
    int Real_Columnas = (ui->anchoEscena->toPlainText()).toInt();
    int Real_Filas = (ui->altoEscena->toPlainText()).toInt();

    //Parametros para la ventana
    int windows_n_cols = (ui->anchoVentana->toPlainText()).toInt();
    int windows_n_rows = (ui->altoVentana->toPlainText()).toInt(); //Siempre debe ser el doble del ancho
    int StepSlide = (ui->stride->toPlainText()).toInt();
    float Scale = (ui->factorVentana->toPlainText()).toFloat();

    int NumeroNombre = 1;
    int ContadorCandidatos = 1;

    //CREACIÓN DEL FICHERO PARA BLOQUE POST-PROCESAMIENTO
    FILE *Fichero;
    std::string aux = PATH_DESCRIPCIONBARRIDOS;
    aux += "DescripcionBarrido.txt";
    Fichero = fopen(aux.c_str(), "w");
    if (Fichero == NULL){
        printf("\nError al crear el archivo de texto DescripcionBarridoExhaustivo.txt... \n");
    }
    else{
        fprintf(Fichero, "%d\n", windows_n_cols); //Escribe el ancho de la ventana
        fprintf(Fichero, "%d\n", windows_n_rows); //Escribe el alto de la ventana
        fprintf(Fichero, "%c\n", '*'); //Indica el final info sobre ventana

        for (int Factor = 0; Factor < (ui->escalas->toPlainText()).toInt(); Factor++) {

            float Escala = pow(Scale, Factor);
            printf("Factor: %f \n", Escala);

            //cv::resize(frameExhaustivo, frameExhaustivo, cv::Size(Real_Columnas / Escala, Real_Filas / Escala));
            //namedWindow(cv::format("Escala %d", Escala), WINDOW_AUTOSIZE);
            //imshow(cv::format("Escala %d", Escala), frameExhaustivo);
            //cv::imwrite(cv::format("C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/TecnicasBarrido/BarridoVentana/Resultados/Factor %f.JPG", Escala), frameExhaustivo);

            cv::Mat DrawResultGrid = frameExhaustivo.clone();

            if(Factor==0){
                for (float row = 0; row <= 300 - windows_n_rows*Escala; row += StepSlide)
                {
                    for (float col = 0; col <= frameExhaustivo.cols - windows_n_cols*Escala; col += StepSlide)
                    {
                        //INFORMACIÓN COORDENADAS DE CANDIDATO ACTUAL
                        fprintf(Fichero, "%f\n", row); //Escribe coordenada en x
                        fprintf(Fichero, "%f\n", col); //Escribe coordenada en y

                        // resulting window
                        cv::Rect windows(col, row, windows_n_cols*Escala, windows_n_rows*Escala);

                        cv::Mat DrawResultHere = frameExhaustivo.clone();

                        // Draw only rectangle
                        cv::rectangle(DrawResultHere, windows, cv::Scalar(255), 1, 8, 0);
                        // Draw grid
                        cv::rectangle(DrawResultGrid, windows, cv::Scalar(255), 1, 8, 0);

                        // Show  rectangle
                        namedWindow("Ventana en movimiento", cv::WINDOW_AUTOSIZE);
                        cv::imshow("Ventana en movimiento", DrawResultHere);

                        // Show grid
                        //namedWindow("Barrido realizado", WINDOW_AUTOSIZE);
                        //imshow("Barrido realizado", DrawResultGrid);
                        cv::imwrite(cv::format("C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/TecnicasBarrido/BarridoVentana/Resultados/Barrido factor %f.JPG", Escala), DrawResultGrid);

                        //Ver ROI
                        cv::Mat Roi = frameExhaustivo(windows);
                        //namedWindow("Ventana segmentada", WINDOW_AUTOSIZE);
                        //imshow("Ventana segmentada", Roi);
                        cv::waitKey(2);
                        cv::resize(Roi, Roi, cv::Size(64, 128));
                        std::string carpeta = PATH_CANDIDATOS;
                        carpeta += "\\%d.JPG";
                        cv::imwrite(cv::format("C:\\Deteccion_Conteo_Personas_Escenas_PUJ-1703\\Recursos\\TecnicasBarrido\\Candidatos\\%d.JPG", NumeroNombre), Roi);
                        NumeroNombre++;
                        ContadorCandidatos++;
                    }
                }

            }
            else if(Factor==1){
                for (float row = 100; row <= 560 - windows_n_rows*Escala; row += StepSlide)
                {
                    for (float col = 0; col <= frameExhaustivo.cols - windows_n_cols*Escala; col += StepSlide)
                    {
                        //INFORMACIÓN COORDENADAS DE CANDIDATO ACTUAL
                        fprintf(Fichero, "%f\n", row/Escala); //Escribe coordenada en x
                        fprintf(Fichero, "%f\n", col)/Escala; //Escribe coordenada en y

                        // resulting window
                        cv::Rect windows(col, row, windows_n_cols*Escala, windows_n_rows*Escala);

                        cv::Mat DrawResultHere = frameExhaustivo.clone();

                        // Draw only rectangle
                        cv::rectangle(DrawResultHere, windows, cv::Scalar(255), 1, 8, 0);
                        // Draw grid
                        cv::rectangle(DrawResultGrid, windows, cv::Scalar(255), 1, 8, 0);

                        // Show  rectangle
                        namedWindow("Ventana en movimiento", cv::WINDOW_AUTOSIZE);
                        cv::imshow("Ventana en movimiento", DrawResultHere);

                        // Show grid
                        //namedWindow("Barrido realizado", WINDOW_AUTOSIZE);
                        //imshow("Barrido realizado", DrawResultGrid);
                        cv::imwrite(cv::format("C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/TecnicasBarrido/BarridoVentana/Resultados/Barrido factor %f.JPG", Escala), DrawResultGrid);

                        //Ver ROI
                        cv::Mat Roi = frameExhaustivo(windows);
                        //namedWindow("Ventana segmentada", WINDOW_AUTOSIZE);
                        //imshow("Ventana segmentada", Roi);
                        cv::waitKey(2);
                        cv::resize(Roi, Roi, cv::Size(64, 128));
                        std::string carpeta = PATH_CANDIDATOS;
                        carpeta += "\\%d.JPG";
                        cv::imwrite(cv::format("C:\\Deteccion_Conteo_Personas_Escenas_PUJ-1703\\Recursos\\TecnicasBarrido\\Candidatos\\%d.JPG", NumeroNombre), Roi);
                        NumeroNombre++;
                        ContadorCandidatos++;
                    }
                }
            }
            else if(Factor==2){
                for (float row = 200; row <= 1080 - windows_n_rows*Escala; row += StepSlide)
                {
                    for (float col = 0; col <= frameExhaustivo.cols - windows_n_cols*Escala; col += StepSlide)
                    {
                        //INFORMACIÓN COORDENADAS DE CANDIDATO ACTUAL
                        fprintf(Fichero, "%f\n", row/Escala); //Escribe coordenada en x
                        fprintf(Fichero, "%f\n", col/Escala); //Escribe coordenada en y

                        // resulting window
                        cv::Rect windows(col, row, windows_n_cols*Escala, windows_n_rows*Escala);

                        cv::Mat DrawResultHere = frameExhaustivo.clone();

                        // Draw only rectangle
                        cv::rectangle(DrawResultHere, windows, cv::Scalar(255), 1, 8, 0);
                        // Draw grid
                        cv::rectangle(DrawResultGrid, windows, cv::Scalar(255), 1, 8, 0);

                        // Show  rectangle
                        namedWindow("Ventana en movimiento", cv::WINDOW_AUTOSIZE);
                        cv::imshow("Ventana en movimiento", DrawResultHere);

                        // Show grid
                        //namedWindow("Barrido realizado", WINDOW_AUTOSIZE);
                        //imshow("Barrido realizado", DrawResultGrid);
                        cv::imwrite(cv::format("C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/TecnicasBarrido/BarridoVentana/Resultados/Barrido factor %f.JPG", Escala), DrawResultGrid);

                        //Ver ROI
                        cv::Mat Roi = frameExhaustivo(windows);
                        //namedWindow("Ventana segmentada", WINDOW_AUTOSIZE);
                        //imshow("Ventana segmentada", Roi);
                        cv::waitKey(2);
                        cv::resize(Roi, Roi, cv::Size(64, 128));
                        std::string carpeta = PATH_CANDIDATOS;
                        carpeta += "\\%d.JPG";
                        cv::imwrite(cv::format("C:\\Deteccion_Conteo_Personas_Escenas_PUJ-1703\\Recursos\\TecnicasBarrido\\Candidatos\\%d.JPG", NumeroNombre), Roi);
                        NumeroNombre++;
                        ContadorCandidatos++;
                    }
                }
            }

            fprintf(Fichero, "%c\n", '*'); //Indica el final de candidatos en una escala
        }
        fprintf(Fichero, "%c", '-'); //Indica el final de todo el proceso de barrido
    }
    fclose(Fichero);

    FILE *Fichero2;
    aux = PATH_DESCRIPCIONBARRIDOS;
    aux += "NumeroCandidatos.txt";
    Fichero2 = fopen(aux.c_str(), "w");
    if (Fichero2 == NULL){
        printf("\nError al crear el archivo de texto NumeroCandidatos.txt... \n");
    }
    else{
        fprintf(Fichero2, "%d", (NumeroNombre - 1)); //Indica el final de todo el proceso de barrido
    }
    fclose(Fichero2);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Detector_Personas::on_ListaBancos_currentIndexChanged(int index){

    std::string EjemPosDir = DIR_BANCO_IMAG;
    std::string EjemNegDir = DIR_BANCO_IMAG;

    EjemPosDir += ui->ListaBancos->currentText().toStdString();
    EjemPosDir += "/Entrenamiento/Persona/Pos20.jpg";
    EjemPos = cv::imread(EjemPosDir);        // open image

    EjemNegDir += ui->ListaBancos->currentText().toStdString();
    EjemNegDir += "/Entrenamiento/No_Persona/neg20.jpg";
    EjemNeg = cv::imread(EjemNegDir);        // open image

    if (EjemPos.empty()||EjemNeg.empty()) {									// if unable to open image
        QString Mensaje = "Error al cargar las imágenes de muestra de ";
        Mensaje += ui->ListaBancos->currentText();
        QMessageBox::information(this, "Error", Mensaje );        // show error message
    }
    else{
        cv::resize(EjemPos, EjemPos, cv::Size(200,400));
        cv::resize(EjemNeg, EjemNeg,  cv::Size(200,400));
        QImage qEjemPosDir = convertOpenCVMatToQtQImage(EjemPos);         // convert original and Canny images to QImage
        ui->EjemPos->setPixmap(QPixmap::fromImage(qEjemPosDir));   // show original and Canny images on labels
        QImage qEjemNegDir = convertOpenCVMatToQtQImage(EjemNeg);         // convert original and Canny images to QImage
        ui->EjemNeg->setPixmap(QPixmap::fromImage(qEjemNegDir));   // show original and Canny images on labels
    }
}

void Detector_Personas::on_bancosTrain_currentIndexChanged(int index){

    std::string EjemPosDir = DIR_BANCO_IMAG;
    std::string EjemNegDir = DIR_BANCO_IMAG;

    EjemPosDir += ui->bancosTrain->currentText().toStdString();
    EjemPosDir += "/Entrenamiento/Persona/Pos20.jpg";
    EjemPos = cv::imread(EjemPosDir);        // open image

    EjemNegDir += ui->bancosTrain->currentText().toStdString();
    EjemNegDir += "/Entrenamiento/No_Persona/neg20.jpg";
    EjemNeg = cv::imread(EjemNegDir);        // open image

    if (EjemPos.empty()||EjemNeg.empty()) {									// if unable to open image
        QString Mensaje = "Error al cargar las imágenes de muestra de ";
        Mensaje += ui->bancosTrain->currentText();
        QMessageBox::information(this, "Error", Mensaje );        // show error message
    }
    else{
        cv::resize(EjemPos, EjemPos, cv::Size(200,400));
        cv::resize(EjemNeg, EjemNeg,  cv::Size(200,400));
        QImage qEjemPosDir = convertOpenCVMatToQtQImage(EjemPos);         // convert original and Canny images to QImage
        ui->EjemPosTrain->setPixmap(QPixmap::fromImage(qEjemPosDir));   // show original and Canny images on labels
        QImage qEjemNegDir = convertOpenCVMatToQtQImage(EjemNeg);         // convert original and Canny images to QImage
        ui->EjemNegTrain->setPixmap(QPixmap::fromImage(qEjemNegDir));   // show original and Canny images on labels
    }
}

void Detector_Personas::on_bancosTest_currentIndexChanged(int index){
    std::string EjemPosDir = DIR_BANCO_IMAG;
    std::string EjemNegDir = DIR_BANCO_IMAG;

    EjemPosDir += ui->bancosTest->currentText().toStdString();
    EjemPosDir += "/Prueba/Persona/Pos5.jpg";
    EjemPos = cv::imread(EjemPosDir);        // open image

    EjemNegDir += ui->bancosTest->currentText().toStdString();
    EjemNegDir += "/Prueba/No_Persona/neg5.jpg";
    EjemNeg = cv::imread(EjemNegDir);        // open image

    if (EjemPos.empty()||EjemNeg.empty()) {									// if unable to open image
        QString Mensaje = "Error al cargar las imágenes de muestra de ";
        Mensaje += ui->bancosTest->currentText();
        QMessageBox::information(this, "Error", Mensaje );        // show error message
    }
    else{
        cv::resize(EjemPos, EjemPos, cv::Size(200,400));
        cv::resize(EjemNeg, EjemNeg,  cv::Size(200,400));
        QImage qEjemPosDir = convertOpenCVMatToQtQImage(EjemPos);         // convert original and Canny images to QImage
        ui->EjemPosTest->setPixmap(QPixmap::fromImage(qEjemPosDir));   // show original and Canny images on labels
        QImage qEjemNegDir = convertOpenCVMatToQtQImage(EjemNeg);         // convert original and Canny images to QImage
        ui->EjemNegTest->setPixmap(QPixmap::fromImage(qEjemNegDir));   // show original and Canny images on labels
    }
}

void Detector_Personas::on_abrirFrame_clicked(){
    strFileName = QFileDialog::getOpenFileName(this, tr("Elegir escena"),
                                               "C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/BancoEscenas",
                                               tr("Images (*.png *.xpm *.jpg)"));       // bring up open file dialog

    if(strFileName == "") {                                     // if file was not chosen
        ui->frameElegido->setText("Error: No se ha elegido el frame.");          // update label
        return;                                                 // and exit function
    }

    frameCargado = cv::imread(strFileName.toStdString(),cv::IMREAD_COLOR);        // open image

    if (frameCargado.empty()) {									// if unable to open image
        QMessageBox::information(this, "Error", "No se pudo leer la imagen" );        // show error message
        return;                                                             // and exit function
    }

    std::size_t pos = (strFileName.toStdString()).find("BancoEscenas");
    std::string str3 = (strFileName.toStdString()).substr (pos);     // get from "live" to the end
    pos = (str3).find("/");
    str3 = (str3).substr (pos+1);     // Otras/1.jpg
    pos = (str3).find("/");
    str3 = (str3).substr (pos+1);     // 1.jpg

    ui->frameElegido->setText(QString::fromStdString(str3));                // update label with file name


    cv::Mat frameCargadoCopia = frameCargado.clone();

    cv::resize(frameCargadoCopia, frameCargadoCopia, cv::Size(820,410));

    QImage qframeCargado = convertOpenCVMatToQtQImage(frameCargadoCopia);         // convert original and Canny images to QImage

    ui->Frame->setPixmap(QPixmap::fromImage(qframeCargado));   // show original and Canny images on labels

}

void Detector_Personas::on_btnEntrenar_clicked(){

    std::clock_t start;
    double duration;

    start = std::clock();

    if(ui->ListaModelos->currentIndex()==0 ){ //HoG con RF

        std::string Clasificador = "HoGRF";

        std::vector< cv::Mat > ImagenesPositivas;
        std::vector< cv::Mat > ImagenesNegativas;
        std::vector< cv::Mat > GradientesImagenes;
        std::vector< int > Etiquetas;

        std::string dirImagPos = DIR_BANCOS_LECTURA;
        std::string dirImagNeg = DIR_BANCOS_LECTURA;

        dirImagPos += ui->ListaBancos->currentText().toStdString();
        dirImagPos += "\\Entrenamiento\\Persona";

        dirImagNeg += ui->ListaBancos->currentText().toStdString();
        dirImagNeg += "\\Entrenamiento\\No_Persona";

        std::cout << "INICIO PROCESO EXTRACCION DE CARACTERISTICAS ENTRENAMIENTO... \n \n";

        std::cout << "1. Lectura imagenes positivas y asignacion de label (+1) \n \n" ;

        Detector_Personas::CargarImagenes(dirImagPos, ImagenesPositivas,1,0);
        Etiquetas.assign(ImagenesPositivas.size(), +1);

        std::cout << "2. Lectura imagenes negativas y asignacion de label (-1) \n \n ";
        Detector_Personas::CargarImagenes(dirImagNeg, ImagenesNegativas,0,0);
        Etiquetas.insert(Etiquetas.end(), ImagenesNegativas.size(), -1);

        std::cout << "3. Extraccion caracteristicas (HoG) a las imagenes positivas (visualizacion) \n" ;

        Detector_Personas::CaracteristicasHoG(ImagenesPositivas, GradientesImagenes, TAMANO_IMAGEN);

        std::cout << "4. Extraccion caracteristicas (HoG) a las imagenes negativas (visualizacion) \n";
        Detector_Personas::CaracteristicasHoG(ImagenesNegativas, GradientesImagenes, TAMANO_IMAGEN);

        std::cout << "5. Creacion ficheros con descriptores y labels \n";
        std::cout << "Tamaño matriz caracteristicas Filas(numero de imagenes): " << GradientesImagenes.size() << " y Columnas(numero de caracteristicas): " << GradientesImagenes[0].rows << "\n \n";

        int numImagenes = GradientesImagenes.size();

        Detector_Personas::CreacionFicherosPython(GradientesImagenes, Etiquetas, "Entrenamiento",Clasificador);

        Detector_Personas::LlamarPythonHOGRF(numImagenes,"Entrenamiento",Clasificador);

        duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

        ui->TiempoEntrenamiento->setText(QString::number(duration));
    }

    else if(ui->ListaModelos->currentIndex()==1 ){ //HoG con SVM

        std::string Clasificador = "HoGSVM";

        std::vector< cv::Mat > ImagenesPositivas;
        std::vector< cv::Mat > ImagenesNegativas;
        std::vector< cv::Mat > GradientesImagenes;
        std::vector< int > Etiquetas;

        std::string dirImagPos = DIR_BANCOS_LECTURA;
        std::string dirImagNeg = DIR_BANCOS_LECTURA;

        dirImagPos += ui->ListaBancos->currentText().toStdString();
        dirImagPos += "\\Entrenamiento\\Persona";

        dirImagNeg += ui->ListaBancos->currentText().toStdString();
        dirImagNeg += "\\Entrenamiento\\No_Persona";

        std::cout << "INICIO PROCESO EXTRACCION DE CARACTERISTICAS ENTRENAMIENTO... \n \n";

        std::cout << "1. Lectura imagenes positivas y asignacion de label (+1) \n \n" ;

        Detector_Personas::CargarImagenes(dirImagPos, ImagenesPositivas,1,0);
        Etiquetas.assign(ImagenesPositivas.size(), +1);

        std::cout << "2. Lectura imagenes negativas y asignacion de label (-1) \n \n ";
        Detector_Personas::CargarImagenes(dirImagNeg, ImagenesNegativas,0,0);
        Etiquetas.insert(Etiquetas.end(), ImagenesNegativas.size(), -1);

        std::cout << "3. Extraccion caracteristicas (HoG) a las imagenes positivas (visualizacion) \n" ;

        Detector_Personas::CaracteristicasHoG(ImagenesPositivas, GradientesImagenes, TAMANO_IMAGEN);

        std::cout << "4. Extraccion caracteristicas (HoG) a las imagenes negativas (visualizacion) \n";
        Detector_Personas::CaracteristicasHoG(ImagenesNegativas, GradientesImagenes, TAMANO_IMAGEN);

        std::cout << "5. Creacion ficheros con descriptores y labels \n";
        std::cout << "Tamaño matriz caracteristicas Filas(numero de imagenes): " << GradientesImagenes.size() << " y Columnas(numero de caracteristicas): " << GradientesImagenes[0].rows << "\n \n";

        int numImagenes = GradientesImagenes.size();

        Detector_Personas::CreacionFicherosPython(GradientesImagenes, Etiquetas, "Entrenamiento",Clasificador);

        Detector_Personas::LlamarPythonHOGSVM(numImagenes,"Entrenamiento",Clasificador);

        duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

        ui->TiempoEntrenamiento->setText(QString::number(duration));
    }

    else if(ui->ListaModelos->currentIndex()==2 ){ //SIFT FV RF

        std::string Clasificador = "SIFTFVRF";
        Detector_Personas::LlamarPythonSIFTFVRF("Entrenamiento",Clasificador);
        duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

        ui->TiempoEntrenamiento->setText(QString::number(duration));
    }

    else if(ui->ListaModelos->currentIndex()==3 ){ //SIFT FV SVM
        std::string Clasificador = "SIFTFVSVM";
        Detector_Personas::LlamarPythonSIFTFVSVM("Entrenamiento",Clasificador);
        duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

        ui->TiempoEntrenamiento->setText(QString::number(duration));
    }

    else if(ui->ListaModelos->currentIndex()==4 ){ //ANN
        std::string Clasificador = "ANN";
        Detector_Personas::LlamarPythonANN("Entrenamiento",Clasificador);
        duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

        ui->TiempoEntrenamiento->setText(QString::number(duration));
    }

    else if(ui->ListaModelos->currentIndex()==5 ){ //CNN
        std::string Clasificador = "CNN";
        Detector_Personas::LlamarPythonCNN("Entrenamiento",Clasificador);
        duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

        ui->TiempoEntrenamiento->setText(QString::number(duration));
    }
}

void Detector_Personas::on_iniciarBusqueda_clicked(){
    std::clock_t start;
    double duration;

    start = std::clock();

    if (frameCargado.empty()) {									// if unable to open image
        QMessageBox::information(this, "Error", "Primero debe elegir una escena." );        // show error message
        return;                                                             // and exit function
    }

    if(ui->ListaBarridos->currentIndex()==0 ){
        if(ui->condicional->currentIndex()==0 ){
            Detector_Personas::BarridoExhaustivoRegiones();
        }
        else{
            Detector_Personas::BarridoExhaustivo();
        }
    }
    else if(ui->ListaBarridos->currentIndex()==1 ){
        if(ui->condicional->currentIndex()==0 ){
            Detector_Personas::BarridoBordesRegiones();
        }
        else{
            Detector_Personas::BarridoBordes();
        }
    }

    if(ui->ListaModelo->currentIndex()==0 ){ //HoG con RF

        std::string Clasificador = "HoGRF";

        std::vector< cv::Mat > Candidatos;
        std::vector< cv::Mat > GradientesImagenes;

        std::cout << "INICIO PROCESO EXTRACCION DE CARACTERISTICAS A LOS CANDIDATOS DE LA ESCENA... \n \n";

        std::cout << "1. Lectura imagenes de los candidatos \n \n" ;

        Detector_Personas::CargarImagenesEscena(PATH_CANDIDATOS, Candidatos);

        std::cout << "2. Extraccion caracteristicas (HoG) a las imagenes de los candidatos (visualizacion)" << endl;

        Detector_Personas::CaracteristicasHoG(Candidatos, GradientesImagenes, TAMANO_IMAGEN);

        std::cout << "3. Creacion ficheros con descriptores \n";
        std::cout << "Tamaño matriz caracteristicas Filas(numero de imagenes): " << GradientesImagenes.size() << " y Columnas(numero de caracteristicas): " << GradientesImagenes[0].rows << "\n \n";

        Detector_Personas::CreacionFicherosPythonEscena(GradientesImagenes, "Escena",Clasificador);

        Detector_Personas::LlamarPythonHOGRFEscena("Escena",Clasificador,1,1);
    }

    else if(ui->ListaModelo->currentIndex()==1 ){ //HoG con SVM
        std::string Clasificador = "HoGSVM";

        std::vector< cv::Mat > Candidatos;
        std::vector< cv::Mat > GradientesImagenes;

        std::cout << "INICIO PROCESO EXTRACCION DE CARACTERISTICAS A LOS CANDIDATOS DE LA ESCENA... \n \n";

        std::cout << "1. Lectura imagenes de los candidatos \n \n" ;

        Detector_Personas::CargarImagenesEscena(PATH_CANDIDATOS, Candidatos);

        std::cout << "2. Extraccion caracteristicas (HoG) a las imagenes de los candidatos (visualizacion)" << endl;

        Detector_Personas::CaracteristicasHoG(Candidatos, GradientesImagenes, TAMANO_IMAGEN);

        std::cout << "3. Creacion ficheros con descriptores \n";
        std::cout << "Tamaño matriz caracteristicas Filas(numero de imagenes): " << GradientesImagenes.size() << " y Columnas(numero de caracteristicas): " << GradientesImagenes[0].rows << "\n \n";

        Detector_Personas::CreacionFicherosPythonEscena(GradientesImagenes, "Escena",Clasificador);

        Detector_Personas::LlamarPythonHOGSVMEscena("Escena",Clasificador,1,tipoBarrido);
    }

    else if(ui->ListaModelo->currentIndex()==2 ){ //SIFT FV RF
        std::string Clasificador = "SIFTFVRF";
        Detector_Personas::LlamarPythonSIFTFVRFEscena("Escena", Clasificador,1,tipoBarrido);
    }

    else if(ui->ListaModelo->currentIndex()==3 ){ //SIFT FV SVM
        std::string Clasificador = "SIFTFVSVM";
        Detector_Personas::LlamarPythonSIFTFVSVMEscena("Escena", Clasificador,1,tipoBarrido);
    }

    else if(ui->ListaModelo->currentIndex()==4 ){ //ANN
        Detector_Personas::LlamarPythonANNEscena(1);
    }

    else if(ui->ListaModelo->currentIndex()==5 ){ //CNN
        Detector_Personas::LlamarPythonCNNEscena(1);
    }

    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    ui->TiempoEscena->setText(QString::number(duration));
}

void Detector_Personas::on_iniciarReporte_clicked(){

    QString OutputFolder = QFileDialog::getExistingDirectory(0, ("Carpeta con las escenas"), "C:/Deteccion_Conteo_Personas_Escenas_PUJ-1703/Recursos/BancoEscenas"); //queda sin / al final
    std::string Escena;
    cv::Mat frameCargadoCopia;

    if(OutputFolder == "") {                                     // if file was not chosen
        ui->frameElegido->setText("Error: No se ha elegido una carpeta.");          // update label
        return;                                                 // and exit function
    }

    float NMSInicial = (ui->criterioNMS->toPlainText()).toFloat();
    float DIFInicial = (ui->criterioDif->toPlainText()).toFloat();
    float ContInicial = (ui->criterioCont->toPlainText()).toFloat();
    float PredicInicial = (ui->criterioPredic->toPlainText()).toFloat();

    for(int i=1;i<=(ui->numeroReporte->toPlainText()).toInt();i++){//Cantidad de imagenes utilizadas en el reporte

        ui->criterioNMS->setText(QString::number(NMSInicial));
        ui->criterioDif->setText(QString::number(DIFInicial));
        ui->criterioCont->setText(QString::number(ContInicial));
        ui->criterioPredic->setText(QString::number(PredicInicial));

        Escena = OutputFolder.toStdString();
        Escena += "/";
        Escena += (QString::number(i)).toStdString();
        Escena += ".jpg";

        frameCargado = cv::imread(Escena,cv::IMREAD_COLOR);        // open image

        if (frameCargado.empty()) {									// if unable to open image
            QMessageBox::information(this, "Error", "En la carpeta solo hay: " + QString::number(i-1) + " escenas.");        // show error message
            return;                                                             // and exit function
        }

        frameCargadoCopia=frameCargado.clone();

        cv::resize(frameCargadoCopia, frameCargadoCopia, cv::Size(820,410));

        QImage qframeReporte = convertOpenCVMatToQtQImage(frameCargadoCopia);         // convert original and Canny images to QImage

        ui->Frame->setPixmap(QPixmap::fromImage(qframeReporte));   // show original and Canny images on labels

        std::size_t pos = (Escena).find("BancoEscenas");
        std::string str3 = (Escena).substr (pos);     // get from "live" to the end
        pos = (str3).find("/");
        str3 = (str3).substr (pos+1);     // Otras/1.jpg
        pos = (str3).find("/");
        str3 = (str3).substr (pos+1);     // 1.jpg
        pos = (str3).find(".");

        std::string nombre = (str3).substr (0,pos);

        ui->frameElegido->setText(QString::fromStdString(str3));                // update label with file name

        if(ui->ListaBarridos->currentIndex()==0 ){
            strFileName=QString::fromStdString(Escena);
            if(ui->condicional->currentIndex()==0 ){
                Detector_Personas::BarridoExhaustivoRegiones();
            }
            else{
                Detector_Personas::BarridoExhaustivo();
            }
        }
        else if(ui->ListaBarridos->currentIndex()==1 ){
            strFileName=QString::fromStdString(Escena);
            if(ui->condicional->currentIndex()==0 ){
                Detector_Personas::BarridoBordesRegiones();
            }
            else{
                Detector_Personas::BarridoBordes();
            }
        }

        for(float j=NMSInicial;j>NMSInicial-((ui->pasoCriterios->toPlainText()).toFloat()*(ui->numeroPasos->toPlainText()).toInt());j=j-(ui->pasoCriterios->toPlainText()).toFloat()){

            ui->criterioNMS->setText(QString::number(j));

            for(float k=DIFInicial;k>DIFInicial-(ui->pasoCriterios2->toPlainText()).toFloat()*(ui->numeroPasos2->toPlainText()).toInt();k=k-(ui->pasoCriterios2->toPlainText()).toFloat()){

                ui->criterioDif->setText(QString::number(k));

                for(float l=ContInicial;l>ContInicial-(ui->pasoCriterios3->toPlainText()).toFloat()*(ui->numeroPasos3->toPlainText()).toInt();l=l-(ui->pasoCriterios3->toPlainText()).toFloat()){

                    ui->criterioCont->setText(QString::number(l));

                    for(float m=PredicInicial;m>PredicInicial-(ui->pasoCriterios4->toPlainText()).toFloat()*(ui->numeroPasos4->toPlainText()).toInt();m=m-(ui->pasoCriterios4->toPlainText()).toFloat()){

                        ui->criterioPredic->setText(QString::number(m));

                        if(ui->ListaModelo->currentIndex()==0 ){ //HoG con RF

                            std::string Clasificador = "HoGRF";

                            std::vector< cv::Mat > Candidatos;
                            std::vector< cv::Mat > GradientesImagenes;

                            std::cout << "INICIO PROCESO EXTRACCION DE CARACTERISTICAS A LOS CANDIDATOS DE LA ESCENA... \n \n";

                            std::cout << "1. Lectura imagenes de los candidatos \n \n" ;

                            Detector_Personas::CargarImagenesEscena(PATH_CANDIDATOS, Candidatos);

                            std::cout << "2. Extraccion caracteristicas (HoG) a las imagenes de los candidatos (visualizacion)" << endl;

                            Detector_Personas::CaracteristicasHoG(Candidatos, GradientesImagenes, TAMANO_IMAGEN);

                            std::cout << "3. Creacion ficheros con descriptores \n";
                            std::cout << "Tamaño matriz caracteristicas Filas(numero de imagenes): " << GradientesImagenes.size() << " y Columnas(numero de caracteristicas): " << GradientesImagenes[0].rows << "\n \n";

                            Detector_Personas::CreacionFicherosPythonEscena(GradientesImagenes, "Escena",Clasificador);

                            Detector_Personas::LlamarPythonHOGRFEscena("Escena",Clasificador,0,tipoBarrido);
                        }

                        else if(ui->ListaModelo->currentIndex()==1 ){ //HoG con SVM
                            std::string Clasificador = "HoGSVM";

                            std::vector< cv::Mat > Candidatos;
                            std::vector< cv::Mat > GradientesImagenes;

                            std::cout << "INICIO PROCESO EXTRACCION DE CARACTERISTICAS A LOS CANDIDATOS DE LA ESCENA... \n \n";

                            std::cout << "1. Lectura imagenes de los candidatos \n \n" ;

                            Detector_Personas::CargarImagenesEscena(PATH_CANDIDATOS, Candidatos);

                            std::cout << "2. Extraccion caracteristicas (HoG) a las imagenes de los candidatos (visualizacion)" << endl;

                            Detector_Personas::CaracteristicasHoG(Candidatos, GradientesImagenes, TAMANO_IMAGEN);

                            std::cout << "3. Creacion ficheros con descriptores \n";
                            std::cout << "Tamaño matriz caracteristicas Filas(numero de imagenes): " << GradientesImagenes.size() << " y Columnas(numero de caracteristicas): " << GradientesImagenes[0].rows << "\n \n";

                            Detector_Personas::CreacionFicherosPythonEscena(GradientesImagenes, "Escena",Clasificador);

                            Detector_Personas::LlamarPythonHOGSVMEscena("Escena",Clasificador,0,tipoBarrido);
                        }

                        else if(ui->ListaModelo->currentIndex()==2 ){ //SIFT FV RF
                            std::string Clasificador = "SIFTFVRF";
                            Detector_Personas::LlamarPythonSIFTFVRFEscena("Escena", Clasificador,0,tipoBarrido);
                        }

                        else if(ui->ListaModelo->currentIndex()==3 ){ //SIFT FV SVM
                            std::string Clasificador = "SIFTFVSVM";
                            Detector_Personas::LlamarPythonSIFTFVSVMEscena("Escena", Clasificador,0,tipoBarrido);
                        }

                        else if(ui->ListaModelo->currentIndex()==4 ){ //ANN
                            Detector_Personas::LlamarPythonANNEscena(0);
                        }

                        else if(ui->ListaModelo->currentIndex()==5 ){ //CNN
                            Detector_Personas::LlamarPythonCNNEscena(0);
                        }

                    }
                }
            }
        }
    }

    ui->criterioNMS->setText(QString::number(NMSInicial));
    ui->criterioDif->setText(QString::number(DIFInicial));
    ui->criterioCont->setText(QString::number(ContInicial));
    ui->criterioPredic->setText(QString::number(PredicInicial));
}

void Detector_Personas::on_iniciarTest_clicked(){
    std::clock_t start;
    double duration=0;

    start = std::clock();

    if(ui->modelos->currentIndex()==0 ){ //HoG con RF

        std::string Clasificador = "HoGRF";

        std::vector< cv::Mat > ImagenesPositivas;
        std::vector< cv::Mat > ImagenesNegativas;
        std::vector< cv::Mat > GradientesImagenes;
        std::vector< int > Etiquetas;

        std::string dirImagPos = DIR_BANCOS_LECTURA;
        std::string dirImagNeg = DIR_BANCOS_LECTURA;

        dirImagPos += ui->bancosTest->currentText().toStdString();
        dirImagPos += "\\Prueba\\Persona";

        dirImagNeg += ui->bancosTest->currentText().toStdString();
        dirImagNeg += "\\Prueba\\No_Persona";

        std::cout << "INICIO PROCESO EXTRACCION DE CARACTERISTICAS ENTRENAMIENTO... \n \n";

        std::cout << "1. Lectura imagenes positivas y asignacion de label (+1) \n \n" ;

        Detector_Personas::CargarImagenes(dirImagPos, ImagenesPositivas,1,1);
        Etiquetas.assign(ImagenesPositivas.size(), +1);

        std::cout << "2. Lectura imagenes negativas y asignacion de label (-1) \n \n ";
        Detector_Personas::CargarImagenes(dirImagNeg, ImagenesNegativas,0,1);
        Etiquetas.insert(Etiquetas.end(), ImagenesNegativas.size(), -1);

        std::cout << "3. Extraccion caracteristicas (HoG) a las imagenes positivas (visualizacion) \n" ;

        Detector_Personas::CaracteristicasHoG(ImagenesPositivas, GradientesImagenes, TAMANO_IMAGEN);

        std::cout << "4. Extraccion caracteristicas (HoG) a las imagenes negativas (visualizacion) \n";
        Detector_Personas::CaracteristicasHoG(ImagenesNegativas, GradientesImagenes, TAMANO_IMAGEN);

        std::cout << "5. Creacion ficheros con descriptores y labels \n";
        std::cout << "Tamaño matriz caracteristicas Filas(numero de imagenes): " << GradientesImagenes.size() << " y Columnas(numero de caracteristicas): " << GradientesImagenes[0].rows << "\n \n";

        int numImagenes = GradientesImagenes.size();

        Detector_Personas::CreacionFicherosPython(GradientesImagenes, Etiquetas, "Prueba",Clasificador);

        Detector_Personas::LlamarPythonHOGRF(numImagenes,"Prueba",Clasificador);

        duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

        ui->tiempoTest->setText(QString::number(duration));
    }

    else if(ui->modelos->currentIndex()==1 ){ //HoG con SVM

        std::string Clasificador = "HoGSVM";

        std::vector< cv::Mat > ImagenesPositivas;
        std::vector< cv::Mat > ImagenesNegativas;
        std::vector< cv::Mat > GradientesImagenes;
        std::vector< int > Etiquetas;

        std::string dirImagPos = DIR_BANCOS_LECTURA;
        std::string dirImagNeg = DIR_BANCOS_LECTURA;

        dirImagPos += ui->bancosTest->currentText().toStdString();
        dirImagPos += "\\Prueba\\Persona";

        dirImagNeg += ui->bancosTest->currentText().toStdString();
        dirImagNeg += "\\Prueba\\No_Persona";

        std::cout << "INICIO PROCESO EXTRACCION DE CARACTERISTICAS ENTRENAMIENTO... \n \n";

        std::cout << "1. Lectura imagenes positivas y asignacion de label (+1) \n \n" ;

        Detector_Personas::CargarImagenes(dirImagPos, ImagenesPositivas,1,1);
        Etiquetas.assign(ImagenesPositivas.size(), +1);

        std::cout << "2. Lectura imagenes negativas y asignacion de label (-1) \n \n ";
        Detector_Personas::CargarImagenes(dirImagNeg, ImagenesNegativas,0,1);
        Etiquetas.insert(Etiquetas.end(), ImagenesNegativas.size(), -1);

        std::cout << "3. Extraccion caracteristicas (HoG) a las imagenes positivas (visualizacion) \n" ;

        Detector_Personas::CaracteristicasHoG(ImagenesPositivas, GradientesImagenes, TAMANO_IMAGEN);

        std::cout << "4. Extraccion caracteristicas (HoG) a las imagenes negativas (visualizacion) \n";
        Detector_Personas::CaracteristicasHoG(ImagenesNegativas, GradientesImagenes, TAMANO_IMAGEN);

        std::cout << "5. Creacion ficheros con descriptores y labels \n";
        std::cout << "Tamaño matriz caracteristicas Filas(numero de imagenes): " << GradientesImagenes.size() << " y Columnas(numero de caracteristicas): " << GradientesImagenes[0].rows << "\n \n";

        int numImagenes = GradientesImagenes.size();

        Detector_Personas::CreacionFicherosPython(GradientesImagenes, Etiquetas, "Prueba",Clasificador);

        Detector_Personas::LlamarPythonHOGSVM(numImagenes,"Prueba",Clasificador);

        duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

        ui->tiempoTest->setText(QString::number(duration));
    }

    else if(ui->modelos->currentIndex()==2 ){ //SIFT FV RF

        std::string Clasificador = "SIFTFVRF";
        Detector_Personas::LlamarPythonSIFTFVRF("Prueba",Clasificador);
        duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

        ui->tiempoTest->setText(QString::number(duration));
    }

    else if(ui->modelos->currentIndex()==3 ){ //SIFT FV SVM
        std::string Clasificador = "SIFTFVSVM";
        Detector_Personas::LlamarPythonSIFTFVSVM("Prueba",Clasificador);
        duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

        ui->tiempoTest->setText(QString::number(duration));
    }

    else if(ui->modelos->currentIndex()==4 ){ //ANN
        std::string Clasificador = "ANN";
        Detector_Personas::TestANN();
        ui->tiempoTest->setText(QString::number(duration));
    }

    else if(ui->modelos->currentIndex()==5 ){ //CNN
        std::string Clasificador = "CNN";
        Detector_Personas::TestCNN();
        ui->tiempoTest->setText(QString::number(duration));
    }
}
