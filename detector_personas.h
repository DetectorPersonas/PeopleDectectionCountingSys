#ifndef DETECTOR_PERSONAS_H
#define DETECTOR_PERSONAS_H

#include <QMainWindow>

//Librerias C++
#include <iostream>
#include <string>
#include <Windows.h>
#include <cstdio>
#include <ctime>
#include <math.h>
#include <direct.h>
#include <stdio.h>

//Librerias OpenCV
#include <opencv2/opencv.hpp>
#include "opencv2\highgui.hpp"
#include "opencv2\imgproc.hpp"
#include "opencv2\ximgproc.hpp"

///////////////////////////////////////////////////////////////////////////////////////
namespace Ui {
    class Detector_Personas;
}
///////////////////////////////////////////////////////////////////////////////////////

class Detector_Personas : public QMainWindow{
    Q_OBJECT

public:
    explicit Detector_Personas(QWidget *parent = 0);
    ~Detector_Personas();

private slots:
    void on_btnEntrenar_clicked();

    void on_ListaBancos_currentIndexChanged(int index);

    void on_abrirFrame_clicked();

    void on_iniciarBusqueda_clicked();

    void on_iniciarReporte_clicked();


    void on_iniciarTest_clicked();

    void on_bancosTrain_currentIndexChanged(int index);

    void on_bancosTest_currentIndexChanged(int index);

private:
    Ui::Detector_Personas *ui;

    QImage Detector_Personas::convertOpenCVMatToQtQImage(cv::Mat mat);

    //Prototipos funciones HoG entrenamiento
    void Detector_Personas::CargarImagenes(std::string Directorio, std::vector<cv::Mat>& Imagenes,int Pos, int Prueba);
    std::vector<std::string> Detector_Personas::ArchivosEnDirectorio(std::string Directorio);
    void Detector_Personas::CaracteristicasHoG(const std::vector< cv::Mat > & Imagenes, std::vector< cv::Mat > & Gradientes, const cv::Size & Tamano);
    void Detector_Personas::CargarImagenesEscena(std::string Directorio, std::vector<cv::Mat>& Imagenes);
    cv::Mat Detector_Personas::VisualizacionHoG(const cv::Mat& color_origImg, std::vector<float>& descriptorValues, const cv::Size & size);
    void Detector_Personas::CreacionFicherosPython(std::vector< cv::Mat > & Gradientes, std::vector< int > Etiquetas, std::string Tipo, std::string Classifier);
    void Detector_Personas::CreacionFicherosPythonEscena(std::vector< cv::Mat > & Gradientes, std::string Tipo, std::string Classifier);
    void Detector_Personas::LlamarPythonHOGRF(int numImagenes, std::string Tipo, std::string Classifier);
    void Detector_Personas::LlamarPythonHOGRFEscena(std::string Tipo, std::string Classifier, int Visua, int Barrido);
    void Detector_Personas::LlamarPythonHOGSVMEscena(std::string Tipo, std::string Classifier, int Visua, int Barrido);
    void Detector_Personas::LlamarPythonHOGSVM(int numImagenes, std::string Tipo, std::string Classifier);
    void Detector_Personas::LlamarPythonSIFTFVRF(std::string Tipo, std::string Classifier);
    void Detector_Personas::LlamarPythonSIFTFVSVM(std::string Tipo, std::string Classifier);
    void Detector_Personas::LlamarPythonSIFTFVRFEscena(std::string Tipo, std::string Classifier, int Visua, int Barrido);
    void Detector_Personas::LlamarPythonSIFTFVSVMEscena(std::string Tipo, std::string Classifier, int Visua, int Barrido);
    void Detector_Personas::LlamarPythonANN(std::string Tipo, std::string Classifier);
    void Detector_Personas::LlamarPythonANNEscena(int Visua);
    void Detector_Personas::LlamarPythonCNNEscena(int Visua);
    void Detector_Personas::LlamarPythonCNN(std::string Tipo, std::string Clasificador);
    void Detector_Personas::BarridoExhaustivo();
    void Detector_Personas::BarridoBordes();
    void Detector_Personas::TestANN();
    void Detector_Personas::TestCNN();
    void Detector_Personas::BarridoBordesRegiones();
    void Detector_Personas::BarridoExhaustivoRegiones();
};

#endif // DETECTOR_PERSONAS_H
