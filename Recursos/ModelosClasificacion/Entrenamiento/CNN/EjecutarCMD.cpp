#include <iostream>
#include <windows.h>
#include <string>

using namespace std;

#define CARPETA_IMAGENES "/Users/user/Desktop/Recursos/BancosImagenes/Daimler/Entrenamiento/"
#define GRAFICA_ENTRENADA "/Users/user/Desktop/Recursos/ModelosClasificacion/ModelosEntrenados/CNN/"
#define CLASES "/Users/user/Desktop/Recursos/ModelosClasificacion/ModelosEntrenados/CNN/"
#define RETRAINLOGS_DIR "/Users/user/Desktop/Recursos/ModelosClasificacion/ModelosEntrenados/CNN/"
#define BOTTLENECK_DIR "/Users/user/Desktop/Recursos/ModelosClasificacion/ModelosEntrenados/CNN/"
#define TRAININGSTEPS 4000
#define LEARNINGRATE 0.01
#define BATCHSIZE 100
#define BANCOIMAGENES "Daimler/"

int main() {
	string filename = "C:/Users/user/Desktop/Recursos/ModelosClasificacion/Entrenamiento/CNN/trainingCNN.py";
	string command = "python ";
	command += filename;

	string dirimag = " --image_dir ";
	string outGraph = " --output_graph ";
	string outLabels = " --output_labels ";
	string summ = " --summaries_dir ";
	string bottle = " --bottleneck_dir ";
	string trainSteps = " --how_many_training_steps ";
	string learningRate = " --learning_rate ";
	string batchSize = " --train_batch_size ";
	string id = " --id ";

	dirimag += CARPETA_IMAGENES;
	command += dirimag;
	outGraph += GRAFICA_ENTRENADA;
	outGraph += BANCOIMAGENES;
	outGraph += "output_graph.pb";
	command += outGraph;
	outLabels += CLASES;
	outLabels += BANCOIMAGENES;
	outLabels += "output_labels.txt";
	command += outLabels;
	summ += RETRAINLOGS_DIR;
	summ += BANCOIMAGENES;
	summ += "logs";
	command += summ;
	bottle += BOTTLENECK_DIR;
	bottle += BANCOIMAGENES;
	bottle += "bottleneck";
	command += bottle;
	trainSteps += to_string(TRAININGSTEPS);
	command += trainSteps;
	learningRate += to_string(LEARNINGRATE);
	command += learningRate;
	batchSize += to_string(BATCHSIZE);
	command += batchSize;
	id += BANCOIMAGENES;
	command += id;

	system(command.c_str());
	Sleep(50000);
}