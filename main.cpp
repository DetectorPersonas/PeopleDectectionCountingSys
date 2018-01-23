#include "detector_personas.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    Detector_Personas w;

    w.show();

    return a.exec();
}
