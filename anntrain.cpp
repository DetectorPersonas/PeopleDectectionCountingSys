#include "anntrain.h"
#include "ui_anntrain.h"

ANNTrain::ANNTrain(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::ANNTrain)
{
    ui->setupUi(this);
}

ANNTrain::~ANNTrain()
{
    delete ui;
}

void ANNTrain::on_buttonBox_accepted()
{

}
