#ifndef ANNTRAIN_H
#define ANNTRAIN_H

#include <QDialog>

namespace Ui {
class ANNTrain;
}

class ANNTrain : public QDialog
{
    Q_OBJECT

public:
    explicit ANNTrain(QWidget *parent = 0);
    ~ANNTrain();

private slots:
    void on_buttonBox_accepted();

private:
    Ui::ANNTrain *ui;
};

#endif // ANNTRAIN_H
