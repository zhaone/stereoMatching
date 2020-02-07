#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void on_slb_clicked();

    void on_srb_clicked();

    void on_startMatching_button_clicked();

private:
    Ui::MainWindow *ui;

    QString leftImgPath="";
    QString rightImgPath="";
    QString alType="SAD";
};

#endif // MAINWINDOW_H
