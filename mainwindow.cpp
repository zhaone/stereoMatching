#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "util.h"
#include "sad.h"
#include "ncc.h"
#include "bp.h"
#include "mbp.h"

#include <QFileDialog>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_slb_clicked()
{
    leftImgPath = QFileDialog::getOpenFileName(NULL, "select left camera image", "./img/Teddy");
    ui->leftImagePath_label->setText(leftImgPath);
}

void MainWindow::on_srb_clicked()
{
    rightImgPath = QFileDialog::getOpenFileName(NULL, "select right camera image", "./img/Teddy");
    ui->rightImagePath_label->setText(rightImgPath);
}


void MainWindow::on_startMatching_button_clicked()
{
    cv::Mat leftImg = cv::imread(leftImgPath.toStdString(), 0);
    cv::Mat rightImg = cv::imread(rightImgPath.toStdString(), 0);
    int windowSize = ui->windowSize_spinBox->value();
    int disp = ui->maxDisp_spinBox->value();
    int radius = windowSize / 2;
    int iter = ui->inter_spinBox->value();
    double smoothCost = ui->doubleSpinBox_sc->value();

    ui->lcdNumber_height->display(leftImg.rows);
    ui->lcdNumber_width->display(leftImg.cols);

    cv::Mat disparity;
    const double beginTime = countTime();
    if(ui->al_sad->isChecked())
    {
        alType = "SAD";
        SAD matcher(radius, disp);
        disparity = matcher.do_match(leftImg, rightImg);
    }
    else if(ui->al_ncc->isChecked())
    {
        alType = "NCC";
        NCC matcher(radius, disp);
        disparity = matcher.do_match(leftImg, rightImg);
    }
    else if(ui->al_bm->isChecked())
    {
        alType = "BP";
        BP matcher(leftImg, rightImg, disp, smoothCost, 2 * float(disp), iter);
        disparity = matcher.do_match();
    }
    else if(ui->al_bpm->isChecked())
    {
        alType = "BPM";
        MBP matcher(leftImg, rightImg, disp, smoothCost, 2 * float(disp), iter);
        disparity = matcher.do_match();
    }
    const double endTime = countTime();
    const double cost_time = (endTime - beginTime) / CLOCKS_PER_SEC / 10;
    std::cout << "cost time: " << cost_time << "s" << std::endl;
    ui->lcdNumber_cost_time->display(cost_time);
    showDispMap(alType.toStdString()+"_left_disp", disparity, disp, false);
}
