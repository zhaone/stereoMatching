#include<stdlib.h>
#include<string>
#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include "ioh.hpp"

int main(int argc, char const *argv[])
{
    // parse args
    if (argc != 5){
        std::cout << "usage: main <im0.png> <im1.png> <ndisp> <outdir>" << std::endl;
        return -1;
    }
    std::string im0 = std::string(argv[1]);
    std::string im1 = std::string(argv[2]);
    int ndisp = atoi(argv[3]);
    std::string outdir = std::string(argv[4]);

    IOHelper ioh;
    ioh.setUp(im0, im1, ndisp, outdir);
    cv::Mat leftMat(3,3, CV_32F), rightMat(3,3, CV_32F);
    cv::Mat cameraMats[2] = {leftMat, rightMat};
    ioh.readCalib(cameraMats);
    std::cout << "left camera:" << leftMat << std::endl
              << "right camera:" << rightMat << std::endl;
    cv::Mat leftImg, rightImg;
    ioh.readImage(leftImg, rightImg);
    cv::namedWindow("debug", cv::WINDOW_AUTOSIZE);
    cv::imshow("debug", leftImg);
    cv::waitKey(0);
    cv::destroyWindow("debug");
    /* 
    1. read image & param
    2. rectify
    3. match
     */
    return 0;
}
