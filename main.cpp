#include <stdlib.h>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#define CVUI_IMPLEMENTATION
#include "cvui.h"
#define WINDOW_NAME "CVUI Test"

#include "ioh.hpp"
#include "sad.hpp"
#include "ncc.hpp"
#include "mrf.hpp"

double countTime()
{
    return static_cast<double>(clock());
}

int main(int argc, char const *argv[])
{
    // parse args
    if (argc != 5 && argc != 6)
    {
        std::cout << "usage: main <im0.png> <im1.png> <ndisp> <outdir> <method>" << std::endl;
        std::cout << "method field is optinal, default: Blief Prapagation" << std::endl;
        return -1;
    }
    std::string im0 = std::string(argv[1]);
    std::string im1 = std::string(argv[2]);
    int ndisp = atoi(argv[3]);
    std::string outdir = std::string(argv[4]);
    std::string method;
    if (argc == 5)
        method = "BP";
    else
        method = argv[5];

    cvui::init(WINDOW_NAME);
    // read param (no use now) and images
    IOHelper ioh;
    ioh.setUp(im0, im1, ndisp, outdir);
    cv::Mat leftMat(3, 3, CV_32F), rightMat(3, 3, CV_32F);
    cv::Mat cameraMats[2] = {leftMat, rightMat};
    ioh.readCalib(cameraMats);

    cv::Mat leftImg, rightImg;
    ioh.readImage(leftImg, rightImg);
    cv::namedWindow("debug", cv::WINDOW_AUTOSIZE);
    cv::imshow("debug", leftImg);
    cv::waitKey(0);
    // matching
    std::cout << "use method: " << method << std::endl;
    std::cout << "start matching..." << std::endl;
    cv::Mat disparity;
    const double beginTime = countTime();
    if (method == "SAD")
    {
        SAD matcher(20, ndisp);
        disparity = matcher.do_match(leftImg, rightImg);
    }
    else if (method == "NCC")
    {
        NCC matcher(20, ndisp);
        disparity = matcher.do_match(leftImg, rightImg);
    }
    else if (method == "BP"){
        disparity = do_match(leftImg, rightImg, 20, 10, ndisp, 2);
    }
    const double endTime = countTime();
    cout << "cost time: " << (endTime - beginTime) / CLOCKS_PER_SEC << endl;

    cv::imshow("debug", disparity);
    cv::imwrite(method + ".png", disparity);
    cv::waitKey(0);
    cv::destroyWindow("debug");
    return 0;
}
