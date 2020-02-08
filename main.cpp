#include <stdlib.h>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "ioh.hpp"
#include "sad.hpp"
#include "ncc.hpp"
#include "bp.hpp"
#include "mbp.hpp"

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
    std::string outPath = std::string(argv[4]);
    std::string method;
    if (argc == 5)
        method = "MBP";
    else
        method = argv[5];
    IOHelper ioh;
    ioh.setUp(im0, im1, ndisp, outPath);
    // cv::Mat leftMat(3, 3, CV_32F), rightMat(3, 3, CV_32F);
    // cv::Mat cameraMats[2] = {leftMat, rightMat};
    // ioh.readCalib(cameraMats);

    cv::Mat leftImg, rightImg;
    ioh.readImage(leftImg, rightImg);

    // cv::namedWindow("debug", cv::WINDOW_AUTOSIZE);
    // cv::imshow("debug", leftImg);
    // cv::waitKey(0);

    // matching
    std::cout << "use method: " << method << std::endl;
    std::cout << "start matching..." << std::endl;
    cv::Mat disparity;
    clock_t beginTime = clock();
    if (method == "SAD")
    {
        SAD matcher(2, ndisp);
        disparity = matcher.do_match(leftImg, rightImg);
    }
    else if (method == "NCC")
    {
        NCC matcher(3, ndisp);
        disparity = matcher.do_match(leftImg, rightImg);
    }
    else if (method == "BP")
    {
        BP matcher(leftImg, rightImg, ndisp, 1, 2 * float(ndisp), 5);
        disparity = matcher.do_match();
    }
    else if (method == "MBP")
    {
        MBP matcher(leftImg, rightImg, ndisp, 1, 2*float(ndisp), 5);
        disparity = matcher.do_match();
    }
    clock_t endTime = clock();
    cout << "method: " << method << " ,cost time: " << (double)(endTime - beginTime) / CLOCKS_PER_SEC << endl;

    // cv::imshow("debug", disparity);
    // for png visualization
    // disparity *= 65535 / ndisp;
    cv::imwrite(outPath, disparity);
    // cv::waitKey(0);
    // cv::destroyWindow("debug");
    return 0;
}
